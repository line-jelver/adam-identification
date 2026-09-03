"""
Materials Project API client for ADaM.

Wraps the ``mp-api`` client and maps summary documents into the unified
``Material`` schema used by the workflow and benchmarks.

- ``search_by_formula()`` is the primary production path used when only a
  formula is known.
- ``get_by_material_id()`` is benchmark-oriented when a known MP identifier is
  available in ground truth.

Cross-references:
- ``adam_identification.models`` for ``Material`` and related typed models.
- ``adam_identification.exceptions`` for typed database errors.
- ``adam_identification._config`` for Materials Project API key loading.

``MaterialsProjectProperties.method`` is left at ``"unknown"`` (model default).
Per-entry run type (GGA vs GGA+U, etc.) requires ``materials/core``; see
``MaterialsProjectProperties`` docstring for details.
"""

from __future__ import annotations

from typing import Any

from adam_identification._config import ConfigurationError, settings
from adam_identification.database.retry import retry_on_transient_error
from adam_identification.exceptions import DatabaseAPIError, MaterialNotFoundError
from adam_identification.models import (
    AtomicPosition,
    CrystalStructure,
    Material,
    MaterialSource,
    MaterialsProjectProperties,
)

MPRester: Any = None
try:
    from mp_api.client import MPRester  # type: ignore[import-untyped,no-redef]
except Exception:  # ImportError or AttributeError from pyOpenSSL
    MPRester = None

MP_SUMMARY_FIELDS: list[str] = [
    "material_id",
    "formula_pretty",
    "structure",
    "symmetry",
    "nsites",
    "nelements",
    "energy_above_hull",
    "formation_energy_per_atom",
    "band_gap",
    "is_metal",
    "is_gap_direct",
    "ordering",
]


class MaterialsProjectClient:
    """Client for Materials Project summary endpoints.

    Args:
        api_key: Materials Project API key. If ``None``, loaded from
            ``adam_identification._config.settings``.
    """

    def __init__(self, api_key: str | None = None) -> None:
        if MPRester is None:
            raise DatabaseAPIError("mp-api package not installed. Install with: pip install mp-api")

        try:
            resolved_key = api_key or settings.require_materials_project()
        except ConfigurationError as error:
            raise DatabaseAPIError(str(error)) from error

        self._mpr = MPRester(resolved_key)

    def search_by_formula(self, formula: str, max_results: int = 10) -> list[Material]:
        """Search crystal entries by chemical formula.

        Args:
            formula: Formula string, e.g. ``\"Si\"`` or ``\"GaAs\"``.
            max_results: Maximum number of mapped results to return.

        Returns:
            Materials ordered by ``energy_above_hull`` ascending.

        Raises:
            DatabaseAPIError: On API/auth/network errors or malformed responses.
        """
        if max_results <= 0:
            return []

        normalized_formula = formula.strip()
        if not normalized_formula:
            return []

        def _search_once() -> list[Any]:
            try:
                return list(
                    self._mpr.summary.search(
                        formula=normalized_formula,
                        deprecated=False,
                        fields=MP_SUMMARY_FIELDS,
                    )
                )
            except Exception as error:
                raise DatabaseAPIError(
                    f"Materials Project formula search failed for '{normalized_formula}': {error}"
                ) from error

        outcome = retry_on_transient_error(_search_once, label="MaterialsProject")
        docs = outcome.value
        materials = [self._map_summary_doc(doc) for doc in docs]
        materials.sort(key=self._energy_sort_key)
        return materials[:max_results]

    def get_by_material_id(self, material_id: str) -> Material:
        """Fetch one material by MP ID.

        Args:
            material_id: Materials Project ID, e.g. ``\"mp-149\"``.

        Returns:
            A mapped ``Material`` instance.

        Raises:
            MaterialNotFoundError: If no entry is found.
            DatabaseAPIError: On API/auth/network errors or malformed responses.
        """
        normalized_id = material_id.strip()
        if not normalized_id:
            raise MaterialNotFoundError("Empty material ID was provided.")

        def _lookup_once() -> list[Any]:
            try:
                return list(
                    self._mpr.summary.search(
                        material_ids=[normalized_id],
                        deprecated=False,
                        fields=MP_SUMMARY_FIELDS,
                    )
                )
            except Exception as error:
                raise DatabaseAPIError(
                    f"Materials Project ID lookup failed for '{normalized_id}': {error}"
                ) from error

        outcome = retry_on_transient_error(_lookup_once, label="MaterialsProject")
        docs = outcome.value
        if not docs:
            raise MaterialNotFoundError(
                f"No Materials Project entry found for ID '{normalized_id}'."
            )

        return self._map_summary_doc(docs[0])

    def _map_summary_doc(self, doc: Any) -> Material:
        """Map an MP summary document into ``Material``."""
        try:
            structure = doc.structure
            lattice = structure.lattice
            symmetry = doc.symmetry

            crystal_system_obj = getattr(symmetry, "crystal_system", None)
            crystal_system = (
                crystal_system_obj.value
                if crystal_system_obj is not None and hasattr(crystal_system_obj, "value")
                else str(crystal_system_obj)
                if crystal_system_obj is not None
                else None
            )
            if crystal_system is None:
                raise ValueError("Missing symmetry.crystal_system")

            ordering = self._extract_enum_value(getattr(doc, "ordering", None))

            crystal_structure = CrystalStructure(
                unit_cell=[list(row) for row in lattice.matrix.tolist()],
                lattice_parameters={
                    "a": float(lattice.a),
                    "b": float(lattice.b),
                    "c": float(lattice.c),
                    "alpha": float(lattice.alpha),
                    "beta": float(lattice.beta),
                    "gamma": float(lattice.gamma),
                },
                atomic_positions=[
                    AtomicPosition(
                        element=str(site.specie),
                        position=(
                            float(site.frac_coords[0]),
                            float(site.frac_coords[1]),
                            float(site.frac_coords[2]),
                        ),
                    )
                    for site in structure.sites
                ],
                crystal_system=crystal_system,
                space_group=str(getattr(symmetry, "symbol", "")),
                nelements=int(doc.nelements),
                nsites=int(doc.nsites),
            )
        except Exception as error:
            material_id = getattr(doc, "material_id", "<unknown>")
            raise DatabaseAPIError(
                f"Failed to map Materials Project summary doc '{material_id}': {error}"
            ) from error

        return Material(
            name=str(doc.formula_pretty),
            chemical_formula=str(doc.formula_pretty),
            source=MaterialSource.MATERIALS_PROJECT,
            mp_id=str(doc.material_id),
            structure=crystal_structure,
            properties=[
                MaterialsProjectProperties(
                    energy_above_hull=self._maybe_float(getattr(doc, "energy_above_hull", None)),
                    formation_energy_per_atom=self._maybe_float(
                        getattr(doc, "formation_energy_per_atom", None)
                    ),
                    band_gap=self._maybe_float(getattr(doc, "band_gap", None)),
                    is_metal=self._maybe_bool(getattr(doc, "is_metal", None)),
                    is_direct_gap=self._maybe_bool(getattr(doc, "is_gap_direct", None)),
                    magnetic_ordering=ordering,
                )
            ],
        )

    @staticmethod
    def _extract_enum_value(value: Any) -> str | None:
        """Extract enum value as string when present."""
        if value is None:
            return None
        return value.value if hasattr(value, "value") else str(value)

    @staticmethod
    def _maybe_float(value: Any) -> float | None:
        """Convert numeric-like value to ``float`` or return ``None``."""
        if value is None:
            return None
        return float(value)

    @staticmethod
    def _maybe_bool(value: Any) -> bool | None:
        """Convert value to ``bool`` while preserving ``None``."""
        if value is None:
            return None
        return bool(value)

    @staticmethod
    def _energy_sort_key(material: Material) -> float:
        """Sort key helper for ascending energy-above-hull ordering."""
        properties = material.get_properties(MaterialSource.MATERIALS_PROJECT)
        if not isinstance(properties, MaterialsProjectProperties):
            return float("inf")
        if properties.energy_above_hull is None:
            return float("inf")
        return properties.energy_above_hull
