"""Materials Project adapter for the provider-neutral crystal search.

Wraps :class:`~adam_identification.database.apis.materials_project.MaterialsProjectClient` so
callers that only depend on
:class:`~adam_identification.database.crystal_provider.CrystalProvider` (e.g. a retrieval
layer that also talks to another crystal database) can use Materials Project
the same way as any other crystal-structure source, without any change to
``MaterialsProjectClient`` itself.

Materials Project's formula search already returns a full, hydrated
``Material`` for every result. This adapter's :meth:`hydrate` is therefore a
local lookup of a ``Material`` already fetched by a preceding
:meth:`search_candidates` call on the same instance, not a second network
request. If the requested candidate was not already fetched by this instance
(e.g. after a resumed session), :meth:`hydrate` falls back to one
direct-by-ID request.

Cross-references:
    - ``adam_identification.database.apis.materials_project`` — the wrapped client.
    - ``adam_identification.database.crystal_provider`` — ``CrystalCandidate``, ``CrystalProvider``.
"""

from __future__ import annotations

from adam_identification.database.crystal_provider import CrystalCandidate, CrystalSearchResult
from adam_identification.database.materials_project import MaterialsProjectClient
from adam_identification.exceptions import DatabaseAPIError
from adam_identification.models import (
    CrystalStructure,
    Material,
    MaterialSource,
    MaterialsProjectProperties,
)


class MaterialsProjectCrystalProvider:
    """Adapts :class:`MaterialsProjectClient` to the ``CrystalProvider`` protocol.

    Args:
        client: The ``MaterialsProjectClient`` to wrap. Its lifecycle
            (construction, credentials) is the caller's responsibility — this
            adapter never constructs or closes one itself.
    """

    source: MaterialSource = MaterialSource.MATERIALS_PROJECT

    def __init__(self, client: MaterialsProjectClient) -> None:
        self._client = client
        self._material_cache: dict[str, Material] = {}

    def search_candidates(self, formula: str, max_results: int) -> CrystalSearchResult:
        """Return up to ``max_results`` candidates for ``formula``.

        Delegates to :meth:`MaterialsProjectClient.search_by_formula`, which
        already returns fully hydrated materials, and caches each one by its
        MP ID so a later :meth:`hydrate` call for the same result is a local
        lookup rather than a second request.

        Args:
            formula: Chemical formula in any common notation, e.g. ``"TiO2"``.
            max_results: Maximum number of candidates to request and return.

        Returns:
            :class:`~adam_identification.database.crystal_provider.CrystalSearchResult`.
            ``total_matches`` is always ``None`` and ``truncated`` is always
            ``False`` — Materials Project's summary search does not report a
            total-match count independent of ``max_results``.

        Raises:
            DatabaseAPIError: On any failure surfaced by the wrapped client.
        """
        materials = self._client.search_by_formula(formula, max_results=max_results)
        candidates: list[CrystalCandidate] = []
        for material in materials:
            candidate = self._candidate_from_material(material)
            self._material_cache[candidate.source_id] = material
            candidates.append(candidate)
        return CrystalSearchResult(candidates=candidates)

    def get_by_id(self, source_id: str) -> Material:
        """Fetch one material directly by its Materials Project ID.

        Args:
            source_id: Materials Project ID, e.g. ``"mp-149"``.

        Returns:
            A mapped ``Material`` instance.

        Raises:
            ~adam_identification.database.exceptions.MaterialNotFoundError: If ``source_id``
                does not exist in Materials Project.
            DatabaseAPIError: On any failure surfaced by the wrapped client.
        """
        material = self._client.get_by_material_id(source_id)
        if material.mp_id is not None:
            self._material_cache[material.mp_id] = material
        return material

    def hydrate(self, candidate: CrystalCandidate) -> Material:
        """Return the full ``Material`` for a previously-searched candidate.

        Returns the cached ``Material`` from the ``search_candidates`` call
        that produced ``candidate`` when available; otherwise falls back to
        one :meth:`get_by_id` request.

        Args:
            candidate: A candidate previously returned by
                :meth:`search_candidates`.

        Returns:
            The hydrated ``Material``.

        Raises:
            DatabaseAPIError: If ``candidate`` was not sourced from Materials
                Project.
            ~adam_identification.database.exceptions.MaterialNotFoundError: If the candidate
                no longer exists and a fallback fetch is required.
        """
        if candidate.source != self.source:
            raise DatabaseAPIError(
                f"Cannot hydrate a {candidate.source!r} candidate with "
                f"MaterialsProjectCrystalProvider (source={self.source!r})."
            )
        cached = self._material_cache.get(candidate.source_id)
        if cached is not None:
            return cached
        return self.get_by_id(candidate.source_id)

    def _candidate_from_material(self, material: Material) -> CrystalCandidate:
        if material.mp_id is None:
            raise DatabaseAPIError("Materials Project material is missing its mp_id.")
        structure = material.structure
        if not isinstance(structure, CrystalStructure):
            raise DatabaseAPIError(
                f"Materials Project material {material.mp_id!r} has no crystal structure."
            )
        props = material.get_properties(MaterialSource.MATERIALS_PROJECT)
        energy_above_hull: float | None = None
        is_metal: bool | None = None
        if isinstance(props, MaterialsProjectProperties):
            energy_above_hull = props.energy_above_hull
            is_metal = props.is_metal
        return CrystalCandidate(
            source=MaterialSource.MATERIALS_PROJECT,
            source_id=material.mp_id,
            structure_ref=material.mp_id,
            formula=material.chemical_formula,
            space_group=structure.space_group,
            crystal_system=structure.crystal_system,
            nsites=structure.nsites,
            energy_above_hull_ev_per_atom=energy_above_hull,
            is_metal=is_metal,
        )


__all__ = ["MaterialsProjectCrystalProvider"]
