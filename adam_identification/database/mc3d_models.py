"""Private wire models and pure mapping helpers for the MC3D API client.

These Pydantic models validate raw JSON from MC3D's OPTIMADE and MCXD-metadata
HTTP APIs before :class:`~adam_identification.database.apis.mc3d.MC3DClient` maps them into
:class:`~adam_identification.database.crystal_provider.CrystalCandidate` or
:class:`~adam_identification.database.models.Material`. Keeping wire shape and mapping logic
separate from the client's HTTP/caching/retry orchestration keeps each module
a manageable size.

Also holds pure, network-free helpers used only by the MC3D mapping path:
OPTIMADE reduced-formula canonicalization, lattice-parameter computation,
Cartesian-to-fractional coordinate conversion, and OPTIMADE species/site
consistency validation. Space-group-to-crystal-system mapping and the
underlying single-site occupancy check live in
``adam_identification.database.crystal_structure_checks`` instead, since neither depends on
MC3D's wire format.

Cross-references:
    - ``adam_identification.database.apis.mc3d`` — the client that consumes this module.
    - ``adam_identification.database.crystal_structure_checks`` — provider-neutral
      structure-validation helpers shared across database clients.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from adam_identification.database.crystal_structure_checks import assert_ordered_site
from adam_identification.exceptions import DatabaseAPIError

# ── OPTIMADE wire models ────────────────────────────────────────────────────


class OptimadeStructureAttributes(BaseModel):
    """``attributes`` block of one OPTIMADE ``structures`` resource.

    A candidate-search response and a single-structure hydration response
    request different, non-overlapping subsets of these fields, so every
    field here is optional at the wire level —
    :class:`~adam_identification.database.apis.mc3d.MC3DClient` enforces which combination
    is actually required for each call.
    """

    model_config = ConfigDict(extra="ignore")

    chemical_formula_reduced: str | None = None
    nsites: int | None = None
    lattice_vectors: list[list[float]] | None = None
    cartesian_site_positions: list[list[float]] | None = None
    species: list[dict[str, Any]] | None = None
    species_at_sites: list[str] | None = None
    structure_features: list[str] | None = None
    mcloud_mc3d_id: str | None = Field(default=None, alias="_mcloud_mc3d_id")
    mcloud_source_db: str | None = Field(default=None, alias="_mcloud_source_db")
    mcloud_source_db_id: str | None = Field(default=None, alias="_mcloud_source_db_id")
    mcloud_total_energy: float | None = Field(default=None, alias="_mcloud_total_energy")
    mcloud_cell_volume: float | None = Field(default=None, alias="_mcloud_cell_volume")
    mcloud_total_magnetization: float | None = Field(
        default=None, alias="_mcloud_total_magnetization"
    )
    mcloud_absolute_magnetization: float | None = Field(
        default=None, alias="_mcloud_absolute_magnetization"
    )


class OptimadeStructureResource(BaseModel):
    """One entry of an OPTIMADE ``structures`` response's ``data``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    attributes: OptimadeStructureAttributes


class OptimadeMeta(BaseModel):
    """``meta`` block of an OPTIMADE response.

    ``data_returned`` is the filtered-match count and may exceed the number
    of rows actually present in ``data`` — this is expected, not an error.
    ``data_available`` (the whole-database size) is intentionally not
    modelled here since it must never be used to drive pagination.
    """

    model_config = ConfigDict(extra="ignore")

    data_returned: int


class OptimadeSearchResponse(BaseModel):
    """Wire model for a bounded, filtered OPTIMADE ``structures`` search."""

    model_config = ConfigDict(extra="ignore")

    data: list[OptimadeStructureResource]
    meta: OptimadeMeta


class OptimadeHydrateResponse(BaseModel):
    """Wire model for a single-structure OPTIMADE lookup by UUID."""

    model_config = ConfigDict(extra="ignore")

    data: OptimadeStructureResource


# ── MCXD ``core_base`` wire models ──────────────────────────────────────────


class CoreBaseGeneral(BaseModel):
    """``general`` block of a ``core_base`` record."""

    model_config = ConfigDict(extra="ignore")

    bravais_lattice: str | None = None
    formula: str | None = None
    formula_hill: str | None = None
    spacegroup_international: str | None = None
    spacegroup_number: int | None = None
    structure_uuid: str


class CoreBaseValue(BaseModel):
    """A single named property value in ``core_base.properties``."""

    model_config = ConfigDict(extra="ignore")

    uuid: str | None = None
    value: float | None = None


class CoreBaseProperties(BaseModel):
    """``properties`` block of a ``core_base`` record. Any entry may be absent."""

    model_config = ConfigDict(extra="ignore")

    total_energy: CoreBaseValue | None = None
    cell_volume: CoreBaseValue | None = None
    total_magnetization: CoreBaseValue | None = None
    absolute_magnetization: CoreBaseValue | None = None


class CoreBaseProvenanceLink(BaseModel):
    """One entry of ``core_base.provenance_links``."""

    model_config = ConfigDict(extra="ignore")

    label: str
    uuid: str


class CoreBaseSourceInfo(BaseModel):
    """``info`` block of one ``core_base.source`` entry.

    Fields are ``None`` when the API did not report the flag — never coerce
    to ``False`` (mirrors :class:`~adam_identification.database.models.MC3DSourceRecord`).
    """

    model_config = ConfigDict(extra="ignore")

    is_theoretical: bool | None = None
    is_high_pressure: bool | None = None
    is_high_temperature: bool | None = None


class CoreBaseSource(BaseModel):
    """One original-database record cited by a ``core_base`` entry."""

    model_config = ConfigDict(extra="ignore")

    database: str
    id: str
    version: str | None = None
    info: CoreBaseSourceInfo | None = None


class CoreBaseRecord(BaseModel):
    """Wire model for one MC3D ``core_base`` metadata record."""

    model_config = ConfigDict(extra="ignore")

    id: str
    general: CoreBaseGeneral
    properties: CoreBaseProperties = Field(default_factory=CoreBaseProperties)
    provenance_links: list[CoreBaseProvenanceLink] = Field(default_factory=list)
    source: list[CoreBaseSource] = Field(default_factory=list)


# ── Pure structure-math helpers (no network, no wire models) ───────────────


def parse_reduced_formula_counts(formula: str) -> dict[str, int]:
    """Parse a simple (no-parentheses) chemical formula into element counts.

    Rejects fractional/malformed input (e.g. ``"Ti0.5O2"``) by requiring the
    matched element/count tokens to cover the entire string.

    Args:
        formula: Chemical formula string, e.g. ``"TiO2"``.

    Returns:
        Mapping of element symbol to integer count.

    Raises:
        DatabaseAPIError: If ``formula`` is empty or cannot be fully parsed.
    """
    stripped = formula.strip()
    if not stripped:
        raise DatabaseAPIError("Empty chemical formula.")

    counts: dict[str, int] = {}
    matched_len = 0
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", stripped):
        element, digits = match.group(1), match.group(2)
        if not element:
            continue
        matched_len += len(match.group(0))
        counts[element] = counts.get(element, 0) + (int(digits) if digits else 1)

    if not counts or matched_len != len(stripped):
        raise DatabaseAPIError(f"Could not parse chemical formula {formula!r}.")
    return counts


def to_optimade_reduced_formula(formula: str) -> str:
    """Canonicalize *formula* into OPTIMADE's ``chemical_formula_reduced`` form.

    Elements are sorted alphabetically by symbol, counts are divided by their
    GCD, and a count of 1 is omitted — e.g. ``"TiO2"`` -> ``"O2Ti"``,
    ``"Fe2O3"`` -> ``"Fe2O3"``. This is computed deterministically from the
    formula string; it is never derived from LLM-formatted text.

    Args:
        formula: Chemical formula in any common element-order notation.

    Returns:
        The OPTIMADE-reduced, alphabetically-ordered formula string.

    Raises:
        DatabaseAPIError: If ``formula`` cannot be parsed.
    """
    counts = parse_reduced_formula_counts(formula)
    divisor = math.gcd(*counts.values())
    parts: list[str] = []
    for element in sorted(counts):
        n = counts[element] // divisor
        parts.append(element if n == 1 else f"{element}{n}")
    return "".join(parts)


def lattice_parameters(lattice_vectors: list[list[float]]) -> dict[str, float]:
    """Compute lattice lengths and angles from three basis vectors.

    Computed directly from the vectors without rounding — comparison
    tolerance is the responsibility of downstream/benchmark code.

    Args:
        lattice_vectors: 3x3 matrix; rows are the ``a``, ``b``, ``c`` basis
            vectors in Å.

    Returns:
        Dict with keys ``a``, ``b``, ``c`` (Å) and ``alpha``, ``beta``,
        ``gamma`` (degrees).
    """
    vectors = np.asarray(lattice_vectors, dtype=np.float64)
    a_vec, b_vec, c_vec = vectors[0], vectors[1], vectors[2]

    def _angle_deg(u: np.ndarray, v: np.ndarray) -> float:
        cos_theta = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        cos_theta = max(-1.0, min(1.0, cos_theta))  # guard float drift outside acos domain
        return float(np.degrees(np.arccos(cos_theta)))

    return {
        "a": float(np.linalg.norm(a_vec)),
        "b": float(np.linalg.norm(b_vec)),
        "c": float(np.linalg.norm(c_vec)),
        "alpha": _angle_deg(b_vec, c_vec),
        "beta": _angle_deg(a_vec, c_vec),
        "gamma": _angle_deg(a_vec, b_vec),
    }


def cartesian_to_fractional(
    cartesian_positions: list[list[float]],
    lattice_vectors: list[list[float]],
    *,
    round_trip_atol: float = 1e-8,
) -> list[tuple[float, float, float]]:
    """Convert Cartesian site positions (Å) to fractional (reduced) coordinates.

    Uses ``fractional = cartesian @ inverse(lattice)`` with ``lattice`` rows
    as the three basis vectors. Works for non-orthogonal cells. Validates a
    Cartesian round-trip before returning, so a bad inversion never reaches
    :class:`~adam_identification.database.models.CrystalStructure`, then wraps the result
    into ``[0, 1)``.

    Args:
        cartesian_positions: Nx3 Cartesian site positions in Å.
        lattice_vectors: 3x3 matrix; rows are the lattice basis vectors in Å.
        round_trip_atol: Absolute tolerance (Å) for the reconstruction check.

    Returns:
        One ``(x, y, z)`` fractional-coordinate tuple per input position, in
        ``[0, 1)``.

    Raises:
        DatabaseAPIError: If the lattice is non-finite/singular, the inputs
            are malformed, or the round-trip check fails.
    """
    lattice = np.asarray(lattice_vectors, dtype=np.float64)
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)):
        raise DatabaseAPIError("MC3D lattice vectors are not a finite 3x3 matrix.")
    if abs(np.linalg.det(lattice)) < 1e-10:
        raise DatabaseAPIError("MC3D lattice vectors are singular (zero-volume cell).")

    cartesian = np.asarray(cartesian_positions, dtype=np.float64)
    if cartesian.ndim != 2 or cartesian.shape[1] != 3:
        raise DatabaseAPIError("MC3D Cartesian site positions are not an Nx3 array.")

    fractional = cartesian @ np.linalg.inv(lattice)

    reconstructed = fractional @ lattice
    if not np.allclose(reconstructed, cartesian, atol=round_trip_atol):
        raise DatabaseAPIError(
            "Cartesian-to-fractional round-trip exceeded tolerance for an MC3D structure."
        )

    wrapped = np.mod(fractional, 1.0)
    # Floating-point noise around an exact lattice point (e.g. a Cartesian
    # position that is analytically zero but stored as a tiny negative
    # number) wraps to a value extremely close to 1.0, not to 0.0. Snap
    # values within round-trip tolerance of the upper boundary back to 0.0
    # so a real zero-coordinate site is never reported as ~1.0.
    wrapped = np.where(wrapped > 1.0 - round_trip_atol, 0.0, wrapped)
    return [(float(row[0]), float(row[1]), float(row[2])) for row in wrapped]


def validate_species_at_sites(
    species_at_sites: list[str],
    species: list[dict[str, Any]],
) -> dict[str, str]:
    """Validate OPTIMADE species/site consistency and return a name->element map.

    Every site name must resolve to exactly one species entry, and every
    resolved species must be a single chemical symbol at full occupancy
    (see :func:`adam_identification.database.crystal_structure_checks.assert_ordered_site`
    for the underlying, database-neutral occupancy rule).

    Args:
        species_at_sites: Per-site species name, length ``nsites``.
        species: OPTIMADE ``species`` list (raw dicts with ``name``,
            ``chemical_symbols``, ``concentration``).

    Returns:
        Mapping from species name to its single chemical symbol.

    Raises:
        DatabaseAPIError: On a missing/duplicate/disordered species entry.
    """
    by_name: dict[str, dict[str, Any]] = {}
    for entry in species:
        name = entry.get("name")
        if not isinstance(name, str):
            raise DatabaseAPIError("MC3D species entry is missing a name.")
        if name in by_name:
            raise DatabaseAPIError(f"Duplicate MC3D species name {name!r}.")
        by_name[name] = entry

    name_to_element: dict[str, str] = {}
    for name in species_at_sites:
        site_species = by_name.get(name)
        if site_species is None:
            raise DatabaseAPIError(f"MC3D site references undefined species {name!r}.")
        symbols = site_species.get("chemical_symbols") or []
        concentration = site_species.get("concentration") or []
        name_to_element[name] = assert_ordered_site(symbols, concentration, site_label=name)
    return name_to_element


__all__ = [
    "CoreBaseGeneral",
    "CoreBaseProperties",
    "CoreBaseProvenanceLink",
    "CoreBaseRecord",
    "CoreBaseSource",
    "CoreBaseSourceInfo",
    "CoreBaseValue",
    "OptimadeHydrateResponse",
    "OptimadeMeta",
    "OptimadeSearchResponse",
    "OptimadeStructureAttributes",
    "OptimadeStructureResource",
    "cartesian_to_fractional",
    "lattice_parameters",
    "parse_reduced_formula_counts",
    "to_optimade_reduced_formula",
    "validate_species_at_sites",
]
