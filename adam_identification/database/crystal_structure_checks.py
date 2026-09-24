"""Provider-neutral crystal-structure validation helpers for ADaM.

Pure, network-free structure checks that do not depend on any single
database's wire format, so every crystal database client can hold its
mapped structures to the same standard rather than each reimplementing its
own version of the same rule.

Cross-references:
    - ``adam_identification.database.apis.mc3d`` — uses these to map MC3D's raw API
      responses into ``Material``.
    - ``adam_identification.database.apis.materials_project`` — Materials Project's summary
      API already reports the crystal system directly, and its underlying
      pymatgen ``Structure`` objects already refuse to expose a single
      chemical symbol for a disordered site, so it does not currently need
      to call these explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

from adam_identification.exceptions import DatabaseAPIError

_SPACE_GROUP_CRYSTAL_SYSTEMS: tuple[tuple[int, int, str], ...] = (
    (1, 2, "Triclinic"),
    (3, 15, "Monoclinic"),
    (16, 74, "Orthorhombic"),
    (75, 142, "Tetragonal"),
    (143, 167, "Trigonal"),
    (168, 194, "Hexagonal"),
    (195, 230, "Cubic"),
)


def crystal_system_from_space_group_number(space_group_number: int) -> str:
    """Map an International space-group number (1-230) to a crystal system.

    Derived deterministically from the standard space-group number ranges —
    never from an LLM.

    Args:
        space_group_number: International Tables space-group number.

    Returns:
        One of ``"Triclinic"``, ``"Monoclinic"``, ``"Orthorhombic"``,
        ``"Tetragonal"``, ``"Trigonal"``, ``"Hexagonal"``, ``"Cubic"``.

    Raises:
        DatabaseAPIError: If ``space_group_number`` is outside ``[1, 230]``.
    """
    for low, high, name in _SPACE_GROUP_CRYSTAL_SYSTEMS:
        if low <= space_group_number <= high:
            return name
    raise DatabaseAPIError(
        f"Space group number {space_group_number} is outside the valid [1, 230] range."
    )


def assert_ordered_site(
    chemical_symbols: Sequence[str],
    concentration: Sequence[float],
    *,
    site_label: str,
) -> str:
    """Return the single chemical symbol occupying a fully-ordered site.

    Rejects disorder, mixed occupancy, and vacancies — exactly one chemical
    symbol at concentration 1.0 is required. Intended to be called by every
    crystal database client so structures mapped from different databases
    are held to the same standard.

    Args:
        chemical_symbols: Chemical symbol(s) occupying the site.
        concentration: Occupancy fraction per symbol, same length and order
            as ``chemical_symbols``.
        site_label: Human-readable site identifier used in the error
            message (e.g. an OPTIMADE species name or a site index).

    Returns:
        The single chemical symbol occupying the site.

    Raises:
        DatabaseAPIError: If the site is disordered, has mixed occupancy, or
            is not fully occupied by exactly one symbol.
    """
    if len(chemical_symbols) != 1 or len(concentration) != 1 or concentration[0] != 1.0:
        raise DatabaseAPIError(f"Site {site_label!r} is disordered/mixed-occupancy; not supported.")
    return chemical_symbols[0]


__all__ = ["assert_ordered_site", "crystal_system_from_space_group_number"]
