"""Rewrite LLM follow-up queries from candidate metadata.

The selection prompts still ask the model for ``suggested_candidates`` with an
``index`` and a free-text ``suggested_query``. This module keeps the model's
index and rebuilds the query string from the corresponding candidate row so
users never see database IDs, and so crystal/molecule follow-ups use a stable
pattern.

Used by :class:`~adam_identification.identifier.MaterialIdentifier` after
:func:`~adam_identification._phase_lookup.parse_phase_selection_response` and
:func:`~adam_identification._phase_lookup.parse_molecule_selection_response`.

Cross-references:
    - ``adam_identification._phase_lookup`` — candidate serialization and parsers.
    - ``adam_identification.identifier`` — production agent.
"""

from __future__ import annotations

from typing import Any, Literal

# Conventional names that uniquely identify a (formula, space group) pair.
# Keys use :func:`_norm_formula` and :func:`_norm_space_group`.
_CRYSTAL_FOLLOWUP_ALIASES: dict[tuple[str, str], str] = {
    ("tio2", "p42/mnm"): "rutile TiO2",
    ("tio2", "i41/amd"): "anatase TiO2",
    ("tio2", "pbca"): "brookite TiO2",
    ("c", "fd3m"): "diamond",
    ("bn", "p63/mmc"): "h-BN P6_3/mmc",
    ("bn", "p6m2"): "h-BN P-6m2",
}


def _norm_formula(formula: str) -> str:
    """Lowercase formula with whitespace removed."""
    return formula.strip().lower().replace(" ", "")


def _norm_space_group(space_group: str) -> str:
    """Lowercase Hermann–Mauguin symbol, ignoring spaces, underscores, and hyphens."""
    return space_group.strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def _crystal_followup_query(candidate: dict[str, Any]) -> str | None:
    """Build a resume query for one Materials Project candidate row."""
    formula = str(candidate.get("formula") or "").strip()
    space_group = str(candidate.get("space_group") or "").strip()
    if formula and space_group:
        alias = _CRYSTAL_FOLLOWUP_ALIASES.get(
            (_norm_formula(formula), _norm_space_group(space_group))
        )
        if alias:
            return alias
        return f"{formula} {space_group}"
    return formula or space_group or None


def _molecule_followup_query(candidate: dict[str, Any]) -> str | None:
    """Build a resume query from the PubChem candidate display name."""
    name = str(candidate.get("name") or "").strip()
    return name or None


def _suggestion_index(item: Any) -> int | None:
    """Return a candidate list index from one LLM suggestion object."""
    if not isinstance(item, dict):
        return None
    raw = item.get("index")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def format_followup_queries(
    suggestions: list[Any],
    candidates: list[dict[str, Any]],
    *,
    domain: Literal["crystal", "molecule"],
) -> list[dict[str, Any]]:
    """Replace LLM ``suggested_query`` strings with metadata-derived queries.

    Keeps each suggestion's ``index``. Drops entries with a missing or
    out-of-range index, or with no usable formula/space group/name. Never uses
    Materials Project ids or PubChem CIDs as the query.

    Args:
        suggestions: Raw ``suggested_candidates`` list from the LLM parser.
        candidates: Serialized candidate rows shown to the model (must include
            ``formula`` and ``space_group`` for crystals, ``name`` for molecules).
        domain: ``"crystal"`` or ``"molecule"``.

    Returns:
        List of ``{"index": int, "suggested_query": str}`` dicts.
    """
    formatted: list[dict[str, Any]] = []
    for item in suggestions:
        idx = _suggestion_index(item)
        if idx is None or idx < 0 or idx >= len(candidates):
            continue
        candidate = candidates[idx]
        if domain == "molecule":
            query = _molecule_followup_query(candidate)
        else:
            query = _crystal_followup_query(candidate)
        if not query:
            continue
        formatted.append({"index": idx, "suggested_query": query})
    return formatted
