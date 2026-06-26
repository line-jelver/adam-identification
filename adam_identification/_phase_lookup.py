"""Shared phase-lookup helpers for crystal and molecule identification.

Encapsulates narrow-then-wide MP search limits, candidate serialisation for
the phase-selection prompt, and parsing of LLM selection responses. Also
provides the molecule-side analogue: candidate serialisation and response
parsing for the PubChem multi-candidate → LLM selection step.

Used by :class:`~adam_identification.identifier.MaterialIdentifier`.
"""

from __future__ import annotations

import json
import re
from typing import Any

from adam_identification._confidence import parse_confidence_level
from adam_identification.database.pubchem import MoleculeCandidate
from adam_identification.models import Material, MaterialSource, MaterialsProjectProperties
from adam_identification.llm.json_utils import parse_llm_json_object

INITIAL_MAX_RESULTS = 20
WIDE_MAX_RESULTS = 50


def candidates_to_selection_json(candidates: list[Material]) -> list[dict[str, Any]]:
    """Serialise MP candidates for the phase-selection LLM prompt.

    Args:
        candidates: Ordered Materials Project candidate materials.

    Returns:
        List of compact dicts suitable for ``json.dumps`` into the prompt.
    """
    out: list[dict[str, Any]] = []
    for idx, mat in enumerate(candidates):
        struct = mat.structure
        props = mat.get_properties(MaterialSource.MATERIALS_PROJECT)
        mp_props: dict[str, Any] = {}
        if isinstance(props, MaterialsProjectProperties):
            mp_props = {
                "energy_above_hull": props.energy_above_hull,
                "is_metal": props.is_metal,
            }
        out.append(
            {
                "index": idx,
                "mp_id": mat.mp_id,
                "formula": mat.chemical_formula,
                "space_group": getattr(struct, "space_group", None),
                "crystal_system": getattr(struct, "crystal_system", None),
                "nsites": getattr(struct, "nsites", None),
                **mp_props,
            }
        )
    return out


def single_candidate_selection(
    candidates: list[Material],
    polymorph_name: str | None,
    space_group_hint: str | None,
) -> dict[str, Any] | None:
    """Return an auto-selection dict when no LLM call is required.

    Args:
        candidates: MP candidate materials from the current search pass.
        polymorph_name: Target polymorph hint from formula extraction.
        space_group_hint: Expected space group hint from formula extraction.

    Returns:
        Selection dict when there is exactly one candidate and no polymorph
        constraints; otherwise ``None``.
    """
    if len(candidates) == 1 and polymorph_name is None and space_group_hint is None:
        return {
            "found": True,
            "selected_index": 0,
            "reason": "Single candidate, no polymorph constraint.",
            "confidence": 5,
        }
    return None


def parse_phase_selection_response(
    data: dict[str, Any],
    n_candidates: int,
) -> dict[str, Any]:
    """Normalise an LLM phase-selection JSON object.

    Args:
        data: Parsed LLM response object.
        n_candidates: Number of candidates in the current search pass.

    Returns:
        Dict with keys ``found``, ``selected_index``, ``reason``, ``confidence``.
    """
    found = bool(data.get("found", False))
    selected_index = data.get("selected_index")
    if found and selected_index is not None:
        idx = max(0, min(int(selected_index), n_candidates - 1))
        selected_index = idx
    return {
        "found": found,
        "selected_index": selected_index,
        "reason": str(data.get("reason", "")),
        "confidence": parse_confidence_level(data.get("confidence")),
    }


def parse_llm_json_tolerant(content: str) -> dict[str, Any]:
    """Parse JSON from LLM output, tolerating fences and surrounding prose.

    Args:
        content: Raw LLM output text.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If no valid JSON object can be extracted.
    """
    try:
        return parse_llm_json_object(content)
    except (json.JSONDecodeError, TypeError):
        text = content.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return parse_llm_json_object(text[start : end + 1])
            except (json.JSONDecodeError, TypeError):
                pass
        msg = f"Could not parse JSON object from LLM output: {content[:200]!r}"
        raise ValueError(msg) from None


def parse_formula_counts(formula: str) -> dict[str, int]:
    """Parse a Hill-notation formula into ``{element: count}``.

    Args:
        formula: Chemical formula string, e.g. ``"H2O"``, ``"C2H6O"``.

    Returns:
        Mapping of element symbol to total atom count.
    """
    counts: dict[str, int] = {}
    for match in re.finditer(r"([A-Z][a-z]?)(\d*)", formula):
        elem, num = match.group(1), match.group(2)
        if elem:
            counts[elem] = counts.get(elem, 0) + (int(num) if num else 1)
    return counts


def formulas_match(formula_a: str, formula_b: str) -> bool:
    """Return whether two formula strings represent the same composition.

    Compares by element-count equality, ignoring ordering and notation
    differences (e.g. ``"C2H5OH"`` == ``"C2H6O"``).

    Args:
        formula_a: First formula string.
        formula_b: Second formula string.

    Returns:
        ``True`` when both formulas have identical element counts.
    """
    return parse_formula_counts(formula_a) == parse_formula_counts(formula_b)


def molecule_candidates_to_selection_json(
    candidates: list[MoleculeCandidate],
) -> list[dict[str, Any]]:
    """Serialise PubChem molecule candidates for the selection prompt.

    Args:
        candidates: List of :class:`~adam_identification.database.pubchem.MoleculeCandidate`
            dicts.

    Returns:
        List of dicts with ``index``, ``cid``, ``name``, ``formula``, ``smiles``.
    """
    return [
        {
            "index": i,
            "cid": c["cid"],
            "name": c["name"],
            "formula": c["formula"],
            "smiles": c.get("smiles"),
        }
        for i, c in enumerate(candidates)
    ]


def parse_molecule_selection_response(
    data: dict[str, Any],
    n_candidates: int,
) -> dict[str, Any]:
    """Normalise an LLM molecule-selection JSON object.

    Args:
        data: Parsed LLM response object.
        n_candidates: Number of candidates shown to the LLM.

    Returns:
        Dict with keys ``selected_index``, ``reason``, ``confidence``.
    """
    selected_index = data.get("selected_index")
    idx = max(0, min(int(selected_index), n_candidates - 1)) if selected_index is not None else 0
    return {
        "selected_index": idx,
        "reason": str(data.get("reason", "")),
        "confidence": parse_confidence_level(data.get("confidence")),
    }


def polymorph_not_found_message(
    query: str,
    formula: str,
    polymorph_name: str | None,
    space_group_hint: str | None,
    n_candidates: int,
    reason: str,
) -> str:
    """Build an error message when no matching phase is found.

    Args:
        query: Original material description.
        formula: Extracted chemical formula.
        polymorph_name: Polymorph hint from formula extraction.
        space_group_hint: Space-group hint from formula extraction.
        n_candidates: Number of wide-search candidates examined.
        reason: LLM reason string from the final selection attempt.

    Returns:
        Human-readable error message.
    """
    hint_parts: list[str] = []
    if polymorph_name:
        hint_parts.append(f"polymorph={polymorph_name!r}")
    if space_group_hint:
        hint_parts.append(f"space_group={space_group_hint!r}")
    hint_str = ", ".join(hint_parts) or "no specific polymorph hint"
    return (
        f"Could not find the requested phase for {query!r} "
        f"(formula={formula!r}, {hint_str}) in the Materials Project database. "
        f"Searched {n_candidates} candidates. "
        f"LLM reason: {reason or 'none'}. "
        "Provide a more specific query or a known Materials Project ID."
    )
