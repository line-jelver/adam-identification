"""Shared phase lookup helpers for material identification (crystals and molecules).

Encapsulates narrow-then-wide MP search limits, candidate serialization for the
phase-selection prompt, and parsing of LLM selection responses (schema with
``decision``/``selection_reason``/``user_message``/``suggested_candidates``/
``needs_review``).  Also provides the molecule-side analogue: candidate serialization
and response parsing for the PubChem multi-candidate → LLM selection step.

Used by :class:`~adam_identification.identifier.MaterialIdentifier`.

Cross-references:
    - ``adam_identification.identifier`` — production ``MaterialIdentifier`` agent.
    - ``adam_identification.templates.identification.phase_selection``
      — crystal prompt.
    - ``adam_identification.templates.identification.molecule_candidate_selection``
      — molecule prompt.
"""

from __future__ import annotations

import json
from typing import Any

from adam_identification._confidence import parse_confidence_level
from adam_identification.database.pubchem import MoleculeCandidate
from adam_identification.llm.json_utils import _try_complete_json, parse_llm_json_object
from adam_identification.models import Material, MaterialSource, MaterialsProjectProperties

INITIAL_MAX_RESULTS = 20
WIDE_MAX_RESULTS = 50


def candidates_to_selection_json(candidates: list[Material]) -> list[dict[str, Any]]:
    """Serialize MP candidates for the phase-selection LLM prompt.

    Matches the compact payload used in production identification: index, MP id,
    formula, symmetry, site count, and key thermodynamic flags only.

    Args:
        candidates: Ordered Materials Project candidate materials.

    Returns:
        List of dicts suitable for ``json.dumps`` into ``phase_selection.j2``.
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


_VALID_DECISIONS: frozenset[str] = frozenset({"select", "ambiguous", "not_found"})


def parse_phase_selection_response(
    data: dict[str, Any],
    n_candidates: int,
) -> dict[str, Any]:
    """Normalise an LLM phase-selection JSON object.

    Args:
        data: Parsed LLM response object.
        n_candidates: Number of candidates in the current search pass.

    Returns:
        Dict with keys ``decision`` (``"select"``/``"ambiguous"``/``"not_found"``),
        ``selected_index`` (int | None), ``selection_reason`` (str),
        ``user_message`` (str | None), ``suggested_candidates`` (list),
        ``confidence`` (int | None), ``needs_review`` (bool).
    """
    decision = str(data.get("decision") or "not_found").lower()
    if decision not in _VALID_DECISIONS:
        decision = "not_found"

    # Always extract selected_index: in minimal_interaction mode the LLM provides
    # one even for "ambiguous" decisions so the caller can fall back to it.
    selected_index: int | None = None
    raw = data.get("selected_index")
    if raw is not None:
        try:
            selected_index = max(0, min(int(raw), n_candidates - 1))
        except (ValueError, TypeError):
            pass

    suggested_candidates = data.get("suggested_candidates")
    if not isinstance(suggested_candidates, list):
        suggested_candidates = []

    return {
        "decision": decision,
        "selected_index": selected_index,
        "selection_reason": str(data.get("selection_reason") or ""),
        "user_message": data.get("user_message") or None,
        "suggested_candidates": suggested_candidates,
        "confidence": parse_confidence_level(data.get("confidence")),
        "needs_review": bool(data.get("needs_review", False)),
    }


def parse_llm_json_tolerant(content: str) -> dict[str, Any]:
    """Parse JSON from LLM output, tolerating fences and surrounding prose.

    Applies three recovery strategies in order:
    1. Standard parse (handles Markdown fences and surrounding prose).
    2. Slice between first ``{`` and last ``}`` (handles trailing garbage).
    3. Brace-completion on truncated objects (handles Gemini mid-JSON cutoff).

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
        pass

    text = content.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return parse_llm_json_object(text[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            pass

    completed = _try_complete_json(text)
    if completed is not None:
        try:
            return parse_llm_json_object(completed)
        except (json.JSONDecodeError, TypeError):
            pass

    msg = f"Could not parse JSON object from LLM output: {content[:200]!r}"
    raise ValueError(msg) from None


def material_to_scoring_payload(material: Material) -> dict[str, Any]:
    """Convert a selected ``Material`` into benchmark scoring fields.

    Args:
        material: Chosen Materials Project material.

    Returns:
        Dict with keys used by identification benchmarks and downstream scoring.
    """
    struct = material.structure
    props = material.get_properties(MaterialSource.MATERIALS_PROJECT)
    electronic: dict[str, Any] = {
        "band_gap": None,
        "is_metal": None,
        "is_direct": None,
    }
    eah: float | None = None
    if isinstance(props, MaterialsProjectProperties):
        electronic = {
            "band_gap": props.band_gap,
            "is_metal": props.is_metal,
            "is_direct": props.is_direct_gap,
        }
        eah = props.energy_above_hull
    return {
        "material_id": material.mp_id or "",
        "chemical_formula": material.chemical_formula,
        "crystal_system": getattr(struct, "crystal_system", None) or "N/A",
        "space_group": getattr(struct, "space_group", None) or "N/A",
        "energy_above_hull": eah,
        "electronic_properties": electronic,
        "common_names": [],
        "structure_type": "N/A",
        "lattice_parameters": dict(getattr(struct, "lattice_parameters", {}) or {}),
        "atomic_positions": [
            {"element": pos.element, "position": list(pos.position)}
            for pos in getattr(struct, "atomic_positions", []) or []
        ],
    }


def parse_formula_counts(formula: str) -> dict[str, int]:
    """Parse a Hill-notation formula into ``{element: count}``.

    Handles standard formulas like ``"H2O"``, ``"C2H6O"``, ``"NaCl"``.
    Does not support parenthesised groups (not needed for small molecules).

    Args:
        formula: Chemical formula string.

    Returns:
        Mapping of element symbol to total atom count.
    """
    import re

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
    """Serialize PubChem molecule candidates for the ``molecule_candidate_selection.j2`` prompt.

    Includes ``isomeric_smiles`` and ``inchi_key`` when available so the v2
    selection template can apply stereochemistry-preserving identifiers.

    Args:
        candidates: List of :class:`~adam_identification.database.pubchem.MoleculeCandidate` dicts.

    Returns:
        List of dicts with ``index``, ``cid``, ``name``, ``formula``,
        ``smiles``, ``isomeric_smiles``, ``inchi_key``.
    """
    return [
        {
            "index": i,
            "cid": c["cid"],
            "name": c["name"],
            "formula": c["formula"],
            "smiles": c.get("smiles"),
            "isomeric_smiles": c.get("isomeric_smiles"),
            "inchi_key": c.get("inchi_key"),
        }
        for i, c in enumerate(candidates)
    ]


def parse_molecule_selection_response(
    data: dict[str, Any],
    n_candidates: int,
) -> dict[str, Any]:
    """Normalise an LLM molecule-selection JSON object.

    Mirrors :func:`parse_phase_selection_response` for the crystal path.

    Args:
        data: Parsed LLM response object.
        n_candidates: Number of candidates shown to the LLM.

    Returns:
        Dict with keys ``decision`` (``"select"``/``"ambiguous"``/``"not_found"``),
        ``selected_index`` (int | None), ``selection_reason`` (str),
        ``user_message`` (str | None), ``suggested_candidates`` (list),
        ``confidence`` (int | None, 1–5 scale), ``needs_review`` (bool).
    """
    decision = str(data.get("decision") or "not_found").lower()
    if decision not in _VALID_DECISIONS:
        decision = "not_found"

    # Always extract selected_index: in minimal_interaction mode the LLM provides
    # one even for "ambiguous" decisions.
    selected_index: int | None = None
    raw = data.get("selected_index")
    if raw is not None:
        try:
            selected_index = max(0, min(int(raw), n_candidates - 1))
        except (ValueError, TypeError):
            pass

    suggested_candidates = data.get("suggested_candidates")
    if not isinstance(suggested_candidates, list):
        suggested_candidates = []

    return {
        "decision": decision,
        "selected_index": selected_index,
        "selection_reason": str(data.get("selection_reason") or ""),
        "user_message": data.get("user_message") or None,
        "suggested_candidates": suggested_candidates,
        "confidence": parse_confidence_level(data.get("confidence")),
        "needs_review": bool(data.get("needs_review", False)),
    }


def phase_not_found_message(
    query: str,
    formula: str,
    n_candidates: int,
    selection_reason: str,
) -> str:
    """Build the production-style error when no matching phase is found.

    Args:
        query: Original material description.
        formula: Extracted chemical formula.
        n_candidates: Number of wide-search candidates examined.
        selection_reason: LLM ``selection_reason`` string from the final attempt.

    Returns:
        Human-readable error message.
    """
    return (
        f"Could not find the requested phase for {query!r} "
        f"(formula={formula!r}) in the Materials Project database. "
        f"Searched {n_candidates} candidates. "
        f"LLM reason: {selection_reason or 'none'}. "
        "Provide a more specific query or a known Materials Project ID."
    )
