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
from adam_identification.database.crystal_provider import CrystalCandidate
from adam_identification.database.pubchem import MoleculeCandidate
from adam_identification.llm.json_utils import _try_complete_json, parse_llm_json_object

INITIAL_MAX_RESULTS = 20
WIDE_MAX_RESULTS = 50


def candidates_to_selection_json(candidates: list[CrystalCandidate]) -> list[dict[str, Any]]:
    """Serialize crystal candidates for the phase-selection LLM prompt.

    The ``mp_id`` key is populated from each candidate's provider-specific
    ``source_id``. The name is kept so already-persisted provenance JSON keeps
    the same shape. ``energy_above_hull`` and ``is_metal`` are ``None`` for a
    provider that does not report them.

    Args:
        candidates: Ordered crystal candidates from a narrow or wide search.

    Returns:
        List of dicts suitable for ``json.dumps`` into ``phase_selection.j2``.
    """
    return [
        {
            "index": idx,
            "mp_id": candidate.source_id,
            "formula": candidate.formula,
            "space_group": candidate.space_group,
            "crystal_system": candidate.crystal_system,
            "nsites": candidate.nsites,
            "energy_above_hull": candidate.energy_above_hull_ev_per_atom,
            "is_metal": candidate.is_metal,
        }
        for idx, candidate in enumerate(candidates)
    ]


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
        f"(formula={formula!r}) in the configured crystal database(s). "
        f"Searched {n_candidates} candidates. "
        f"LLM reason: {selection_reason or 'none'}. "
        "Provide a more specific query or a known database ID (e.g. an MC3D "
        "or Materials Project ID)."
    )
