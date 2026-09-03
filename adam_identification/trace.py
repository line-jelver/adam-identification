"""Provenance record for a single material identification run.

Captures the parsed intermediate state at each agent step so callers
can inspect decisions without needing verbatim LLM response strings.

Pipeline position: populated by :class:`~adam_identification.identifier.MaterialIdentifier`
when ``return_trace=True``.

Cross-references:
    - ``adam_identification.identifier`` — ``MaterialIdentifier`` populates this.
    - ``adam_identification.exceptions`` — ``AmbiguousIdentificationError`` may carry a trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class IdentificationTrace:
    """Full provenance of one identification call.

    Attributes:
        query: Original natural-language query.
        domain: ``"crystal"``, ``"molecule"``, or ``"unknown"``.
        extraction: Parsed extraction step output: decision, domain, formula,
            search_name, confidence (int|None).
        n_candidates_narrow: Number of MP candidates in the narrow search (crystals).
            None for the molecule path.
        candidates_shown_narrow: Compact candidate list from the narrow search.
            Each dict has mp_id, formula, space_group, crystal_system,
            energy_above_hull, is_metal. Empty when not applicable.
        n_candidates_wide: Number of MP candidates in the wide search (crystals,
            only when narrow selection returned ``decision="not_found"``).
            None otherwise.
        candidates_shown_wide: Compact candidate list from the wide search.
            Empty when not applicable.
        candidates_shown_molecule: Compact PubChem candidate list (molecules).
            Each dict has cid, name, formula, smiles. Empty for crystals.
        narrow_selection: Parsed narrow phase-selection output: decision (str),
            selected_index (int|None), selection_reason (str), user_message
            (str|None), suggested_candidates (list), confidence (int|None).
            Always populated for crystals; None for the molecule path.
        wide_selection: Parsed wide phase-selection output (same keys as
            narrow_selection). None when wide search was not triggered.
        molecule_selection: Parsed molecule candidate-selection output: decision
            (str), selected_index (int|None), selection_reason (str),
            user_message (str|None), suggested_candidates (list), confidence
            (int|None). None for the crystal path or when selection was skipped.
        selection_decision: Final selection step decision:
            ``"select"``, ``"ambiguous"``, or ``"not_found"``.  None when
            the pipeline did not reach the selection step (e.g. clarify path).
        selection_reason: Brief audit string from the LLM's ``selection_reason``
            field. None when selection was not reached.
        clarification_message: LLM-generated clarification message when
            ``extraction["decision"] == "clarify"``. None otherwise.
        suggested_candidates: Structured follow-up suggestions from the last
            selection step. Each dict has ``index`` and ``suggested_query``.
            None when selection was not reached.
        outcome: Final outcome: ``"selected"`` (success), ``"not_found"``
            (formula absent or OOD), ``"ambiguous"`` (formula in DB but query
            underspecified), ``"clarify"`` (composition unresolvable at extraction).
        selected_id: Winning MP material_id or PubChem CID string. None when
            outcome is not ``"selected"``.
        needs_review: True when ``minimal_interaction`` mode selected under a
            non-obvious conventional assumption or resolved an ambiguous response.
    """

    query: str
    domain: Literal["crystal", "molecule", "unknown"]
    extraction: dict[str, Any] = field(default_factory=dict)
    n_candidates_narrow: int | None = None
    candidates_shown_narrow: list[dict[str, Any]] = field(default_factory=list)
    n_candidates_wide: int | None = None
    candidates_shown_wide: list[dict[str, Any]] = field(default_factory=list)
    candidates_shown_molecule: list[dict[str, Any]] = field(default_factory=list)
    narrow_selection: dict[str, Any] | None = None
    wide_selection: dict[str, Any] | None = None
    molecule_selection: dict[str, Any] | None = None
    selection_decision: str | None = None
    selection_reason: str | None = None
    clarification_message: str | None = None
    suggested_candidates: list[dict[str, Any]] | None = None
    outcome: Literal["selected", "not_found", "ambiguous", "clarify"] = "not_found"
    selected_id: str | None = None
    needs_review: bool = False
