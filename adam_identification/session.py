"""Interactive identification session with clarification/resume support.

Each ``identify`` / ``resume`` call is a separate :class:`Trace`. Ambiguity
completes the current run; ``resume`` starts a linked child via ``parent_run_id``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from adam_identification._phase_lookup import (
    candidates_to_selection_json,
    molecule_candidates_to_selection_json,
    parse_molecule_selection_response,
)
from adam_identification.database.pubchem import MoleculeCandidate
from adam_identification.exceptions import (
    AmbiguousIdentificationError,
    ClarificationNeededError,
    MaterialNotFoundError,
)
from adam_identification.identifier import MaterialIdentifier
from adam_identification.models import Material
from adam_identification.provenance.context import checkpoint, identification_stage
from adam_identification.provenance.lifecycle import scoped_identify
from adam_identification.provenance.trace import (
    IdentificationSection,
    IdentificationStage,
    Trace,
    record_selected_material,
)


@dataclass
class CachedState:
    """Pipeline state preserved across a linked resume Trace."""

    original_query: str
    formula: str
    domain: Literal["crystal", "molecule"]
    search_name: str | None
    parent_run_id: str
    extraction: dict[str, Any]
    n_candidates_narrow: int | None
    candidates_shown_narrow: list[dict[str, Any]]
    n_candidates_wide: int | None
    candidates_shown_wide: list[dict[str, Any]]
    stage: Literal["crystal_selection", "molecule_selection"]
    parent_trace: Trace | None = None


class IdentificationSession:
    """Stateful identifier with clarification/resume for interactive use.

    Args:
        identifier: A configured :class:`~adam_identification.identifier.MaterialIdentifier`.
    """

    def __init__(self, identifier: MaterialIdentifier) -> None:
        self._id = identifier
        self._cached: CachedState | None = None
        self._last_ambiguity: AmbiguousIdentificationError | None = None
        self._last_clarification: ClarificationNeededError | None = None

    def identify(self, query: str) -> Material:
        """Run identification. Protocol outcomes complete the Trace then raise."""
        self._cached = None
        self._last_ambiguity = None
        self._last_clarification = None
        try:
            material, _trace = scoped_identify(query, lambda q, tr: self._id.identify(q, trace=tr))
            return material
        except ClarificationNeededError as exc:
            self._last_clarification = exc
            raise
        except AmbiguousIdentificationError as exc:
            self._last_ambiguity = exc
            self._cache_state_from_exc(exc, query)
            raise

    def identify_choice(self, n: int) -> Material:
        """Re-run identification with choice *n* from the last clarification error."""
        if self._last_clarification is None:
            raise ValueError(
                "No pending clarification. Call identify() first and handle "
                "the ClarificationNeededError before calling identify_choice()."
            )
        suggestions = self._last_clarification.suggested_queries
        if not suggestions:
            raise ValueError("The last ClarificationNeededError has no suggested_queries.")
        if not (1 <= n <= len(suggestions)):
            raise ValueError(
                f"Choice {n} is out of range — there are {len(suggestions)} "
                f"suggestions (1–{len(suggestions)})."
            )
        parent_id = None
        if self._last_clarification.trace is not None:
            parent_id = self._last_clarification.trace.run_id
        query = suggestions[n - 1]
        self._cached = None
        self._last_ambiguity = None
        self._last_clarification = None
        try:
            material, _trace = scoped_identify(
                query,
                lambda q, tr: self._id.identify(q, trace=tr),
                parent_run_id=parent_id,
            )
            return material
        except ClarificationNeededError as exc:
            self._last_clarification = exc
            raise
        except AmbiguousIdentificationError as exc:
            self._last_ambiguity = exc
            self._cache_state_from_exc(exc, query)
            raise

    def resume(self, choice_query: str) -> Material:
        """Resume from cached state on a new linked Trace."""
        if self._cached is None:
            raise ValueError(
                "No cached state to resume from. Call identify() first and handle "
                "the AmbiguousIdentificationError before calling resume()."
            )
        cached = self._cached
        self._cached = None

        if cached.stage == "crystal_selection":
            return self._resume_crystal(choice_query, cached)
        return self._resume_molecule(choice_query, cached)

    def resume_choice(self, n: int) -> Material:
        """Resume with choice *n* from the last ambiguity error."""
        if self._last_ambiguity is None:
            raise ValueError(
                "No pending ambiguity. Call identify() first and handle "
                "the AmbiguousIdentificationError before calling resume_choice()."
            )
        suggestions = self._last_ambiguity.suggested_candidates
        if not suggestions:
            raise ValueError("The last AmbiguousIdentificationError has no suggested_candidates.")
        if not (1 <= n <= len(suggestions)):
            raise ValueError(
                f"Choice {n} is out of range — there are {len(suggestions)} "
                f"suggestions (1–{len(suggestions)})."
            )
        return self.resume(suggestions[n - 1]["suggested_query"])

    def _cache_state_from_exc(self, exc: AmbiguousIdentificationError, original_query: str) -> None:
        """Populate :attr:`_cached` from the completed parent Trace."""
        parent: Trace | None = getattr(exc, "trace", None)
        if parent is None:
            return
        domain = exc.domain
        if domain not in ("crystal", "molecule"):
            return
        section = parent.identification
        self._cached = CachedState(
            original_query=original_query,
            formula=exc.formula,
            domain=domain,  # type: ignore[arg-type]
            search_name=section.extraction.get("search_name"),
            parent_run_id=parent.run_id,
            extraction=dict(section.extraction),
            n_candidates_narrow=section.n_candidates_narrow,
            candidates_shown_narrow=list(section.candidates_shown_narrow),
            n_candidates_wide=section.n_candidates_wide,
            candidates_shown_wide=list(section.candidates_shown_wide),
            stage="crystal_selection" if domain == "crystal" else "molecule_selection",
            parent_trace=parent,
        )

    def _seed_child(self, trace: Trace, cached: CachedState) -> IdentificationSection:
        """Copy cached extraction and candidate lists onto a new child Trace."""
        section = trace.identification
        section.domain = cached.domain
        section.extraction = dict(cached.extraction)
        section.n_candidates_narrow = cached.n_candidates_narrow
        section.candidates_shown_narrow = list(cached.candidates_shown_narrow)
        section.n_candidates_wide = cached.n_candidates_wide
        section.candidates_shown_wide = list(cached.candidates_shown_wide)
        return section

    def _resume_crystal(self, choice_query: str, cached: CachedState) -> Material:
        """Re-run crystal selection only with the new choice query."""
        from adam_identification._phase_lookup import WIDE_MAX_RESULTS

        def _fn(query: str, trace: Trace) -> Material:
            section = self._seed_child(trace, cached)
            checkpoint()
            formula = cached.formula
            with identification_stage(IdentificationStage.DATABASE_SEARCH):
                candidates = self._id._retriever.search(formula, WIDE_MAX_RESULTS).candidates
            section.n_candidates_wide = len(candidates)
            section.candidates_shown_wide = candidates_to_selection_json(candidates)
            checkpoint()
            if not candidates:
                section.outcome = "not_found"
                raise MaterialNotFoundError(
                    f"No crystal-database entries found for formula {formula!r}."
                )
            with identification_stage(IdentificationStage.SELECTION):
                selection = self._id._select_phase(
                    query, formula, candidates, purpose="identification.wide_selection"
                )
            section.wide_selection = selection
            section.selection_decision = selection["decision"]
            section.selection_reason = selection["selection_reason"]
            section.suggested_candidates = selection["suggested_candidates"] or None
            checkpoint()
            if selection["decision"] == "select":
                idx = int(selection["selected_index"])
                material = self._id._retriever.hydrate(candidates[idx])
                section.outcome = "selected"
                record_selected_material(section, material)
                return material
            if selection["decision"] == "ambiguous":
                section.outcome = "ambiguous"
                exc = AmbiguousIdentificationError(
                    f"Still ambiguous after follow-up {query!r}: "
                    f"formula {formula!r} still has multiple plausible phases.",
                    query=query,
                    formula=formula,
                    domain="crystal",
                    candidates=section.candidates_shown_wide,
                    user_message=selection.get("user_message"),
                    suggested_candidates=selection.get("suggested_candidates"),
                )
                exc.trace = trace
                raise exc
            section.outcome = "not_found"
            raise MaterialNotFoundError(
                f"No matching phase found for {query!r} (formula={formula!r}) "
                f"in {len(candidates)} candidates. "
                f"LLM reason: {selection.get('selection_reason', 'none')}."
            )

        try:
            material, _trace = scoped_identify(
                choice_query, _fn, parent_run_id=cached.parent_run_id
            )
            return material
        except AmbiguousIdentificationError as exc:
            self._last_ambiguity = exc
            self._cache_state_from_exc(exc, cached.original_query)
            raise

    def _resume_molecule(self, choice_query: str, cached: CachedState) -> Material:
        """Re-run molecule selection only with the new choice query."""
        if self._id._pubchem is None:
            raise ValueError("No PubChemClient configured.")

        def _fn(query: str, trace: Trace) -> Material:
            section = self._seed_child(trace, cached)
            formula = cached.formula
            search_name = cached.search_name
            name_to_search = search_name or cached.original_query
            try:
                with identification_stage(IdentificationStage.DATABASE_SEARCH):
                    candidates: list[MoleculeCandidate] = [
                        c
                        for c in self._id._pubchem.get_molecule_candidates(name_to_search)
                        if c["formula"] == formula
                    ]
            except MaterialNotFoundError:
                candidates = []
            if not candidates:
                section.outcome = "not_found"
                raise MaterialNotFoundError(
                    f"PubChem returned no candidates for formula {formula!r}."
                )
            section.n_candidates_narrow = len(candidates)
            section.candidates_shown_narrow = [
                {
                    "cid": c["cid"],
                    "name": c["name"],
                    "formula": c["formula"],
                    "smiles": c.get("smiles"),
                }
                for c in candidates
            ]
            checkpoint()
            prompt = self._id._loader.render(
                "identification/molecule_candidate_selection.j2",
                query=query,
                formula=formula,
                search_name=search_name,
                candidates=molecule_candidates_to_selection_json(candidates),
                minimal_interaction=self._id._minimal_interaction,
            )
            with identification_stage(IdentificationStage.SELECTION):
                data = self._id._call_llm_json(prompt, purpose="identification.narrow_selection")
            selection = parse_molecule_selection_response(data, len(candidates))
            section.narrow_selection = selection
            checkpoint()
            section.selection_decision = selection["decision"]
            section.selection_reason = selection["selection_reason"]
            section.suggested_candidates = selection["suggested_candidates"] or None
            if selection["decision"] == "select":
                idx = int(selection["selected_index"])
                winner_cid = candidates[idx]["cid"]
                section.outcome = "selected"
                section.selected_id = str(winner_cid)
                return self._id._pubchem.get_by_cid(winner_cid)
            if selection["decision"] == "ambiguous":
                section.outcome = "ambiguous"
                exc = AmbiguousIdentificationError(
                    f"Still ambiguous after follow-up {query!r}: "
                    f"formula {formula!r} still has multiple plausible isomers.",
                    query=query,
                    formula=formula,
                    domain="molecule",
                    candidates=section.candidates_shown_narrow,
                    user_message=selection.get("user_message"),
                    suggested_candidates=selection.get("suggested_candidates"),
                )
                exc.trace = trace
                raise exc
            section.outcome = "not_found"
            raise MaterialNotFoundError(
                f"No matching molecule found for {query!r} (formula={formula!r}). "
                f"LLM reason: {selection.get('selection_reason', 'none')}."
            )

        try:
            material, _trace = scoped_identify(
                choice_query, _fn, parent_run_id=cached.parent_run_id
            )
            return material
        except AmbiguousIdentificationError as exc:
            self._last_ambiguity = exc
            self._cache_state_from_exc(exc, cached.original_query)
            raise


__all__ = ["IdentificationSession", "CachedState"]
