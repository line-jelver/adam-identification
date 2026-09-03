"""Interactive identification session with clarification/resume support.

Wraps :class:`~adam_identification.identifier.MaterialIdentifier` with stateful
caching so that when the identifier raises
:class:`~adam_identification.exceptions.AmbiguousIdentificationError` or
:class:`~adam_identification.exceptions.ClarificationNeededError`, the caller can
supply a follow-up query and the session skips already-completed pipeline
stages.

Pipeline role:
    Optional layer on top of :class:`~adam_identification.identifier.MaterialIdentifier`.
    The stateless ``MaterialIdentifier`` is unchanged and remains the correct
    entry point for headless/benchmark use.

Cross-references:
    - ``adam_identification.identifier`` — ``MaterialIdentifier`` (stateless core).
    - ``adam_identification.exceptions`` — ``AmbiguousIdentificationError``,
      ``ClarificationNeededError``.
    - ``adam_identification.database.pubchem`` — ``MoleculeCandidate``, ``PubChemClient``.
    - ``adam_identification._phase_lookup`` — ``parse_molecule_selection_response``.
    - ``adam_identification.trace`` — ``IdentificationTrace``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
from adam_identification.trace import IdentificationTrace

logger = logging.getLogger(__name__)


@dataclass
class CachedState:
    """Intermediate pipeline state preserved across a clarification handoff.

    Populated by :class:`IdentificationSession` when the identifier raises an
    ambiguity or clarification error.  Used by :meth:`IdentificationSession.resume`
    to skip already-completed stages.

    Attributes:
        original_query: The first user query that initiated this session.
        formula: Chemical formula extracted in the first call.
        domain: ``"crystal"`` or ``"molecule"``.
        search_name: Normalized PubChem name from extraction (molecules only).
        candidates: Database candidates already fetched. For the crystal path
            these are :class:`~adam_identification.models.Material` objects; for the
            molecule path these are :class:`~adam_identification.database.pubchem.MoleculeCandidate`
            dicts.
        stage: Which selection stage is cached.
        trace: Partial :class:`IdentificationTrace` up to the cached stage.
    """

    original_query: str
    formula: str
    domain: Literal["crystal", "molecule"]
    search_name: str | None
    candidates: list[Any]
    stage: Literal["crystal_selection", "molecule_selection"]
    trace: IdentificationTrace = field(
        default_factory=lambda: IdentificationTrace(query="", domain="crystal")
    )


class IdentificationSession:
    """Stateful identifier with clarification/resume for interactive use.

    Wraps :class:`~adam_identification.identifier.MaterialIdentifier` and caches
    intermediate pipeline state so that selection-stage ambiguities can be
    resolved with a single follow-up message rather than a full restart.

    Usage example::

        session = IdentificationSession(identifier)
        try:
            material = session.identify("quartz")
        except AmbiguousIdentificationError as exc:
            print(exc.user_message)
            for s in exc.suggested_candidates:
                print(f"  {s['suggested_query']}")
            material = session.resume("right-handed alpha-quartz")

    Args:
        identifier: A configured :class:`~adam_identification.identifier.MaterialIdentifier`.
    """

    def __init__(self, identifier: MaterialIdentifier) -> None:
        self._id = identifier
        self._cached: CachedState | None = None
        self._last_ambiguity: AmbiguousIdentificationError | None = None
        self._last_clarification: ClarificationNeededError | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def identify(
        self,
        query: str,
        *,
        return_trace: bool = False,
    ) -> Material | tuple[Material, IdentificationTrace]:
        """Run identification. Raises on clarification or ambiguity with cached state.

        On success, returns a :class:`~adam_identification.models.Material`.  On
        ambiguity at the selection stage, raises
        :class:`~adam_identification.exceptions.AmbiguousIdentificationError` and
        caches state for :meth:`resume` / :meth:`resume_choice`.  On
        extraction-level clarification need, raises
        :class:`~adam_identification.exceptions.ClarificationNeededError` (no state
        is cached — re-call :meth:`identify` or use :meth:`identify_choice`
        with the refined query).

        Args:
            query: Natural language material description.
            return_trace: When ``True``, return ``(material, trace)`` on success.

        Returns:
            A ``Material`` or ``(Material, IdentificationTrace)`` on success.

        Raises:
            ClarificationNeededError: Extraction could not determine composition.
            AmbiguousIdentificationError: Multiple candidates remain; call
                :meth:`resume` or :meth:`resume_choice` with the user's choice.
            MaterialNotFoundError: No database match found.
        """
        self._cached = None
        self._last_ambiguity = None
        self._last_clarification = None
        try:
            return self._id.identify(query, return_trace=return_trace)
        except ClarificationNeededError as exc:
            self._last_clarification = exc
            raise
        except AmbiguousIdentificationError as exc:
            self._last_ambiguity = exc
            self._cache_state_from_exc(exc, query)
            raise

    def identify_choice(
        self,
        n: int,
        *,
        return_trace: bool = False,
    ) -> Material | tuple[Material, IdentificationTrace]:
        """Re-run identification with choice *n* from the last clarification error.

        The last error is a
        :class:`~adam_identification.exceptions.ClarificationNeededError`.

        Selects from
        :attr:`~adam_identification.exceptions.ClarificationNeededError.suggested_queries`
        using 1-based indexing, then calls :meth:`identify` with that query string.

        Args:
            n: 1-based index into the last ``ClarificationNeededError.suggested_queries``.
            return_trace: Forwarded to :meth:`identify`.

        Returns:
            A ``Material`` or ``(Material, IdentificationTrace)`` on success.

        Raises:
            ValueError: If there is no pending clarification, or *n* is out of range.
        """
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
        return self.identify(suggestions[n - 1], return_trace=return_trace)

    def resume(
        self,
        choice_query: str,
        *,
        return_trace: bool = False,
    ) -> Material | tuple[Material, IdentificationTrace]:
        """Resume from cached state using the user's follow-up query.

        For **selection-stage** ambiguity (``stage="crystal_selection"`` or
        ``"molecule_selection"``): skips re-extraction and database search;
        re-runs selection only with ``choice_query`` as the new query against
        the already-fetched candidate list.

        Args:
            choice_query: The user's follow-up, e.g. ``"right-handed quartz"``
                or ``"para-xylene"``.
            return_trace: When ``True``, return ``(material, trace)`` on success.

        Returns:
            A ``Material`` or ``(Material, IdentificationTrace)`` on success.

        Raises:
            ValueError: If there is no cached state to resume from.
            AmbiguousIdentificationError: If the follow-up is still ambiguous.
            MaterialNotFoundError: If the follow-up resolves to nothing.
        """
        if self._cached is None:
            raise ValueError(
                "No cached state to resume from. Call identify() first and handle "
                "the AmbiguousIdentificationError before calling resume()."
            )
        cached = self._cached
        self._cached = None

        if cached.stage == "crystal_selection":
            return self._resume_crystal(choice_query, cached, return_trace=return_trace)
        return self._resume_molecule(choice_query, cached, return_trace=return_trace)

    def resume_choice(
        self,
        n: int,
        *,
        return_trace: bool = False,
    ) -> Material | tuple[Material, IdentificationTrace]:
        """Resume from cached state with choice *n* from the last ambiguity error.

        The last error is an
        :class:`~adam_identification.exceptions.AmbiguousIdentificationError`.
        Selects from
        :attr:`~adam_identification.exceptions.AmbiguousIdentificationError.suggested_candidates`
        using 1-based indexing, then calls :meth:`resume` with the corresponding
        ``suggested_query`` string.

        Args:
            n: 1-based index into the last ``AmbiguousIdentificationError.suggested_candidates``.
            return_trace: Forwarded to :meth:`resume`.

        Returns:
            A ``Material`` or ``(Material, IdentificationTrace)`` on success.

        Raises:
            ValueError: If there is no pending ambiguity, or *n* is out of range.
            AmbiguousIdentificationError: If the chosen follow-up is still ambiguous.
            MaterialNotFoundError: If the chosen follow-up resolves to nothing.
        """
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
        return self.resume(suggestions[n - 1]["suggested_query"], return_trace=return_trace)

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _cache_state_from_exc(self, exc: AmbiguousIdentificationError, original_query: str) -> None:
        """Populate :attr:`_cached` from the exception and its trace."""
        trace: IdentificationTrace | None = getattr(exc, "trace", None)
        if trace is None:
            return

        domain = exc.domain
        if domain not in ("crystal", "molecule"):
            return

        formula = exc.formula
        search_name: str | None = None
        candidates: list[Any] = []

        if domain == "crystal":
            stage: Literal["crystal_selection", "molecule_selection"] = "crystal_selection"
            search_name = None
            # Reconstruct Material list from trace candidates is not feasible
            # without re-fetching; store the raw serialized list instead.
            # _resume_crystal will re-fetch when needed.
            candidates = []
        else:
            stage = "molecule_selection"
            search_name = trace.extraction.get("search_name")
            candidates = []

        if trace is None:
            return

        self._cached = CachedState(
            original_query=original_query,
            formula=formula,
            domain=domain,  # type: ignore[arg-type]
            search_name=search_name,
            candidates=candidates,
            stage=stage,
            trace=trace,
        )

    def _resume_crystal(
        self,
        choice_query: str,
        cached: CachedState,
        *,
        return_trace: bool,
    ) -> Material | tuple[Material, IdentificationTrace]:
        """Re-run crystal selection only with the new choice query."""
        from adam_identification._phase_lookup import WIDE_MAX_RESULTS

        trace = cached.trace
        formula = cached.formula

        # Re-fetch the wide candidate list (already done once; small overhead).
        candidates = self._id._mp.search_by_formula(formula, max_results=WIDE_MAX_RESULTS)
        trace.n_candidates_wide = len(candidates)
        trace.candidates_shown_wide = candidates_to_selection_json(candidates)

        if not candidates:
            trace.outcome = "not_found"
            raise MaterialNotFoundError(
                f"No Materials Project entries found for formula {formula!r}."
            )

        selection = self._id._select_phase(choice_query, formula, candidates)
        trace.wide_selection = selection
        trace.selection_decision = selection["decision"]
        trace.selection_reason = selection["selection_reason"]
        trace.suggested_candidates = selection["suggested_candidates"] or None

        if selection["decision"] == "select":
            idx = int(selection["selected_index"])
            material = candidates[idx]
            trace.outcome = "selected"
            trace.selected_id = material.mp_id
            if return_trace:
                return material, trace
            return material

        if selection["decision"] == "ambiguous":
            trace.outcome = "ambiguous"
            # Re-cache for further resume.
            self._cached = CachedState(
                original_query=cached.original_query,
                formula=formula,
                domain="crystal",
                search_name=None,
                candidates=[],
                stage="crystal_selection",
                trace=trace,
            )
            exc = AmbiguousIdentificationError(
                f"Still ambiguous after follow-up {choice_query!r}: "
                f"formula {formula!r} still has multiple plausible phases.",
                query=choice_query,
                formula=formula,
                domain="crystal",
                candidates=trace.candidates_shown_wide,
                user_message=selection.get("user_message"),
                suggested_candidates=selection.get("suggested_candidates"),
            )
            exc.trace = trace
            raise exc

        trace.outcome = "not_found"
        raise MaterialNotFoundError(
            f"No matching phase found for {choice_query!r} (formula={formula!r}) "
            f"in {len(candidates)} candidates. "
            f"LLM reason: {selection.get('selection_reason', 'none')}."
        )

    def _resume_molecule(
        self,
        choice_query: str,
        cached: CachedState,
        *,
        return_trace: bool,
    ) -> Material | tuple[Material, IdentificationTrace]:
        """Re-run molecule selection only with the new choice query."""
        if self._id._pubchem is None:
            raise ValueError("No PubChemClient configured.")

        trace = cached.trace
        formula = cached.formula
        search_name = cached.search_name

        # Re-fetch candidates (small overhead).
        name_to_search = search_name or cached.original_query
        try:
            candidates: list[MoleculeCandidate] = [
                c
                for c in self._id._pubchem.get_molecule_candidates(name_to_search)
                if c["formula"] == formula
            ]
        except MaterialNotFoundError:
            candidates = []

        if not candidates:
            trace.outcome = "not_found"
            raise MaterialNotFoundError(f"PubChem returned no candidates for formula {formula!r}.")

        prompt = self._id._loader.render(
            "identification/molecule_candidate_selection.j2",
            query=choice_query,
            formula=formula,
            search_name=search_name,
            candidates=molecule_candidates_to_selection_json(candidates),
        )
        data = self._id._call_llm_json(prompt)
        selection = parse_molecule_selection_response(data, len(candidates))
        trace.molecule_selection = selection
        trace.selection_decision = selection["decision"]
        trace.selection_reason = selection["selection_reason"]
        trace.suggested_candidates = selection["suggested_candidates"] or None

        if selection["decision"] == "select":
            idx = int(selection["selected_index"])
            winner_cid = candidates[idx]["cid"]
            trace.outcome = "selected"
            trace.selected_id = str(winner_cid)
            material = self._id._pubchem.get_by_cid(winner_cid)
            if return_trace:
                return material, trace
            return material

        if selection["decision"] == "ambiguous":
            trace.outcome = "ambiguous"
            self._cached = CachedState(
                original_query=cached.original_query,
                formula=formula,
                domain="molecule",
                search_name=search_name,
                candidates=[],
                stage="molecule_selection",
                trace=trace,
            )
            exc = AmbiguousIdentificationError(
                f"Still ambiguous after follow-up {choice_query!r}: "
                f"formula {formula!r} still has multiple plausible isomers.",
                query=choice_query,
                formula=formula,
                domain="molecule",
                candidates=trace.candidates_shown_molecule,
                user_message=selection.get("user_message"),
                suggested_candidates=selection.get("suggested_candidates"),
            )
            exc.trace = trace
            raise exc

        trace.outcome = "not_found"
        raise MaterialNotFoundError(
            f"No matching molecule found for {choice_query!r} (formula={formula!r}). "
            f"LLM reason: {selection.get('selection_reason', 'none')}."
        )


__all__ = ["IdentificationSession", "CachedState"]
