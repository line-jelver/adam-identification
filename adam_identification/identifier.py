"""Material Identifier agent for ADaM.

Takes a natural language description of a material or molecule and resolves it
to a canonical :class:`~adam_identification.models.Material`.

* **Crystal path** — an explicit database ID (``mc3d-<n>`` or ``mp-<n>``) is
  resolved directly; otherwise a formula is extracted by the LLM and searched
  via a :class:`~adam_identification.database.crystal_retrieval.CrystalRetriever`
  (narrow → wide phase selection).  Returns a ``Material`` with
  ``CrystalStructure``.
* **Molecule path** — LLM signals ``domain="molecule"``; the agent collects
  up to 5 PubChem name-search candidates, lets the LLM select the correct
  structural isomer, then fetches 3D coordinates only for the winner.  Returns
  a ``Material`` with ``MoleculeStructure``.
* **Clarify path** — LLM signals ``decision="clarify"`` when the query does not
  determine a unique composition or domain; raises
  :class:`~adam_identification.exceptions.ClarificationNeededError` immediately.

Pipeline role:
    Resolves a natural-language query to a :class:`~adam_identification.models.Material`
    before structure conversion (ASE / pymatgen) or downstream DFT setup.

Cross-references:
    - ``adam_identification.database.materials_project`` — MP formula search.
    - ``adam_identification.database.pubchem`` — PubChem name lookup for molecules.
    - ``adam_identification.models`` — ``Material``, structure types.
    - ``adam_identification.exceptions`` — ``MaterialNotFoundError``,
      ``AmbiguousIdentificationError``, ``ClarificationNeededError``.
    - ``adam_identification.llm.base`` — ``BaseLLM`` provider interface.
    - ``adam_identification._prompts`` — Jinja2 template rendering.
    - ``adam_identification._phase_lookup`` — shared narrow/wide search policy.
    - ``adam_identification._followup`` — rewrite LLM follow-up query strings.
    - ``adam_identification.provenance.trace`` — ``Trace`` / ``IdentificationSection``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast

from adam_identification._confidence import parse_confidence_level
from adam_identification._followup import format_followup_queries
from adam_identification._phase_lookup import (
    INITIAL_MAX_RESULTS,
    WIDE_MAX_RESULTS,
    candidates_to_selection_json,
    molecule_candidates_to_selection_json,
    parse_llm_json_tolerant,
    parse_molecule_selection_response,
    parse_phase_selection_response,
    phase_not_found_message,
)
from adam_identification._phase_lookup import (
    formulas_match as _formulas_match,
)
from adam_identification._prompts import PromptLoader
from adam_identification.database.crystal_retrieval import (
    CrystalRetriever,
    CrystalSourcePolicy,
    looks_like_explicit_crystal_id,
)
from adam_identification.database.materials_project import MaterialsProjectClient
from adam_identification.database.materials_project_provider import (
    MaterialsProjectCrystalProvider,
)
from adam_identification.database.pubchem import MoleculeCandidate, PubChemClient
from adam_identification.exceptions import (
    AmbiguousIdentificationError,
    ClarificationNeededError,
    DatabaseAPIError,
    MaterialNotFoundError,
)
from adam_identification.llm.base import BaseLLM
from adam_identification.models import Material
from adam_identification.provenance.context import (
    checkpoint,
    identification_stage,
    llm_purpose,
)
from adam_identification.provenance.trace import (
    IdentificationStage,
    Trace,
    record_selected_material,
)

logger = logging.getLogger(__name__)


def _molecule_candidates_for_trace(
    candidates: list[MoleculeCandidate],
) -> list[dict[str, Any]]:
    """Serialize PubChem candidates into the shared narrow/wide candidate lists."""
    return [
        {
            "cid": candidate["cid"],
            "name": candidate["name"],
            "formula": candidate["formula"],
            "smiles": candidate.get("smiles"),
            "isomeric_smiles": candidate.get("isomeric_smiles"),
            "inchi_key": candidate.get("inchi_key"),
        }
        for candidate in candidates
    ]


def _attach_trace(exc: Exception, trace: Trace) -> None:
    """Attach the full provenance :class:`Trace` to an identification exception."""
    exc.trace = trace  # type: ignore[attr-defined]


def _checkpoint_identification() -> None:
    """Persist identification mutations when a store is bound."""
    checkpoint()


def _coerce_crystal_retriever(
    crystal_source: CrystalRetriever | MaterialsProjectClient,
) -> CrystalRetriever:
    """Wrap a bare Materials Project client as a Materials-Project-only retriever.

    A :class:`CrystalRetriever` is returned unchanged. Any other object is
    treated as an already-built retriever so tests can pass a double.
    """
    if isinstance(crystal_source, MaterialsProjectClient):
        return CrystalRetriever(
            [MaterialsProjectCrystalProvider(crystal_source)],
            policy=CrystalSourcePolicy.MATERIALS_PROJECT,
        )
    return crystal_source  # type: ignore[return-value]


class MaterialIdentifier:
    """Resolve a natural language material description to a ``Material``.

    Handles both periodic crystals (via a
    :class:`~adam_identification.database.crystal_retrieval.CrystalRetriever`)
    and discrete molecules (via PubChem).  The routing decision is made by the
    LLM during formula extraction: ``domain="molecule"`` triggers the PubChem
    path and ``domain="crystal"`` triggers the crystal-retriever path.

    Crystal identification:
        0. **Explicit-ID fast path** — a query that is exactly ``mc3d-<n>`` or
           ``mp-<n>`` is resolved via the retriever and skips the LLM.
        1. **Formula extraction** — LLM extracts formula, domain, and
           ``search_name``.  If ``decision="clarify"``, raises
           :class:`~adam_identification.exceptions.ClarificationNeededError`.
        2. **Phase selection (narrow)** — the retriever is searched for up to
           ``INITIAL_MAX_RESULTS`` candidates; LLM selects the best match.
           In ``minimal_interaction`` mode, ``"ambiguous"`` results are treated
           as a selection using ``selected_index`` from the LLM response.
        3. **Phase selection (wide, if needed)** — the retriever is re-queried
           with ``WIDE_MAX_RESULTS`` when the narrow selection returned
           ``decision="not_found"``.

    Molecule identification (two steps):
        1. **Formula extraction** — LLM extracts formula, domain, and
           ``search_name`` (a normalized PubChem name).
        2. **Candidate collection + selection** — top-5 PubChem name-search
           results for ``search_name`` (or the query when ``search_name`` is
           absent), formula-filtered, then the LLM selects the correct
           structural isomer.  3D coordinates are fetched only for the winner.

    Args:
        llm: Any ``BaseLLM`` provider.
        crystal_retriever: A configured
            :class:`~adam_identification.database.crystal_retrieval.CrystalRetriever`,
            or a bare
            :class:`~adam_identification.database.materials_project.MaterialsProjectClient`
            wrapped as a Materials-Project-only retriever.
        pubchem_client: Optional
            :class:`~adam_identification.database.pubchem.PubChemClient`.
            Required when molecule queries are expected.  If ``None`` and the LLM
            classifies the input as a molecule, :class:`ValueError` is raised.
        minimal_interaction: When ``True``, use minimal-interaction mode: the LLM
            always selects a candidate (never returns ``ambiguous``); uncertain
            selections are flagged with ``needs_review=True`` on the trace.
            When ``False`` (default), the LLM may return ``ambiguous`` and raise
            :class:`~adam_identification.exceptions.AmbiguousIdentificationError`.
    """

    def __init__(
        self,
        llm: BaseLLM,
        crystal_retriever: CrystalRetriever | MaterialsProjectClient,
        pubchem_client: PubChemClient | None = None,
        minimal_interaction: bool = False,
    ) -> None:
        self._llm = llm
        self._retriever = _coerce_crystal_retriever(crystal_retriever)
        self._pubchem = pubchem_client
        self._loader = PromptLoader()
        self._minimal_interaction = minimal_interaction

    def identify(self, query: str, *, trace: Trace) -> Material:
        """Resolve ``query`` to a ``Material`` from MP (crystal) or PubChem (molecule).

        Args:
            query: Natural language description, e.g. ``"silicon"``, ``"glucose"``,
                ``"hexagonal graphite"``.
            trace: Active :class:`~adam_identification.provenance.trace.Trace`.
                This method writes :attr:`Trace.identification` and checkpoints.

        Returns:
            A ``Material`` with ``CrystalStructure`` (crystals) or
            ``MoleculeStructure`` (molecules).

        Raises:
            ClarificationNeededError: If the extraction step cannot determine
                a unique composition or domain and needs user input.
            MaterialNotFoundError: If the database search returns no usable match.
            AmbiguousIdentificationError: If the formula is in the database but
                the query lacks enough information to pick one candidate.
            DatabaseAPIError: On API/network failures.
            ValueError: If the LLM returns an empty formula, or if
                ``domain="molecule"`` but no ``pubchem_client`` was provided.
        """
        section = trace.identification
        stripped_query = query.strip()
        if looks_like_explicit_crystal_id(stripped_query):
            return self._identify_by_explicit_id(stripped_query, trace)

        extraction = self._extract_formula(query)
        section.extraction = extraction
        domain: str = extraction["domain"]
        section.domain = domain  # type: ignore[assignment]
        _checkpoint_identification()

        if extraction["decision"] == "clarify":
            section.outcome = "clarify"
            section.clarification_message = extraction.get("user_message")
            _checkpoint_identification()
            exc = ClarificationNeededError(
                extraction.get("user_message")
                or f"LLM could not determine composition or domain for {query!r}.",
                user_message=extraction.get("user_message") or "",
                suggested_queries=extraction.get("suggested_queries") or [],
            )
            _attach_trace(exc, trace)
            raise exc

        formula: str = extraction["formula"]
        logger.info("[%s] formula=%s domain=%s", query, formula, domain)

        if domain == "molecule":
            return self._identify_molecule(query, extraction, trace)
        return self._identify_crystal(query, extraction, trace)

    def _identify_by_explicit_id(self, source_id: str, trace: Trace) -> Material:
        """Resolve a query that is exactly a database ID, skipping the LLM."""
        section = trace.identification
        section.domain = "crystal"
        section.extraction = {"decision": "proceed", "domain": "crystal", "formula": None}
        _checkpoint_identification()
        try:
            with identification_stage(IdentificationStage.DATABASE_SEARCH):
                material = self._retriever.get_by_id(source_id)
        except (MaterialNotFoundError, DatabaseAPIError) as error:
            section.outcome = "not_found"
            _checkpoint_identification()
            exc = MaterialNotFoundError(
                f"No crystal entry found for explicit database ID {source_id!r}: {error}"
            )
            _attach_trace(exc, trace)
            raise exc from error

        section.outcome = "selected"
        record_selected_material(section, material)
        section.selection_decision = "select"
        section.selection_reason = (
            f"Explicit database ID {source_id!r} bypasses formula extraction and phase selection."
        )
        _checkpoint_identification()
        logger.info("[%s] resolved directly via explicit database ID", source_id)
        return material

    def _identify_crystal(
        self,
        query: str,
        extraction: dict[str, Any],
        trace: Trace,
    ) -> Material:
        """Resolve a crystal query via the configured crystal retriever."""
        section = trace.identification
        formula: str = extraction["formula"]

        with identification_stage(IdentificationStage.DATABASE_SEARCH):
            candidates = self._retriever.search(formula, INITIAL_MAX_RESULTS).candidates
        section.n_candidates_narrow = len(candidates)
        section.candidates_shown_narrow = candidates_to_selection_json(candidates)
        _checkpoint_identification()
        if not candidates:
            section.outcome = "not_found"
            _checkpoint_identification()
            msg = (
                f"No crystal-database entries found for formula {formula!r} "
                f"(extracted from query {query!r})."
            )
            exc = MaterialNotFoundError(msg)
            _attach_trace(exc, trace)
            raise exc

        logger.info("[%s] retriever returned %d candidates (narrow)", query, len(candidates))
        with identification_stage(IdentificationStage.SELECTION):
            selection = self._select_phase(
                query, formula, candidates, purpose="identification.narrow_selection"
            )
        section.narrow_selection = selection
        section.selection_decision = selection["decision"]
        section.selection_reason = selection["selection_reason"]
        section.suggested_candidates = selection["suggested_candidates"] or None
        _checkpoint_identification()

        if selection["decision"] == "select" or (
            self._minimal_interaction and selection["decision"] == "ambiguous"
        ):
            if selection["decision"] == "ambiguous":
                sel_idx = selection.get("selected_index")
                idx = int(sel_idx) if sel_idx is not None else 0
                section.needs_review = True
                logger.warning(
                    "[%s] minimal_interaction: ambiguous→select index=%d needs_review=True",
                    query,
                    idx,
                )
            else:
                idx = int(selection["selected_index"])
                if selection.get("needs_review"):
                    section.needs_review = True
            material = self._retriever.hydrate(candidates[idx])
            section.outcome = "selected"
            record_selected_material(section, material)
            _checkpoint_identification()
            logger.info(
                "[%s] selected %s (%s) — %s",
                query,
                material.primary_id,
                getattr(material.structure, "space_group", "?"),
                selection.get("selection_reason", ""),
            )
            return material

        if selection["decision"] == "ambiguous":
            section.outcome = "ambiguous"
            _checkpoint_identification()
            exc = AmbiguousIdentificationError(
                f"Ambiguous query {query!r}: formula {formula!r} matches multiple "
                "phases and the query does not uniquely identify one.",
                query=query,
                formula=formula,
                domain="crystal",
                candidates=section.candidates_shown_narrow,
                user_message=selection.get("user_message"),
                suggested_candidates=selection.get("suggested_candidates"),
            )
            exc.trace = trace
            raise exc

        logger.warning(
            "[%s] phase not matched in narrow set (%d candidates): %s — widening to %d",
            query,
            len(candidates),
            selection.get("selection_reason", ""),
            WIDE_MAX_RESULTS,
        )
        with identification_stage(IdentificationStage.DATABASE_SEARCH):
            wide_candidates = self._retriever.search(formula, WIDE_MAX_RESULTS).candidates
        section.n_candidates_wide = len(wide_candidates)
        section.candidates_shown_wide = candidates_to_selection_json(wide_candidates)
        _checkpoint_identification()
        logger.info("[%s] retriever returned %d candidates (wide)", query, len(wide_candidates))

        with identification_stage(IdentificationStage.SELECTION):
            wide_selection = self._select_phase(
                query, formula, wide_candidates, purpose="identification.wide_selection"
            )
        section.wide_selection = wide_selection
        section.selection_decision = wide_selection["decision"]
        section.selection_reason = wide_selection["selection_reason"]
        section.suggested_candidates = wide_selection["suggested_candidates"] or None
        _checkpoint_identification()

        if wide_selection["decision"] == "select" or (
            self._minimal_interaction and wide_selection["decision"] == "ambiguous"
        ):
            if wide_selection["decision"] == "ambiguous":
                sel_idx = wide_selection.get("selected_index")
                idx = int(sel_idx) if sel_idx is not None else 0
                section.needs_review = True
                logger.warning(
                    "[%s] minimal_interaction: wide ambiguous→select index=%d needs_review=True",
                    query,
                    idx,
                )
            else:
                idx = int(wide_selection["selected_index"])
                if wide_selection.get("needs_review"):
                    section.needs_review = True
            material = self._retriever.hydrate(wide_candidates[idx])
            section.outcome = "selected"
            record_selected_material(section, material)
            _checkpoint_identification()
            logger.info(
                "[%s] selected %s (%s) after wide search — %s",
                query,
                material.primary_id,
                getattr(material.structure, "space_group", "?"),
                wide_selection.get("selection_reason", ""),
            )
            return material

        if wide_selection["decision"] == "ambiguous":
            section.outcome = "ambiguous"
            _checkpoint_identification()
            exc = AmbiguousIdentificationError(
                f"Ambiguous query {query!r}: formula {formula!r} matches "
                f"{len(wide_candidates)} phases and no candidate matched the request.",
                query=query,
                formula=formula,
                domain="crystal",
                candidates=section.candidates_shown_wide,
                user_message=wide_selection.get("user_message"),
                suggested_candidates=wide_selection.get("suggested_candidates"),
            )
            exc.trace = trace
            raise exc

        section.outcome = "not_found"
        _checkpoint_identification()
        msg = phase_not_found_message(
            query,
            formula,
            len(wide_candidates),
            str(wide_selection.get("selection_reason", "")),
        )
        exc = MaterialNotFoundError(msg)
        _attach_trace(exc, trace)
        raise exc

    def _identify_molecule(
        self,
        query: str,
        extraction: dict[str, Any],
        trace: Trace,
    ) -> Material:
        """Resolve a molecule query via PubChem with multi-candidate LLM selection."""
        section = trace.identification
        if self._pubchem is None:
            raise ValueError(
                f"Query {query!r} was identified as a molecule but no PubChemClient "
                "was provided. Pass pubchem_client= to MaterialIdentifier."
            )

        search_name: str | None = extraction.get("search_name")
        formula: str = extraction["formula"]

        name_to_search = search_name or query
        name_candidates: list[MoleculeCandidate] = []
        with identification_stage(IdentificationStage.DATABASE_SEARCH):
            try:
                name_candidates = self._pubchem.get_molecule_candidates(name_to_search)
            except MaterialNotFoundError as exc:
                logger.debug(
                    "[%s] Name candidate search for %r returned nothing: %s",
                    query,
                    name_to_search,
                    exc,
                )

        formula_candidates = [
            candidate
            for candidate in name_candidates
            if _formulas_match(candidate["formula"], formula)
        ]
        section.n_candidates_narrow = len(formula_candidates)
        section.candidates_shown_narrow = _molecule_candidates_for_trace(formula_candidates)
        _checkpoint_identification()

        if not formula_candidates:
            logger.warning(
                "[%s] No formula-matching candidates from name=%r; "
                "falling back to fastformula search for %r",
                query,
                name_to_search,
                formula,
            )
            with identification_stage(IdentificationStage.DATABASE_SEARCH):
                try:
                    formula_candidates = [
                        candidate
                        for candidate in self._pubchem.get_molecule_candidates_by_formula(formula)
                        if _formulas_match(candidate["formula"], formula)
                    ]
                    section.n_candidates_narrow = len(formula_candidates)
                    section.candidates_shown_narrow = _molecule_candidates_for_trace(
                        formula_candidates
                    )
                    _checkpoint_identification()
                except MaterialNotFoundError:
                    pass
                except DatabaseAPIError as db_err:
                    logger.warning(
                        "[%s] PubChem fastformula search failed after retries (formula=%r): %s; "
                        "treating as not found.",
                        query,
                        formula,
                        db_err,
                    )

        if not formula_candidates:
            section.outcome = "not_found"
            _checkpoint_identification()
            not_found_error = MaterialNotFoundError(
                f"PubChem returned no matching molecule for query {query!r}. "
                f"Tried name={name_to_search!r}, formula={formula!r}."
            )
            _attach_trace(not_found_error, trace)
            raise not_found_error

        with identification_stage(IdentificationStage.SELECTION):
            selection = self._call_molecule_selection(
                query, formula, search_name, formula_candidates
            )
        section.narrow_selection = selection
        section.selection_decision = selection["decision"]
        section.selection_reason = selection["selection_reason"]
        section.suggested_candidates = selection["suggested_candidates"] or None
        _checkpoint_identification()

        if selection["decision"] != "select":
            if self._minimal_interaction and selection["decision"] == "ambiguous":
                sel_idx = selection.get("selected_index")
                idx = int(sel_idx) if sel_idx is not None else 0
                section.needs_review = True
                logger.warning(
                    "[%s] minimal_interaction: molecule ambiguous→select idx=%d needs_review=True",
                    query,
                    idx,
                )
            else:
                section.outcome = "ambiguous"
                _checkpoint_identification()
                ambiguous_error = AmbiguousIdentificationError(
                    f"Ambiguous query {query!r}: formula {formula!r} matches "
                    f"{len(formula_candidates)} candidates and the query does not "
                    f"specify which structural isomer is intended.",
                    query=query,
                    formula=formula,
                    domain="molecule",
                    candidates=section.candidates_shown_narrow,
                    user_message=selection.get("user_message"),
                    suggested_candidates=selection.get("suggested_candidates"),
                )
                ambiguous_error.trace = trace
                raise ambiguous_error
        else:
            idx = int(selection["selected_index"])
            if selection.get("needs_review"):
                section.needs_review = True
        winner_cid = formula_candidates[idx]["cid"]
        logger.info(
            "[%s] LLM selected molecule candidate index %d: CID %d (%s) — %s",
            query,
            idx,
            winner_cid,
            formula_candidates[idx]["name"],
            selection.get("selection_reason", ""),
        )

        section.outcome = "selected"
        section.selected_id = str(winner_cid)
        _checkpoint_identification()

        logger.info("[%s] Fetching 3D structure for winner CID %d", query, winner_cid)
        with identification_stage(IdentificationStage.DATABASE_SEARCH):
            return self._pubchem.get_by_cid(winner_cid)

    def _call_molecule_selection(
        self,
        query: str,
        formula: str,
        search_name: str | None,
        candidates: list[MoleculeCandidate],
    ) -> dict[str, Any]:
        """Call the LLM to select among PubChem candidates."""
        prompt = self._loader.render(
            "identification/molecule_candidate_selection.j2",
            query=query,
            formula=formula,
            search_name=search_name,
            candidates=molecule_candidates_to_selection_json(candidates),
            minimal_interaction=self._minimal_interaction,
        )
        data = self._call_llm_json(prompt, purpose="identification.narrow_selection")
        result = parse_molecule_selection_response(data, len(candidates))
        result["suggested_candidates"] = format_followup_queries(
            result["suggested_candidates"],
            molecule_candidates_to_selection_json(candidates),
            domain="molecule",
        )
        logger.info(
            "[%s] molecule selection decision=%s index=%s reason=%s",
            query,
            result["decision"],
            result["selected_index"],
            result.get("selection_reason", ""),
        )
        return result

    def _run_async(self, coro: Any) -> Any:  # type: ignore[misc]
        """Run an async LLM coroutine in a fresh event loop (sync context)."""
        return asyncio.run(coro)

    def _call_llm_json(
        self, prompt: str, *, purpose: str = "identification.extraction"
    ) -> dict[str, Any]:
        """Call the LLM and parse JSON from its response."""

        async def _call() -> dict[str, Any]:
            with identification_stage(IdentificationStage.LLM):
                response = await self._llm.complete_single(prompt, response_format="json")
            return cast(dict[str, Any], parse_llm_json_tolerant(response.content))

        with llm_purpose(purpose):
            return cast(dict[str, Any], self._run_async(_call()))

    def _extract_formula(self, query: str) -> dict[str, Any]:
        """Run formula-extraction prompt; return extraction dict."""
        prompt = self._loader.render("identification/formula_extraction.j2", query=query)
        data = self._call_llm_json(prompt, purpose="identification.extraction")
        decision = str(data.get("decision") or "proceed").lower()
        domain = str(data.get("domain") or "crystal").lower()

        if decision != "clarify":
            formula = str(data.get("formula") or "").strip()
            if not formula:
                msg = f"LLM returned an empty formula for query {query!r}"
                raise ValueError(msg)
        else:
            formula = ""

        return {
            "decision": decision,
            "domain": domain,
            "formula": formula,
            "search_name": data.get("search_name") or None,
            "user_message": data.get("user_message") or None,
            "suggested_queries": data.get("suggested_queries") or [],
            "confidence": parse_confidence_level(data.get("confidence")),
        }

    def _select_phase(
        self,
        query: str,
        formula: str,
        candidates: list[Any],
        *,
        purpose: str = "identification.narrow_selection",
    ) -> dict[str, Any]:
        """Run phase-selection prompt and return the selection dict."""
        prompt = self._loader.render(
            "identification/phase_selection.j2",
            query=query,
            formula=formula,
            candidates_json=json.dumps(candidates_to_selection_json(candidates), indent=2),
            minimal_interaction=self._minimal_interaction,
        )
        data = self._call_llm_json(prompt, purpose=purpose)
        result = parse_phase_selection_response(data, len(candidates))
        result["suggested_candidates"] = format_followup_queries(
            result["suggested_candidates"],
            candidates_to_selection_json(candidates),
            domain="crystal",
        )
        return result


__all__ = ["MaterialIdentifier"]
