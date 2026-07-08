"""Material identifier agent for adam-identification.

Takes a natural language description of a material or molecule and resolves it
to a canonical :class:`~adam_identification.models.Material`.

* **Crystal path** — formula extracted by LLM, then queried against the Materials
  Project (narrow → wide phase selection).  Returns a ``Material`` with
  ``CrystalStructure``.
* **Molecule path** — LLM signals ``is_molecule=True``; the agent collects
  up to 5 PubChem candidates (SMILES exact-match + name-based multi-candidate),
  lets the LLM select the correct structural isomer, then fetches 3D coordinates
  only for the winner.  Returns a ``Material`` with ``MoleculeStructure``.

    - ``adam_identification.database.materials_project`` — MP formula search and structure mapping.
    - ``adam_identification.database.pubchem`` — PubChem SMILES/name lookup for molecules.
    - ``adam_identification.models`` — ``Material``, ``CrystalStructure``, ``MoleculeStructure``.
    - ``adam_identification.exceptions`` — ``MaterialNotFoundError``, ``AmbiguousIdentificationError``.
    - ``adam_identification.llm.base`` — ``BaseLLM`` provider interface.
    - ``adam_identification._prompts`` — Jinja2 template rendering.
    - ``adam_identification._phase_lookup`` — shared narrow/wide search policy.
    - ``adam_identification.trace`` — ``IdentificationTrace`` provenance record.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast

from adam_identification.database.materials_project import MaterialsProjectClient
from adam_identification.database.pubchem import MoleculeCandidate, PubChemClient
from adam_identification.exceptions import (
    AmbiguousIdentificationError,
    DatabaseAPIError,
    MaterialNotFoundError,
)
from adam_identification.models import Material
from adam_identification.llm.base import BaseLLM
from adam_identification._confidence import parse_confidence_level
from adam_identification._phase_lookup import (
    INITIAL_MAX_RESULTS,
    WIDE_MAX_RESULTS,
    candidates_to_selection_json,
    molecule_candidates_to_selection_json,
    parse_llm_json_tolerant,
    parse_molecule_selection_response,
    parse_phase_selection_response,
    polymorph_not_found_message,
)
from adam_identification._phase_lookup import (
    formulas_match as _formulas_match,
)
from adam_identification.trace import IdentificationTrace
from adam_identification._prompts import PromptLoader

logger = logging.getLogger(__name__)


def _molecule_candidates_for_trace(
    candidates: list[MoleculeCandidate],
) -> list[dict[str, Any]]:
    """Serialize PubChem candidates for :class:`IdentificationTrace`."""
    return [
        {
            "cid": candidate["cid"],
            "name": candidate["name"],
            "formula": candidate["formula"],
            "smiles": candidate.get("smiles"),
        }
        for candidate in candidates
    ]


def _attach_trace(exc: Exception, trace: IdentificationTrace) -> None:
    """Attach provenance to an identification exception when supported."""
    exc.trace = trace  # type: ignore[attr-defined]


class MaterialIdentifier:
    """Resolve a natural language material description to a ``Material``.

    Handles both periodic crystals (via the Materials Project) and discrete
    molecules (via PubChem).  The routing decision is made by the LLM during
    formula extraction: ``is_molecule=True`` triggers the PubChem path.

    Crystal identification (three steps):
        1. **Formula extraction** — LLM extracts formula, polymorph hint, and
           ``is_molecule=False``.
        2. **Phase selection (narrow)** — MP searched for up to
           ``INITIAL_MAX_RESULTS`` candidates; LLM selects the best match.
        3. **Phase selection (wide, if needed)** — MP re-queried with
           ``WIDE_MAX_RESULTS`` when the target polymorph was not in the narrow set.

    Molecule identification (three steps):
        1. **Formula extraction** — LLM extracts formula, SMILES, and
           ``is_molecule=True``.
        2. **Candidate collection** — SMILES exact-match (0–1 hit) + top-5
           PubChem name-search results, deduplicated and formula-filtered.
        3. **Candidate selection** — the LLM selects the correct structural
           isomer from every non-empty candidate list (including single-candidate
           cases).  3D coordinates are fetched only for the winning CID.

    Args:
        llm: Any ``BaseLLM`` provider.
        mp_client: Configured :class:`~adam_identification.database.materials_project.MaterialsProjectClient`.
        pubchem_client: Optional :class:`~adam_identification.database.pubchem.PubChemClient`.
            Required when molecule queries are expected.  If ``None`` and the LLM
            classifies the input as a molecule, :class:`ValueError` is raised.
    """

    def __init__(
        self,
        llm: BaseLLM,
        mp_client: MaterialsProjectClient,
        pubchem_client: PubChemClient | None = None,
    ) -> None:
        self._llm = llm
        self._mp = mp_client
        self._pubchem = pubchem_client
        self._loader = PromptLoader()

    # ── Public API ──────────────────────────────────────────────────────────────

    def identify(
        self,
        query: str,
        *,
        return_trace: bool = False,
    ) -> Material | tuple[Material, IdentificationTrace]:
        """Resolve ``query`` to a ``Material`` from MP (crystal) or PubChem (molecule).

        Args:
            query: Natural language description, e.g. ``"silicon"``, ``"glucose"``,
                ``"hexagonal graphite"``.
            return_trace: When ``True``, return ``(material, trace)`` instead of
                only the material.  On failure, exceptions may carry a ``trace``
                attribute when populated.

        Returns:
            A ``Material`` with ``CrystalStructure`` (crystals) or
            ``MoleculeStructure`` (molecules), or a ``(Material, IdentificationTrace)``
            tuple when ``return_trace=True``.

        Raises:
            MaterialNotFoundError: If the database search returns no usable match.
            AmbiguousIdentificationError: If the formula is in the database but
                the query lacks enough information to pick one candidate.
            DatabaseAPIError: On API/network failures.
            ValueError: If the LLM returns an empty formula, or if ``is_molecule=True``
                but no ``pubchem_client`` was provided.
        """
        trace = IdentificationTrace(query=query, domain="crystal")
        extraction = self._extract_formula(query)
        trace.extraction = extraction
        formula: str = extraction["formula"]
        is_molecule: bool = extraction["is_molecule"]
        logger.info(
            "[%s] formula=%s is_molecule=%s",
            query,
            formula,
            is_molecule,
        )

        if is_molecule:
            trace.domain = "molecule"
            material = self._identify_molecule(query, extraction, trace)
            if return_trace:
                return material, trace
            return material

        material = self._identify_crystal(query, extraction, trace)
        if return_trace:
            return material, trace
        return material

    # ── Internal helpers ────────────────────────────────────────────────────────

    def _identify_crystal(
        self,
        query: str,
        extraction: dict[str, Any],
        trace: IdentificationTrace,
    ) -> Material:
        """Resolve a crystal query via Materials Project phase selection."""
        formula: str = extraction["formula"]
        polymorph_name: str | None = extraction.get("polymorph_name")
        space_group_hint: str | None = extraction.get("space_group_hint")
        logger.debug("[%s] polymorph=%s sg=%s", query, polymorph_name, space_group_hint)

        candidates = self._mp.search_by_formula(formula, max_results=INITIAL_MAX_RESULTS)
        trace.n_candidates_narrow = len(candidates)
        trace.candidates_shown_narrow = candidates_to_selection_json(candidates)
        if not candidates:
            trace.outcome = "not_found"
            msg = (
                f"No Materials Project entries found for formula {formula!r} "
                f"(extracted from query {query!r})."
            )
            exc = MaterialNotFoundError(msg)
            _attach_trace(exc, trace)
            raise exc

        logger.info("[%s] MP returned %d candidates (narrow)", query, len(candidates))
        selection = self._select_phase(query, formula, candidates, polymorph_name, space_group_hint)
        trace.narrow_selection = selection

        if selection["found"]:
            idx = int(selection["selected_index"])
            material = candidates[idx]
            trace.outcome = "selected"
            trace.selected_id = material.mp_id
            logger.info(
                "[%s] selected %s (%s) — %s",
                query,
                material.mp_id,
                getattr(material.structure, "space_group", "?"),
                selection.get("reason", ""),
            )
            return material

        logger.warning(
            "[%s] polymorph not matched in narrow set (%d candidates): %s — widening to %d",
            query,
            len(candidates),
            selection.get("reason", ""),
            WIDE_MAX_RESULTS,
        )
        wide_candidates = self._mp.search_by_formula(formula, max_results=WIDE_MAX_RESULTS)
        trace.n_candidates_wide = len(wide_candidates)
        trace.candidates_shown_wide = candidates_to_selection_json(wide_candidates)
        logger.info("[%s] MP returned %d candidates (wide)", query, len(wide_candidates))

        wide_selection = self._select_phase(
            query, formula, wide_candidates, polymorph_name, space_group_hint
        )
        trace.wide_selection = wide_selection

        if wide_selection["found"]:
            idx = int(wide_selection["selected_index"])
            material = wide_candidates[idx]
            trace.outcome = "selected"
            trace.selected_id = material.mp_id
            logger.info(
                "[%s] selected %s (%s) after wide search — %s",
                query,
                material.mp_id,
                getattr(material.structure, "space_group", "?"),
                wide_selection.get("reason", ""),
            )
            return material

        if len(wide_candidates) > 1:
            trace.outcome = "ambiguous"
            exc = AmbiguousIdentificationError(
                f"Ambiguous query {query!r}: formula {formula!r} matches "
                f"{len(wide_candidates)} phases and no candidate matched the request.",
                query=query,
                formula=formula,
                domain="crystal",
                candidates=trace.candidates_shown_wide,
            )
            exc.trace = trace
            raise exc

        trace.outcome = "not_found"
        msg = polymorph_not_found_message(
            query,
            formula,
            polymorph_name,
            space_group_hint,
            len(wide_candidates),
            str(wide_selection.get("reason", "")),
        )
        exc = MaterialNotFoundError(msg)
        _attach_trace(exc, trace)
        raise exc

    def _identify_molecule(
        self,
        query: str,
        extraction: dict[str, Any],
        trace: IdentificationTrace,
    ) -> Material:
        """Resolve a molecule query via PubChem with multi-candidate LLM selection."""
        if self._pubchem is None:
            raise ValueError(
                f"Query {query!r} was identified as a molecule but no PubChemClient "
                "was provided. Pass pubchem_client= to MaterialIdentifier."
            )

        smiles: str | None = extraction.get("smiles")
        common_name: str | None = extraction.get("common_name")
        formula: str = extraction["formula"]

        smiles_material: Material | None = None
        if smiles:
            try:
                candidate = self._pubchem.get_by_smiles(smiles)
                if _formulas_match(candidate.chemical_formula, formula):
                    smiles_material = candidate
                else:
                    logger.warning(
                        "[%s] SMILES=%r formula mismatch (%s vs expected %s); "
                        "continuing to name search",
                        query,
                        smiles,
                        candidate.chemical_formula,
                        formula,
                    )
            except MaterialNotFoundError as exc:
                logger.debug("[%s] SMILES lookup failed: %s", query, exc)

        name_to_search = common_name or query
        name_candidates: list[MoleculeCandidate] = []
        try:
            name_candidates = self._pubchem.get_molecule_candidates(name_to_search)
        except MaterialNotFoundError as exc:
            logger.debug(
                "[%s] Name candidate search for %r returned nothing: %s",
                query,
                name_to_search,
                exc,
            )

        seen_cids: set[int] = set()
        all_candidates: list[MoleculeCandidate] = []

        if smiles_material is not None and smiles_material.pc_cid is not None:
            struct = smiles_material.structure
            all_candidates.append(
                MoleculeCandidate(
                    cid=int(smiles_material.pc_cid),
                    name=smiles_material.name,
                    formula=smiles_material.chemical_formula,
                    smiles=getattr(struct, "smiles", None),
                )
            )
            seen_cids.add(int(smiles_material.pc_cid))

        for candidate in name_candidates:
            if candidate["cid"] not in seen_cids:
                seen_cids.add(candidate["cid"])
                all_candidates.append(candidate)

        formula_candidates = [
            candidate for candidate in all_candidates if _formulas_match(candidate["formula"], formula)
        ]
        trace.candidates_shown_molecule = _molecule_candidates_for_trace(formula_candidates)

        if not formula_candidates:
            logger.warning(
                "[%s] No formula-matching candidates from SMILES or name=%r; "
                "falling back to fastformula search for %r",
                query,
                name_to_search,
                formula,
            )
            try:
                formula_candidates = [
                    candidate
                    for candidate in self._pubchem.get_molecule_candidates_by_formula(formula)
                    if _formulas_match(candidate["formula"], formula)
                ]
                trace.candidates_shown_molecule = _molecule_candidates_for_trace(formula_candidates)
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
            trace.outcome = "not_found"
            exc = MaterialNotFoundError(
                f"PubChem returned no matching molecule for query {query!r}. "
                f"Tried SMILES={smiles!r}, name={name_to_search!r}, formula={formula!r}."
            )
            _attach_trace(exc, trace)
            raise exc

        selection = self._call_molecule_selection(query, formula, common_name, smiles, formula_candidates)
        trace.molecule_selection = selection
        if not selection["found"]:
            trace.outcome = "ambiguous"
            exc = AmbiguousIdentificationError(
                f"Ambiguous query {query!r}: formula {formula!r} matches "
                f"{len(formula_candidates)} candidates and the query does not "
                f"specify which structural isomer is intended.",
                query=query,
                formula=formula,
                domain="molecule",
                candidates=trace.candidates_shown_molecule,
            )
            exc.trace = trace
            raise exc
        idx = int(selection["selected_index"])
        winner_cid = formula_candidates[idx]["cid"]
        logger.info(
            "[%s] LLM selected molecule candidate index %d: CID %d (%s) — %s",
            query,
            idx,
            winner_cid,
            formula_candidates[idx]["name"],
            selection.get("reason", ""),
        )

        trace.outcome = "selected"
        trace.selected_id = str(winner_cid)

        if (
            smiles_material is not None
            and smiles_material.pc_cid is not None
            and int(smiles_material.pc_cid) == winner_cid
        ):
            logger.info(
                "[%s] Winner CID %d came from SMILES lookup; reusing fetched material",
                query,
                winner_cid,
            )
            return smiles_material

        logger.info("[%s] Fetching 3D structure for winner CID %d", query, winner_cid)
        return self._pubchem.get_by_cid(winner_cid)

    def _call_molecule_selection(
        self,
        query: str,
        formula: str,
        common_name: str | None,
        smiles: str | None,
        candidates: list[MoleculeCandidate],
    ) -> dict[str, Any]:
        """Call the LLM to select among PubChem candidates.

        Args:
            query: Original user query.
            formula: LLM-extracted formula (all candidates already pass the
                formula filter at this point).
            candidates: PubChem candidate dicts with CID, name, formula, SMILES.

        Returns:
            Parsed selection dict with keys ``found``, ``selected_index``,
            ``reason``, ``confidence``.
        """
        prompt = self._loader.render(
            "identification/molecule_candidate_selection.j2",
            query=query,
            formula=formula,
            common_name=common_name,
            smiles=smiles,
            candidates=molecule_candidates_to_selection_json(candidates),
        )
        data = self._call_llm_json(prompt)
        result = parse_molecule_selection_response(data, len(candidates))
        logger.info(
            "[%s] molecule selection found=%s index=%s reason=%s",
            query,
            result["found"],
            result["selected_index"],
            result.get("reason", ""),
        )
        return result

    def _run_async(self, coro: Any) -> Any:  # type: ignore[misc]
        """Run an async LLM coroutine in a fresh event loop (sync context)."""
        return asyncio.run(coro)

    def _call_llm_json(self, prompt: str) -> dict[str, Any]:
        """Call the LLM and parse JSON from its response.

        Args:
            prompt: Full rendered prompt text.

        Returns:
            Parsed JSON object.

        Raises:
            ValueError: If the LLM response cannot be parsed as a JSON object.
        """

        async def _call() -> dict[str, Any]:
            response = await self._llm.complete_single(prompt, response_format="json")
            return cast(dict[str, Any], parse_llm_json_tolerant(response.content))

        return cast(dict[str, Any], self._run_async(_call()))

    def _extract_formula(self, query: str) -> dict[str, Any]:
        """Run formula-extraction prompt; return extraction dict.

        Args:
            query: User material description.

        Returns:
            Dict with keys:

            * ``formula`` — reduced chemical formula string.
            * ``is_molecule`` — ``True`` for discrete molecules, ``False`` for
              periodic crystals.
            * ``smiles`` — canonical SMILES string (molecules only, else ``None``).
            * ``common_name`` — common chemical name (molecules only, else ``None``).
            * ``polymorph_name`` — crystal phase name, or ``None``.
            * ``space_group_hint`` — Hermann-Mauguin symbol, or ``None``.
            * ``confidence`` — integer ``1`` (low) through ``5`` (high).

        Raises:
            ValueError: If LLM returns an empty formula.
        """
        prompt = self._loader.render("identification/formula_extraction.j2", query=query)
        data = self._call_llm_json(prompt)
        formula = str(data.get("formula") or "").strip()
        if not formula:
            msg = f"LLM returned an empty formula for query {query!r}"
            raise ValueError(msg)
        return {
            "formula": formula,
            "is_molecule": bool(data.get("is_molecule", False)),
            "smiles": data.get("smiles") or None,
            "common_name": data.get("common_name") or None,
            "polymorph_name": data.get("polymorph_name") or None,
            "space_group_hint": data.get("space_group_hint") or None,
            "confidence": parse_confidence_level(data.get("confidence")),
        }

    def _select_phase(
        self,
        query: str,
        formula: str,
        candidates: list[Material],
        polymorph_name: str | None,
        space_group_hint: str | None,
    ) -> dict[str, Any]:
        """Run phase-selection prompt and return the selection dict.

        Args:
            query: Original user query.
            formula: Extracted chemical formula.
            candidates: Ordered MP candidate materials.
            polymorph_name: Target polymorph name (may be ``None``).
            space_group_hint: Expected space group (may be ``None``).

        Returns:
            Dict with keys ``found`` (bool), ``selected_index`` (int | None),
            ``reason`` (str), ``confidence`` (int | None, 1–5 scale).
        """
        prompt = self._loader.render(
            "identification/phase_selection.j2",
            query=query,
            formula=formula,
            polymorph_name=polymorph_name,
            space_group_hint=space_group_hint,
            candidates_json=json.dumps(candidates_to_selection_json(candidates), indent=2),
        )
        data = self._call_llm_json(prompt)
        return parse_phase_selection_response(data, len(candidates))


__all__ = ["MaterialIdentifier", "IdentificationTrace"]
