"""Material identifier agent for adam-identification.

Takes a natural-language description of a material or molecule and resolves it
to a canonical :class:`~adam_identification.models.Material` using the
database-grounded workflow described in the accompanying publication:

* **Crystal path** — formula extracted by LLM, queried against Materials
  Project (narrow → wide phase selection). Returns a ``Material`` with
  :class:`~adam_identification.models.CrystalStructure`.
* **Molecule path** — LLM signals ``is_molecule=True``; the agent collects up
  to 5 PubChem candidates (SMILES exact-match + name-based multi-candidate),
  lets the LLM select the correct structural isomer, then fetches 3D
  coordinates for the winner. Returns a ``Material`` with
  :class:`~adam_identification.models.MoleculeStructure`.

Example::

    from adam_identification import identify
    atoms = identify("hexagonal boron nitride")  # returns ase.Atoms
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, cast

from adam_identification._confidence import parse_confidence_level
from adam_identification._phase_lookup import (
    INITIAL_MAX_RESULTS,
    WIDE_MAX_RESULTS,
    candidates_to_selection_json,
    formulas_match as _formulas_match,
    molecule_candidates_to_selection_json,
    parse_llm_json_tolerant,
    parse_molecule_selection_response,
    parse_phase_selection_response,
    polymorph_not_found_message,
    single_candidate_selection,
)
from adam_identification._prompts import PromptLoader
from adam_identification.database.materials_project import MaterialsProjectClient
from adam_identification.database.pubchem import MoleculeCandidate, PubChemClient
from adam_identification.exceptions import MaterialNotFoundError
from adam_identification.models import Material
from adam_identification.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class MaterialIdentifier:
    """Resolve a natural-language material description to a :class:`~adam_identification.models.Material`.

    Handles both periodic crystals (via the Materials Project) and discrete
    molecules (via PubChem). The routing decision is made by the LLM during
    formula extraction: ``is_molecule=True`` triggers the PubChem path.

    Crystal identification (three steps):

    1. **Formula extraction** — LLM extracts formula, polymorph hint, and
       ``is_molecule=False``.
    2. **Phase selection (narrow)** — MP searched for up to
       :data:`~adam_identification._phase_lookup.INITIAL_MAX_RESULTS` candidates;
       LLM selects the best match.
    3. **Phase selection (wide, if needed)** — MP re-queried with
       :data:`~adam_identification._phase_lookup.WIDE_MAX_RESULTS` when the
       target polymorph was not in the narrow set.

    Molecule identification (three steps):

    1. **Formula extraction** — LLM extracts formula, SMILES, and
       ``is_molecule=True``.
    2. **Candidate collection** — SMILES exact-match (0–1 hit) + top-5
       PubChem name-search results, deduplicated and formula-filtered.
    3. **Candidate selection** — if more than one candidate survives, the LLM
       selects the correct structural isomer; otherwise the single match is
       accepted directly. 3D coordinates are fetched only for the winner.

    Args:
        llm: Any :class:`~adam_identification.llm.base.BaseLLM` provider.
        mp_client: Configured :class:`~adam_identification.database.materials_project.MaterialsProjectClient`.
        pubchem_client: Optional :class:`~adam_identification.database.pubchem.PubChemClient`.
            Required when molecule queries are expected. If ``None`` and the LLM
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

    def identify(self, query: str) -> Material:
        """Resolve ``query`` to a :class:`~adam_identification.models.Material`.

        Args:
            query: Natural-language description, e.g. ``"silicon"``,
                ``"glucose"``, ``"hexagonal graphite"``.

        Returns:
            A :class:`~adam_identification.models.Material` with
            :class:`~adam_identification.models.CrystalStructure` (crystals) or
            :class:`~adam_identification.models.MoleculeStructure` (molecules).

        Raises:
            MaterialNotFoundError: If the database search returns no usable match.
            DatabaseAPIError: On API/network failures.
            ValueError: If the LLM returns an empty formula, or if
                ``is_molecule=True`` but no ``pubchem_client`` was provided.
        """
        extraction = self._extract_formula(query)
        formula: str = extraction["formula"]
        is_molecule: bool = extraction["is_molecule"]
        logger.info("[%s] formula=%s is_molecule=%s", query, formula, is_molecule)

        if is_molecule:
            return self._identify_molecule(query, extraction)

        # ── Crystal path ────────────────────────────────────────────────────────
        polymorph_name: str | None = extraction.get("polymorph_name")
        space_group_hint: str | None = extraction.get("space_group_hint")
        logger.debug("[%s] polymorph=%s sg=%s", query, polymorph_name, space_group_hint)

        candidates = self._mp.search_by_formula(formula, max_results=INITIAL_MAX_RESULTS)
        if not candidates:
            msg = (
                f"No Materials Project entries found for formula {formula!r} "
                f"(extracted from query {query!r})."
            )
            raise MaterialNotFoundError(msg)

        logger.info("[%s] MP returned %d candidates (narrow)", query, len(candidates))
        selection = self._select_phase(query, formula, candidates, polymorph_name, space_group_hint)

        if selection["found"]:
            idx = int(selection["selected_index"])
            material = candidates[idx]
            logger.info(
                "[%s] selected %s (%s) — %s",
                query,
                material.mp_id,
                getattr(material.structure, "space_group", "?"),
                selection.get("reason", ""),
            )
            return material

        # ── Wide search if polymorph not found in narrow set ─────────────────
        logger.warning(
            "[%s] polymorph not matched in narrow set (%d candidates): %s — widening to %d",
            query,
            len(candidates),
            selection.get("reason", ""),
            WIDE_MAX_RESULTS,
        )
        wide_candidates = self._mp.search_by_formula(formula, max_results=WIDE_MAX_RESULTS)
        logger.info("[%s] MP returned %d candidates (wide)", query, len(wide_candidates))

        wide_selection = self._select_phase(
            query, formula, wide_candidates, polymorph_name, space_group_hint
        )

        if wide_selection["found"]:
            idx = int(wide_selection["selected_index"])
            material = wide_candidates[idx]
            logger.info(
                "[%s] selected %s (%s) after wide search — %s",
                query,
                material.mp_id,
                getattr(material.structure, "space_group", "?"),
                wide_selection.get("reason", ""),
            )
            return material

        msg = polymorph_not_found_message(
            query,
            formula,
            polymorph_name,
            space_group_hint,
            len(wide_candidates),
            str(wide_selection.get("reason", "")),
        )
        raise MaterialNotFoundError(msg)

    # ── Internal helpers ────────────────────────────────────────────────────

    def _identify_molecule(self, query: str, extraction: dict[str, Any]) -> Material:
        if self._pubchem is None:
            raise ValueError(
                f"Query {query!r} was identified as a molecule but no PubChemClient "
                "was provided. Pass pubchem_client= to MaterialIdentifier."
            )

        smiles: str | None = extraction.get("smiles")
        common_name: str | None = extraction.get("common_name")
        formula: str = extraction["formula"]

        # Step 1: SMILES exact-match (0 or 1 result)
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

        # Step 2: Name-based multi-candidate (no 3D coords)
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

        # Step 3: Merge, deduplicate, formula-filter
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

        for c in name_candidates:
            if c["cid"] not in seen_cids:
                seen_cids.add(c["cid"])
                all_candidates.append(c)

        formula_candidates = [
            c for c in all_candidates if _formulas_match(c["formula"], formula)
        ]

        # Step 4: Selection
        if not formula_candidates:
            logger.warning(
                "[%s] No formula-matching candidates from SMILES or name=%r; "
                "falling back to formula string %r",
                query,
                name_to_search,
                formula,
            )
            try:
                fallback = self._pubchem.get_by_name(formula)
                if _formulas_match(fallback.chemical_formula, formula):
                    logger.info(
                        "[%s] Formula-string fallback succeeded (CID %s)",
                        query,
                        fallback.pc_cid,
                    )
                    return fallback
            except MaterialNotFoundError:
                pass
            raise MaterialNotFoundError(
                f"PubChem returned no matching molecule for query {query!r}. "
                f"Tried SMILES={smiles!r}, name={name_to_search!r}, formula={formula!r}."
            )

        if len(formula_candidates) == 1:
            winner_cid = formula_candidates[0]["cid"]
            logger.info(
                "[%s] Single formula-matching candidate: CID %d (%s)",
                query,
                winner_cid,
                formula_candidates[0]["name"],
            )
        else:
            winner_cid = self._select_molecule(query, formula, formula_candidates)

        # Step 5: Fetch 3D structure for winner (reuse SMILES material if it won)
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

    def _select_molecule(
        self,
        query: str,
        formula: str,
        candidates: list[MoleculeCandidate],
    ) -> int:
        prompt = self._loader.render(
            "identification/molecule_candidate_selection.j2",
            query=query,
            formula=formula,
            candidates=molecule_candidates_to_selection_json(candidates),
        )
        data = self._call_llm_json(prompt)
        result = parse_molecule_selection_response(data, len(candidates))
        idx = result["selected_index"]
        winner = candidates[idx]
        logger.info(
            "[%s] LLM selected molecule candidate index %d: CID %d (%s) — %s",
            query,
            idx,
            winner["cid"],
            winner["name"],
            result.get("reason", ""),
        )
        return int(winner["cid"])

    def _call_llm_json(self, prompt: str) -> dict[str, Any]:
        async def _call() -> dict[str, Any]:
            response = await self._llm.complete_single(prompt, response_format="json")
            return cast(dict[str, Any], parse_llm_json_tolerant(response.content))

        return cast(dict[str, Any], asyncio.run(_call()))

    def _extract_formula(self, query: str) -> dict[str, Any]:
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
        shortcut = single_candidate_selection(candidates, polymorph_name, space_group_hint)
        if shortcut is not None:
            return shortcut

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
