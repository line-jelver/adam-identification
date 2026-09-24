"""Unit tests for MaterialIdentifier formula extraction and material-type routing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from adam_identification.exceptions import (
    AmbiguousIdentificationError,
    ClarificationNeededError,
    MaterialNotFoundError,
)
from adam_identification.identifier import MaterialIdentifier, _formulas_match
from adam_identification.provenance.trace import Trace

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_identifier(
    llm_response_json: dict,
    *,
    pubchem_material: object | None = None,
    pubchem_raises: Exception | None = None,
) -> MaterialIdentifier:
    """Build a MaterialIdentifier with mocked LLM, MP client, and optionally PubChem."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(llm_response_json)
    mock_llm.complete_single = AsyncMock(return_value=mock_response)

    mock_mp = MagicMock()

    if pubchem_material is not None or pubchem_raises is not None:
        mock_pubchem = MagicMock()
        if pubchem_raises is not None:
            mock_pubchem.get_by_name.side_effect = pubchem_raises
            mock_pubchem.get_molecule_candidates.return_value = []
        else:
            mock_pubchem.get_by_name.return_value = pubchem_material
            mock_pubchem.get_by_cid.return_value = pubchem_material
            mock_pubchem.get_molecule_candidates.return_value = []
    else:
        mock_pubchem = None

    return MaterialIdentifier(llm=mock_llm, crystal_retriever=mock_mp, pubchem_client=mock_pubchem)


def _trace(query: str) -> Trace:
    """Fresh provenance Trace for identifier unit tests."""
    return Trace(query=query)


def _mock_molecule_material(
    formula: str = "H2O",
    name: str = "water",
    pc_cid: int = 962,
) -> MagicMock:
    """Minimal Material mock with MoleculeStructure metadata.

    Args:
        formula: ``chemical_formula`` string (must parse correctly as a formula).
        name: Common name returned by the mock.
        pc_cid: PubChem CID integer.
    """
    mat = MagicMock()
    mat.pc_cid = pc_cid
    mat.name = name
    mat.chemical_formula = formula
    mat.structure.material_type = "molecule"
    return mat


# ── _extract_formula tests ────────────────────────────────────────────────────


def test_extract_formula_crystal() -> None:
    identifier = _make_identifier(
        {
            "decision": "proceed",
            "domain": "crystal",
            "formula": "Si",
            "search_name": None,
            "confidence": "high",
        }
    )
    result = identifier._extract_formula("silicon")
    assert result["formula"] == "Si"
    assert result["domain"] == "crystal"
    assert result["decision"] == "proceed"
    assert result["search_name"] is None
    assert result["confidence"] == 5


def test_extract_formula_molecule() -> None:
    identifier = _make_identifier(
        {
            "decision": "proceed",
            "domain": "molecule",
            "formula": "H2O",
            "search_name": "water",
            "confidence": "high",
        }
    )
    result = identifier._extract_formula("water molecule")
    assert result["formula"] == "H2O"
    assert result["domain"] == "molecule"
    assert result["decision"] == "proceed"
    assert result["search_name"] == "water"


def test_extract_formula_raises_on_empty_formula() -> None:
    identifier = _make_identifier({"decision": "proceed", "domain": "crystal", "formula": ""})
    with pytest.raises(ValueError, match="empty formula"):
        identifier._extract_formula("???")


def test_extract_formula_defaults_domain_crystal() -> None:
    identifier = _make_identifier(
        {"decision": "proceed", "formula": "Fe2O3", "confidence": "medium"}
    )
    result = identifier._extract_formula("iron oxide")
    assert result["domain"] == "crystal"
    assert result["confidence"] == 3


def test_extract_formula_clarify_raises_clarification_error() -> None:
    """decision='clarify' at extraction raises ClarificationNeededError."""
    identifier = _make_identifier(
        {
            "decision": "clarify",
            "domain": "unknown",
            "formula": None,
            "user_message": "Please specify which iron oxide you mean.",
            "suggested_queries": ["Fe2O3 hematite", "Fe3O4 magnetite"],
        }
    )
    with pytest.raises(ClarificationNeededError) as exc_info:
        identifier.identify("iron oxide", trace=_trace("iron oxide"))
    err = exc_info.value
    assert "iron oxide" in err.user_message
    assert len(err.suggested_queries) == 2


# ── identify() routing tests ──────────────────────────────────────────────────


def test_identify_molecule_routes_to_pubchem() -> None:
    """identify() with domain='molecule' calls PubChem, not MP."""
    mol_material = _mock_molecule_material(formula="H2O", name="water", pc_cid=962)
    mock_llm = MagicMock()
    # First call: extraction; Second call: molecule selection
    extraction = {
        "decision": "proceed",
        "domain": "molecule",
        "formula": "H2O",
        "search_name": "water",
    }
    selection = {
        "decision": "select",
        "selected_index": 0,
        "selection_reason": "water is unambiguous",
        "confidence": 5,
    }
    responses = [
        MagicMock(content=json.dumps(extraction)),
        MagicMock(content=json.dumps(selection)),
    ]
    mock_llm.complete_single = AsyncMock(side_effect=responses)

    mock_pubchem = MagicMock()
    mock_pubchem.get_molecule_candidates.return_value = [
        {"cid": 962, "name": "water", "formula": "H2O", "smiles": "O"},
    ]
    mock_pubchem.get_by_cid.return_value = mol_material

    mock_mp = MagicMock()
    identifier = MaterialIdentifier(
        llm=mock_llm, crystal_retriever=mock_mp, pubchem_client=mock_pubchem
    )

    trace = _trace("water")
    result = identifier.identify("water", trace=trace)
    assert result is mol_material
    mock_pubchem.get_molecule_candidates.assert_called_once_with("water")
    mock_pubchem.get_by_cid.assert_called_once_with(962)
    mock_mp.search.assert_not_called()
    assert trace.identification.domain == "molecule"
    assert trace.identification.n_candidates_narrow == 1
    assert trace.identification.candidates_shown_narrow[0]["cid"] == 962
    assert trace.identification.narrow_selection is not None
    assert trace.identification.outcome == "selected"


def test_identify_molecule_raises_when_no_pubchem_client() -> None:
    """identify() raises ValueError when molecule detected but no PubChemClient given."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {"decision": "proceed", "domain": "molecule", "formula": "H2O", "search_name": "water"}
    )
    mock_llm.complete_single = AsyncMock(return_value=mock_response)
    identifier = MaterialIdentifier(
        llm=mock_llm, crystal_retriever=MagicMock(), pubchem_client=None
    )

    with pytest.raises(ValueError, match="PubChemClient"):
        identifier.identify("water", trace=_trace("water"))


def test_identify_crystal_does_not_call_pubchem() -> None:
    """Crystal query never touches PubChem, even when client is present."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {"decision": "proceed", "domain": "crystal", "formula": "Si"}
    )
    mock_llm.complete_single = AsyncMock(return_value=mock_response)

    mock_mp = MagicMock()
    mock_mp.search.return_value.candidates = []
    mock_pubchem = MagicMock()

    identifier = MaterialIdentifier(
        llm=mock_llm, crystal_retriever=mock_mp, pubchem_client=mock_pubchem
    )
    with pytest.raises(MaterialNotFoundError):
        identifier.identify("silicon", trace=_trace("silicon"))

    mock_pubchem.get_by_name.assert_not_called()


def test_ambiguous_molecule_raises_ambiguous_identification_error() -> None:
    """When LLM returns decision='ambiguous', raise AmbiguousIdentificationError."""
    extraction = {
        "decision": "proceed",
        "domain": "molecule",
        "formula": "C8H10",
        "search_name": "xylene",
        "confidence": 4,
    }
    selection = {
        "decision": "ambiguous",
        "selected_index": None,
        "selection_reason": "multiple xylene isomers — ortho, meta, para",
        "user_message": "Please specify which xylene isomer: o-xylene, m-xylene, or p-xylene.",
        "suggested_candidates": [
            {"index": 0, "suggested_query": "CID 11583"},
            {"index": 1, "suggested_query": "CID 7929"},
        ],
        "confidence": 2,
    }
    mock_llm = MagicMock()
    responses = [
        MagicMock(content=json.dumps(extraction)),
        MagicMock(content=json.dumps(selection)),
    ]
    mock_llm.complete_single = AsyncMock(side_effect=responses)

    mock_pubchem = MagicMock()
    mock_pubchem.get_molecule_candidates.return_value = [
        {"cid": 11583, "name": "o-xylene", "formula": "C8H10", "smiles": "Cc1ccccc1C"},
        {"cid": 7929, "name": "m-xylene", "formula": "C8H10", "smiles": "Cc1cccc(C)c1"},
    ]

    identifier = MaterialIdentifier(
        llm=mock_llm, crystal_retriever=MagicMock(), pubchem_client=mock_pubchem
    )
    with pytest.raises(AmbiguousIdentificationError) as exc_info:
        identifier.identify("xylene", trace=_trace("xylene"))
    err = exc_info.value
    assert err.domain == "molecule"
    assert len(err.candidates) == 2
    assert err.user_message is not None
    assert err.suggested_candidates == [
        {"index": 0, "suggested_query": "o-xylene"},
        {"index": 1, "suggested_query": "m-xylene"},
    ]


def test_ambiguous_error_str_contains_user_message() -> None:
    err = AmbiguousIdentificationError(
        "Ambiguous",
        query="xylene",
        formula="C8H10",
        domain="molecule",
        candidates=[
            {"cid": 11583, "name": "o-xylene", "formula": "C8H10", "smiles": "Cc1ccccc1C"},
            {"cid": 7929, "name": "m-xylene", "formula": "C8H10", "smiles": "Cc1cccc(C)c1"},
        ],
        user_message="Please specify which xylene isomer.",
    )
    text = str(err)
    assert "xylene isomer" in text
    assert "o-xylene (CID 11583)" in text
    assert "m-xylene (CID 7929)" in text


def test_ambiguous_error_str_candidate_table_fallback() -> None:
    """Without user_message, __str__ still lists candidate names and CIDs."""
    err = AmbiguousIdentificationError(
        "Ambiguous",
        query="xylene",
        formula="C8H10",
        domain="molecule",
        candidates=[
            {"cid": 11583, "name": "o-xylene", "formula": "C8H10", "smiles": "Cc1ccccc1C"},
            {"cid": 7929, "name": "m-xylene", "formula": "C8H10", "smiles": "Cc1cccc(C)c1"},
        ],
    )
    text = str(err)
    assert "Candidates include: o-xylene, m-xylene." in text
    assert "PubChem entries" in text
    assert "CID 11583" in text


def test_ambiguous_error_str_lists_crystal_options_with_user_message() -> None:
    """Crystal ambiguity always lists mp_id / space group, even with user_message."""
    err = AmbiguousIdentificationError(
        "Ambiguous",
        query="TiO2",
        formula="TiO2",
        domain="crystal",
        candidates=[
            {
                "mp_id": "mp-2657",
                "formula": "TiO2",
                "space_group": "P4_2/mnm",
                "nsites": 6,
            },
            {
                "mp_id": "mp-390",
                "formula": "TiO2",
                "space_group": "I4_1/amd",
                "nsites": 4,
            },
        ],
        user_message="Please specify which TiO2 polymorph.",
        suggested_candidates=[
            {"index": 0, "suggested_query": "rutile TiO2"},
        ],
    )
    text = str(err)
    assert "Please specify which TiO2 polymorph." in text
    assert "rutile TiO2" in text
    assert "mp-2657" in text
    assert "P4_2/mnm" in text
    assert "mp-390" in text
    assert "I4_1/amd" in text


def test_candidates_for_display_caps_and_skips_bulky_p1() -> None:
    """User-facing tables keep suggested rows and unique space groups, not 20 P1 cells."""
    candidates = [
        {"mp_id": "mp-390", "formula": "TiO2", "space_group": "I4_1/amd", "nsites": 6},
        {"mp_id": "mp-2657", "formula": "TiO2", "space_group": "P4_2/mnm", "nsites": 6},
        {"mp_id": "mp-1840", "formula": "TiO2", "space_group": "Pbca", "nsites": 24},
    ]
    for i in range(17):
        candidates.append(
            {
                "mp_id": f"mp-p1-{i}",
                "formula": "TiO2",
                "space_group": "P1",
                "nsites": 100,
            }
        )
    err = AmbiguousIdentificationError(
        "Ambiguous",
        query="TiO2",
        formula="TiO2",
        domain="crystal",
        candidates=candidates,
        suggested_candidates=[
            {"index": 1, "suggested_query": "rutile TiO2"},
            {"index": 0, "suggested_query": "anatase TiO2"},
        ],
    )
    shown = err.candidates_for_display()
    mp_ids = [row["mp_id"] for row in shown]
    assert mp_ids[0] == "mp-2657"
    assert mp_ids[1] == "mp-390"
    assert "mp-1840" in mp_ids
    assert all(not str(mid).startswith("mp-p1-") for mid in mp_ids)
    assert len(shown) <= 8
    assert "8 of 20" in str(err) or "of 20" in str(err)
    assert len(err.candidates_for_display(verbose=True)) == 20


# ── _formulas_match tests ─────────────────────────────────────────────────────


def test_formulas_match_same_formula() -> None:
    assert _formulas_match("H2O", "H2O") is True


def test_formulas_match_different_notation() -> None:
    # C2H5OH and C2H6O represent the same composition
    assert _formulas_match("C2H5OH", "C2H6O") is True


def test_formulas_match_different_composition() -> None:
    # C6H12O6 (glucose) vs C2H6O (ethanol)
    assert _formulas_match("C6H12O6", "C2H6O") is False


def test_formulas_match_isomers_same_formula() -> None:
    # Glucose and fructose have the same formula — correctly reported as matching
    assert _formulas_match("C6H12O6", "C6H12O6") is True
