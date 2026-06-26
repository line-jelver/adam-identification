"""Unit tests for MaterialIdentifier routing and identification logic."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from adam_identification.identifier import MaterialIdentifier
from adam_identification._phase_lookup import formulas_match


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
            mock_pubchem.get_by_smiles.side_effect = pubchem_raises
            mock_pubchem.get_by_name.side_effect = pubchem_raises
            mock_pubchem.get_molecule_candidates.return_value = []
        else:
            mock_pubchem.get_by_smiles.return_value = pubchem_material
            mock_pubchem.get_by_name.return_value = pubchem_material
            mock_pubchem.get_by_cid.return_value = pubchem_material
            mock_pubchem.get_molecule_candidates.return_value = []
    else:
        mock_pubchem = None

    return MaterialIdentifier(llm=mock_llm, mp_client=mock_mp, pubchem_client=mock_pubchem)


def _mock_molecule_material(
    formula: str = "H2O",
    name: str = "water",
    pc_cid: int = 962,
) -> MagicMock:
    mat = MagicMock()
    mat.pc_cid = pc_cid
    mat.name = name
    mat.chemical_formula = formula
    mat.structure.material_type = "molecule"
    return mat


# ── _extract_formula ──────────────────────────────────────────────────────────


def test_extract_formula_crystal() -> None:
    identifier = _make_identifier(
        {
            "formula": "Si",
            "is_molecule": False,
            "smiles": None,
            "common_name": None,
            "polymorph_name": "diamond",
            "space_group_hint": "Fd-3m",
            "confidence": "high",
        }
    )
    result = identifier._extract_formula("silicon")
    assert result["formula"] == "Si"
    assert result["is_molecule"] is False
    assert result["smiles"] is None
    assert result["polymorph_name"] == "diamond"
    assert result["space_group_hint"] == "Fd-3m"
    assert result["confidence"] == 5


def test_extract_formula_molecule() -> None:
    identifier = _make_identifier(
        {
            "formula": "H2O",
            "is_molecule": True,
            "smiles": "O",
            "common_name": "water",
            "polymorph_name": None,
            "space_group_hint": None,
            "confidence": "high",
        }
    )
    result = identifier._extract_formula("water molecule")
    assert result["formula"] == "H2O"
    assert result["is_molecule"] is True
    assert result["smiles"] == "O"
    assert result["common_name"] == "water"
    assert result["polymorph_name"] is None


def test_extract_formula_raises_on_empty() -> None:
    identifier = _make_identifier({"formula": "", "is_molecule": False})
    with pytest.raises(ValueError, match="empty formula"):
        identifier._extract_formula("???")


def test_extract_formula_defaults_is_molecule_false() -> None:
    identifier = _make_identifier(
        {"formula": "Fe2O3", "polymorph_name": None, "confidence": "medium"}
    )
    result = identifier._extract_formula("iron oxide")
    assert result["is_molecule"] is False
    assert result["confidence"] == 3


# ── identify() routing ────────────────────────────────────────────────────────


def test_identify_molecule_routes_to_pubchem() -> None:
    mol_material = _mock_molecule_material(formula="H2O", name="water", pc_cid=962)
    identifier = _make_identifier(
        {"formula": "H2O", "is_molecule": True, "smiles": "O", "common_name": "water"},
        pubchem_material=mol_material,
    )
    result = identifier.identify("water")
    assert result is mol_material
    identifier._pubchem.get_by_smiles.assert_called_once_with("O")
    identifier._mp.search_by_formula.assert_not_called()


def test_identify_molecule_falls_back_to_name_when_smiles_fails() -> None:
    from adam_identification.exceptions import MaterialNotFoundError

    glucose_cid = 5793
    glucose_material = _mock_molecule_material(
        formula="C6H12O6", name="glucose", pc_cid=glucose_cid
    )
    identifier = _make_identifier(
        {
            "formula": "C6H12O6",
            "is_molecule": True,
            "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
            "common_name": "glucose",
        },
        pubchem_material=glucose_material,
    )
    identifier._pubchem.get_by_smiles.side_effect = MaterialNotFoundError("no smiles")
    identifier._pubchem.get_molecule_candidates.return_value = [
        {
            "cid": glucose_cid,
            "name": "glucose",
            "formula": "C6H12O6",
            "smiles": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
        }
    ]
    identifier._pubchem.get_by_cid.return_value = glucose_material

    result = identifier.identify("glucose")
    assert result is glucose_material
    identifier._pubchem.get_by_smiles.assert_called_once()
    identifier._pubchem.get_by_cid.assert_called_once_with(glucose_cid)


def test_identify_molecule_raises_without_pubchem_client() -> None:
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {"formula": "H2O", "is_molecule": True, "smiles": "O", "common_name": "water"}
    )
    mock_llm.complete_single = AsyncMock(return_value=mock_response)
    identifier = MaterialIdentifier(llm=mock_llm, mp_client=MagicMock(), pubchem_client=None)
    with pytest.raises(ValueError, match="PubChemClient"):
        identifier.identify("water")


def test_identify_crystal_does_not_call_pubchem() -> None:
    from adam_identification.exceptions import MaterialNotFoundError

    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = json.dumps(
        {"formula": "Si", "is_molecule": False, "polymorph_name": "diamond"}
    )
    mock_llm.complete_single = AsyncMock(return_value=mock_response)

    mock_mp = MagicMock()
    mock_mp.search_by_formula.return_value = []
    mock_pubchem = MagicMock()

    identifier = MaterialIdentifier(llm=mock_llm, mp_client=mock_mp, pubchem_client=mock_pubchem)
    with pytest.raises(MaterialNotFoundError):
        identifier.identify("silicon")

    mock_pubchem.get_by_smiles.assert_not_called()
    mock_pubchem.get_by_name.assert_not_called()


# ── formulas_match ────────────────────────────────────────────────────────────


def test_formulas_match_same() -> None:
    assert formulas_match("H2O", "H2O") is True


def test_formulas_match_different_notation() -> None:
    assert formulas_match("C2H5OH", "C2H6O") is True


def test_formulas_match_different_composition() -> None:
    assert formulas_match("C6H12O6", "C2H6O") is False
