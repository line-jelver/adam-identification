"""Unit tests for shared material-identification phase lookup helpers."""

from __future__ import annotations

from adam_identification._followup import format_followup_queries
from adam_identification._phase_lookup import (
    INITIAL_MAX_RESULTS,
    WIDE_MAX_RESULTS,
    candidates_to_selection_json,
    material_to_scoring_payload,
    parse_molecule_selection_response,
    parse_phase_selection_response,
    phase_not_found_message,
)
from adam_identification.models import (
    AtomicPosition,
    CrystalStructure,
    Material,
    MaterialSource,
    MaterialsProjectProperties,
)


def _make_material(
    mp_id: str,
    formula: str,
    space_group: str,
    *,
    energy_above_hull: float = 0.0,
) -> Material:
    structure = CrystalStructure(
        unit_cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        lattice_parameters={
            "a": 3.0,
            "b": 3.0,
            "c": 3.0,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        atomic_positions=[
            AtomicPosition(element="Si", position=(0.0, 0.0, 0.0)),
        ],
        crystal_system="Cubic",
        space_group=space_group,
        nelements=1,
        nsites=1,
    )
    return Material(
        name=formula,
        chemical_formula=formula,
        source=MaterialSource.MATERIALS_PROJECT,
        mp_id=mp_id,
        structure=structure,
        properties=[
            MaterialsProjectProperties(
                energy_above_hull=energy_above_hull,
                is_metal=False,
                band_gap=1.1,
                is_direct_gap=False,
            )
        ],
    )


def test_search_limits_match_production_policy() -> None:
    assert INITIAL_MAX_RESULTS == 20
    assert WIDE_MAX_RESULTS == 50


def test_parse_phase_selection_response_not_found() -> None:
    parsed = parse_phase_selection_response(
        {"decision": "not_found", "selected_index": None, "selection_reason": "no match"},
        n_candidates=10,
    )
    assert parsed["decision"] == "not_found"
    assert parsed["selected_index"] is None
    assert parsed["selection_reason"] == "no match"


def test_parse_phase_selection_response_select_clamps_index() -> None:
    parsed = parse_phase_selection_response(
        {"decision": "select", "selected_index": 99, "selection_reason": "picked"},
        n_candidates=5,
    )
    assert parsed["decision"] == "select"
    assert parsed["selected_index"] == 4


def test_parse_phase_selection_response_ambiguous() -> None:
    parsed = parse_phase_selection_response(
        {
            "decision": "ambiguous",
            "selected_index": None,
            "selection_reason": "multiple phases",
            "user_message": "Please specify the polymorph.",
            "suggested_candidates": [{"index": 0, "suggested_query": "rutile TiO2"}],
        },
        n_candidates=3,
    )
    assert parsed["decision"] == "ambiguous"
    assert parsed["selected_index"] is None
    assert parsed["user_message"] == "Please specify the polymorph."
    assert len(parsed["suggested_candidates"]) == 1


def test_parse_phase_selection_response_unknown_decision_defaults_to_not_found() -> None:
    parsed = parse_phase_selection_response(
        {"decision": "bogus", "selected_index": 0},
        n_candidates=3,
    )
    assert parsed["decision"] == "not_found"


def test_candidates_to_selection_json_uses_compact_payload() -> None:
    payload = candidates_to_selection_json([_make_material("mp-66", "C", "Fd-3m")])
    assert payload == [
        {
            "index": 0,
            "mp_id": "mp-66",
            "formula": "C",
            "space_group": "Fd-3m",
            "crystal_system": "Cubic",
            "nsites": 1,
            "energy_above_hull": 0.0,
            "is_metal": False,
        }
    ]


def test_material_to_scoring_payload_includes_geometry() -> None:
    payload = material_to_scoring_payload(_make_material("mp-149", "Si", "Fd-3m"))
    assert payload["material_id"] == "mp-149"
    assert payload["lattice_parameters"]["a"] == 3.0
    assert payload["atomic_positions"][0]["element"] == "Si"


def test_phase_not_found_message_includes_info() -> None:
    msg = phase_not_found_message(
        "diamond",
        "C",
        50,
        "not in list",
    )
    assert "diamond" in msg
    assert "50 candidates" in msg
    assert "not in list" in msg


def test_parse_molecule_selection_not_found() -> None:
    data = {
        "decision": "not_found",
        "selected_index": None,
        "selection_reason": "no stereo match",
        "confidence": 2,
    }
    result = parse_molecule_selection_response(data, n_candidates=3)
    assert result["decision"] == "not_found"
    assert result["selected_index"] is None


def test_parse_molecule_selection_ambiguous() -> None:
    data = {
        "decision": "ambiguous",
        "selected_index": None,
        "selection_reason": "multiple isomers",
        "user_message": "Which xylene isomer?",
        "suggested_candidates": [{"index": 0, "suggested_query": "o-xylene"}],
    }
    result = parse_molecule_selection_response(data, n_candidates=3)
    assert result["decision"] == "ambiguous"
    assert result["selected_index"] is None
    assert result["user_message"] == "Which xylene isomer?"


def test_parse_molecule_selection_select() -> None:
    data = {
        "decision": "select",
        "selected_index": 0,
        "selection_reason": "only match",
        "confidence": 5,
    }
    result = parse_molecule_selection_response(data, n_candidates=2)
    assert result["decision"] == "select"
    assert result["selected_index"] == 0


def test_format_followup_queries_crystal_aliases_and_formula_space_group() -> None:
    """Crystal follow-ups use aliases when known, else formula plus space group."""
    candidates = [
        {"index": 0, "mp_id": "mp-2657", "formula": "TiO2", "space_group": "P4_2/mnm"},
        {"index": 1, "mp_id": "mp-390", "formula": "TiO2", "space_group": "I4_1/amd"},
        {"index": 2, "mp_id": "mp-149", "formula": "Si", "space_group": "Fd-3m"},
        {"index": 3, "mp_id": "mp-604884", "formula": "BN", "space_group": "P6_3/mmc"},
    ]
    formatted = format_followup_queries(
        [
            {"index": 0, "suggested_query": "mp-2657"},
            {"index": 1, "suggested_query": "the anatase phase"},
            {"index": 2, "suggested_query": "silicon mp-149"},
            {"index": 3, "suggested_query": "hexagonal boron nitride"},
        ],
        candidates,
        domain="crystal",
    )
    assert formatted == [
        {"index": 0, "suggested_query": "rutile TiO2"},
        {"index": 1, "suggested_query": "anatase TiO2"},
        {"index": 2, "suggested_query": "Si Fd-3m"},
        {"index": 3, "suggested_query": "h-BN P6_3/mmc"},
    ]
    assert all("mp-" not in row["suggested_query"] for row in formatted)


def test_format_followup_queries_molecule_uses_name_not_cid() -> None:
    """Molecule follow-ups use the PubChem name, never the CID."""
    formatted = format_followup_queries(
        [
            {"index": 0, "suggested_query": "CID 11583"},
            {"index": 1, "suggested_query": "xylene isomer 2"},
        ],
        [
            {"index": 0, "cid": 11583, "name": "o-xylene", "formula": "C8H10"},
            {"index": 1, "cid": 7929, "name": "m-xylene", "formula": "C8H10"},
        ],
        domain="molecule",
    )
    assert formatted == [
        {"index": 0, "suggested_query": "o-xylene"},
        {"index": 1, "suggested_query": "m-xylene"},
    ]


def test_format_followup_queries_drops_invalid_indices() -> None:
    """Suggestions without a usable candidate index are discarded."""
    formatted = format_followup_queries(
        [
            {"index": 9, "suggested_query": "rutile TiO2"},
            {"suggested_query": "anatase TiO2"},
            "not a dict",
            {"index": 0, "suggested_query": "mp-2657"},
        ],
        [{"formula": "TiO2", "space_group": "P4_2/mnm"}],
        domain="crystal",
    )
    assert formatted == [{"index": 0, "suggested_query": "rutile TiO2"}]
