"""Unit tests for shared material-identification phase lookup helpers."""

from __future__ import annotations

from adam_identification.models import (
    AtomicPosition,
    CrystalStructure,
    Material,
    MaterialSource,
    MaterialsProjectProperties,
)
from adam_identification._phase_lookup import (
    INITIAL_MAX_RESULTS,
    WIDE_MAX_RESULTS,
    candidates_to_selection_json,
    formulas_match,
    parse_phase_selection_response,
    polymorph_not_found_message,
    single_candidate_selection,
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
        atomic_positions=[AtomicPosition(element="Si", position=(0.0, 0.0, 0.0))],
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


def test_search_limits() -> None:
    assert INITIAL_MAX_RESULTS == 20
    assert WIDE_MAX_RESULTS == 50


def test_single_candidate_selection_without_hints() -> None:
    candidates = [_make_material("mp-1", "Si", "Fd-3m")]
    result = single_candidate_selection(candidates, None, None)
    assert result is not None
    assert result["found"] is True
    assert result["selected_index"] == 0


def test_single_candidate_selection_skipped_with_polymorph_hint() -> None:
    candidates = [_make_material("mp-1", "C", "P6_3/mmc")]
    assert single_candidate_selection(candidates, "diamond", "Fd-3m") is None


def test_parse_phase_selection_respects_found_false() -> None:
    parsed = parse_phase_selection_response(
        {"found": False, "selected_index": 3, "reason": "no match"},
        n_candidates=10,
    )
    assert parsed["found"] is False
    assert parsed["selected_index"] == 3


def test_parse_phase_selection_clamps_index() -> None:
    parsed = parse_phase_selection_response(
        {"found": True, "selected_index": 99, "reason": "picked"},
        n_candidates=5,
    )
    assert parsed["found"] is True
    assert parsed["selected_index"] == 4


def test_candidates_to_selection_json_compact_payload() -> None:
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


def test_formulas_match_same() -> None:
    assert formulas_match("H2O", "H2O") is True


def test_formulas_match_different_notation() -> None:
    assert formulas_match("C2H5OH", "C2H6O") is True


def test_formulas_match_different_composition() -> None:
    assert formulas_match("C6H12O6", "C2H6O") is False


def test_polymorph_not_found_message_includes_hints() -> None:
    msg = polymorph_not_found_message("diamond", "C", "diamond", "Fd-3m", 50, "not in list")
    assert "diamond" in msg
    assert "Fd-3m" in msg
    assert "50 candidates" in msg
