"""Unit tests for identification candidate-selection prompt modes.

Verifies that standard (default) prompts retain abstention semantics while
minimal-interaction prompts always select plausible candidates and never emit
contradictory ambiguous/null-index instructions.
"""

from __future__ import annotations

import json

import pytest

from adam_identification._prompts import PromptLoader

_PHASE_TEMPLATE = "identification/phase_selection.j2"
_MOLECULE_TEMPLATE = "identification/molecule_candidate_selection.j2"

_SAMPLE_CANDIDATES_JSON = json.dumps(
    [
        {
            "index": 0,
            "material_id": "mp-123",
            "formula": "C",
            "spacegroup": "Fd-3m",
        },
        {
            "index": 1,
            "material_id": "mp-456",
            "formula": "C",
            "spacegroup": "P6_3/mmc",
        },
    ],
    indent=2,
)


def _normalize(text: str) -> str:
    """Collapse whitespace so wrapped prompt lines match contiguous phrases."""

    return " ".join(text.split())


_SAMPLE_MOLECULE_CANDIDATES = [
    {
        "index": 0,
        "cid": 5793,
        "name": "D-glucose",
        "formula": "C6H12O6",
        "isomeric_smiles": "C([C@@H]1[C@H]([C@@H]([C@H](C(O1)O)O)O)O)O",
        "inchi_key": "WQZGKKKJIJFFOK-GASJEMHNSA-N",
    },
    {
        "index": 1,
        "cid": 5790,
        "name": "L-glucose",
        "formula": "C6H12O6",
        "isomeric_smiles": "C([C@H]1[C@@H]([C@H]([C@@H](C(O1)O)O)O)O)O",
        "inchi_key": "WQZGKKKJIJFFOK-RVGMQGWHA-N",
    },
]


@pytest.fixture
def loader() -> PromptLoader:
    """Shared prompt loader for identification templates."""

    return PromptLoader()


def _render_phase(loader: PromptLoader, *, minimal_interaction: bool) -> str:
    return loader.render(
        _PHASE_TEMPLATE,
        query="graphite",
        formula="C",
        candidates_json=_SAMPLE_CANDIDATES_JSON,
        minimal_interaction=minimal_interaction,
    )


def _render_molecule(loader: PromptLoader, *, minimal_interaction: bool) -> str:
    return loader.render(
        _MOLECULE_TEMPLATE,
        query="glucose",
        formula="C6H12O6",
        search_name="glucose",
        candidates=_SAMPLE_MOLECULE_CANDIDATES,
        minimal_interaction=minimal_interaction,
    )


@pytest.mark.parametrize(
    "render_fn",
    [_render_phase, _render_molecule],
    ids=["phase", "molecule"],
)
def test_standard_mode_retains_ambiguous_abstention(
    loader: PromptLoader,
    render_fn: object,
) -> None:
    """Standard prompts document ambiguous abstention and null selected_index."""

    text = _normalize(render_fn(loader, minimal_interaction=False))  # type: ignore[operator]
    assert 'decision="ambiguous"' in text
    assert "select | ambiguous | not_found" in text
    assert "ambiguous or not_found, use selected_index=null" in text
    assert "Return ambiguous when" in text or "return ambiguous" in text


@pytest.mark.parametrize(
    "render_fn",
    [_render_phase, _render_molecule],
    ids=["phase", "molecule"],
)
def test_minimal_mode_forbids_ambiguous_decision(
    loader: PromptLoader,
    render_fn: object,
) -> None:
    """Minimal prompts omit ambiguous from schema and forbid abstention."""

    text = _normalize(render_fn(loader, minimal_interaction=True))  # type: ignore[operator]
    assert "select | not_found" in text
    assert "select | ambiguous | not_found" not in text
    assert "Do not return ambiguous" in text
    assert 'decision="ambiguous"' not in text


@pytest.mark.parametrize(
    "render_fn",
    [_render_phase, _render_molecule],
    ids=["phase", "molecule"],
)
def test_minimal_mode_requires_select_with_valid_index(
    loader: PromptLoader,
    render_fn: object,
) -> None:
    """Minimal mode always selects when plausible candidates exist."""

    text = _normalize(render_fn(loader, minimal_interaction=True))  # type: ignore[operator]
    assert 'always return decision="select"' in text
    assert "provide a valid selected_index" in text
    assert "lower-index candidate when equally defensible" in text


@pytest.mark.parametrize(
    "render_fn",
    [_render_phase, _render_molecule],
    ids=["phase", "molecule"],
)
def test_minimal_mode_no_contradictory_ambiguous_null_rule(
    loader: PromptLoader,
    render_fn: object,
) -> None:
    """Minimal mode must not globally require null index for ambiguous."""

    text = _normalize(render_fn(loader, minimal_interaction=True))  # type: ignore[operator]
    assert "ambiguous or not_found, use selected_index=null" not in text
    assert "For not_found: use selected_index=null" in text


def test_minimal_phase_selection_needs_review_semantics(loader: PromptLoader) -> None:
    """Crystal minimal mode flags unresolved assumptions via needs_review."""

    text = _normalize(_render_phase(loader, minimal_interaction=True))
    assert "needs_review=true" in text
    assert "needs_review does not indicate failure" in text
    assert "state the assumption in selection_reason" in text
    assert "lower-index candidate" in text


def test_minimal_molecule_selection_needs_review_semantics(loader: PromptLoader) -> None:
    """Molecule minimal mode flags unresolved stereochemistry via needs_review."""

    text = _normalize(_render_molecule(loader, minimal_interaction=True))
    assert "needs_review=true" in text
    assert "needs_review does not indicate failure" in text
    assert "stereochemistry, tautomer, protonation" in text
    assert "glucose" in text
    assert "D-glucose" in text


def test_standard_phase_selection_chirality_abstention(loader: PromptLoader) -> None:
    """Standard crystal prompts abstain on unspecified chirality."""

    text = _normalize(_render_phase(loader, minimal_interaction=False))
    assert "return ambiguous and list both" in text


def test_minimal_phase_selection_chirality_tie_break(loader: PromptLoader) -> None:
    """Minimal crystal prompts tie-break chirality at lower index."""

    text = _normalize(_render_phase(loader, minimal_interaction=True))
    assert "select the lower-index candidate" in text
    assert "include both in suggested_candidates" in text


def test_selection_prompts_do_not_encode_followup_wording(loader: PromptLoader) -> None:
    """Follow-up query strings are built in Python, not in the agent prompt."""

    phase = _normalize(_render_phase(loader, minimal_interaction=False))
    molecule = _normalize(_render_molecule(loader, minimal_interaction=False))
    for text in (phase, molecule):
        assert "Follow-up query convention" not in text
        assert "Do not use database IDs" not in text


def test_minimal_mode_preserves_not_found(loader: PromptLoader) -> None:
    """Both templates preserve not_found when identity does not match."""

    phase_text = _normalize(_render_phase(loader, minimal_interaction=True))
    molecule_text = _normalize(_render_molecule(loader, minimal_interaction=True))
    for text in (phase_text, molecule_text):
        assert 'decision="not_found"' in text
        assert "no candidate matches" in text
