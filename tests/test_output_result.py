"""Smoke tests for identify(..., output='result') and IdentificationResult."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adam_identification import identify
from adam_identification.result import IdentificationResult
from adam_identification.trace import IdentificationTrace


def test_identify_output_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """output='result' returns IdentificationResult with atoms, material, and trace."""
    material = MagicMock()
    atoms = MagicMock()
    trace = IdentificationTrace(query="silicon", domain="crystal", outcome="selected")

    class FakeIdentifier:
        def __init__(
            self,
            llm: object,
            mp_client: object,
            pubchem_client: object | None = None,
            minimal_interaction: bool = False,
        ) -> None:
            self.minimal_interaction = minimal_interaction

        def identify(
            self, query: str, *, return_trace: bool = False
        ) -> object | tuple[object, IdentificationTrace]:
            assert query == "silicon"
            if return_trace:
                return material, trace
            return material

    monkeypatch.setattr("adam_identification.MaterialIdentifier", FakeIdentifier)
    monkeypatch.setattr("adam_identification.llm.get_provider", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        "adam_identification.database.materials_project.MaterialsProjectClient",
        lambda *a, **k: MagicMock(),
    )
    monkeypatch.setattr(
        "adam_identification.database.pubchem.PubChemClient",
        lambda *a, **k: MagicMock(),
    )
    monkeypatch.setattr("adam_identification.output.to_ase", lambda _m: atoms)

    result = identify("silicon", model="test-model", output="result", minimal_interaction=True)
    assert isinstance(result, IdentificationResult)
    assert result.atoms is atoms
    assert result.material is material
    assert result.trace is trace
    assert result.trace.outcome == "selected"
    assert result.trace.needs_review is False


def test_identify_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public identify() does not pick a default model."""
    from adam_identification._config import ConfigurationError

    with pytest.raises(ConfigurationError, match="no default"):
        identify("silicon")
