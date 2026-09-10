"""Smoke tests for identify(..., output='result') and IdentificationResult."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from adam_identification import identify
from adam_identification.provenance.trace import Trace, TraceStatus
from adam_identification.result import IdentificationResult


def test_identify_output_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """output='result' returns IdentificationResult with atoms, material, and trace."""
    material = MagicMock()
    atoms = MagicMock()
    trace = Trace(query="silicon", status=TraceStatus.COMPLETED)
    trace.identification.outcome = "selected"

    def fake_run(query: str, **kwargs: object) -> tuple[object, Trace]:
        assert query == "silicon"
        assert kwargs.get("minimal_interaction") is True
        return material, trace

    monkeypatch.setattr("adam_identification.provenance.lifecycle.run_identification", fake_run)
    monkeypatch.setattr("adam_identification.output.to_ase", lambda _m: atoms)

    result = identify("silicon", model="test-model", output="result", minimal_interaction=True)
    assert isinstance(result, IdentificationResult)
    assert result.atoms is atoms
    assert result.material is material
    assert result.trace is trace
    assert result.trace.identification.outcome == "selected"
    assert result.trace.needs_review is False


def test_identify_requires_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Public identify() does not pick a default model."""
    from adam_identification._config import ConfigurationError

    with pytest.raises(ConfigurationError, match="no default"):
        identify("silicon")
