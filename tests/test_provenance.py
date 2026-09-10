"""FAIR provenance Trace, store, lifecycle, session linking, and LLM recording."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from adam_identification._config import ConfigurationError
from adam_identification.exceptions import AmbiguousIdentificationError
from adam_identification.identifier import MaterialIdentifier
from adam_identification.llm.base import BaseLLM, LLMResponse, Message
from adam_identification.models import (
    AtomicPosition,
    CrystalStructure,
    Material,
    MaterialSource,
)
from adam_identification.provenance.context import (
    current_stage,
    llm_purpose,
    set_stage,
    set_trace,
)
from adam_identification.provenance.lifecycle import run_identification, scoped_identify
from adam_identification.provenance.store import TraceStore
from adam_identification.provenance.trace import IdentificationStage, Trace, TraceStatus
from adam_identification.session import IdentificationSession


def _si_material() -> Material:
    structure = CrystalStructure(
        unit_cell=[[5.43, 0.0, 0.0], [0.0, 5.43, 0.0], [0.0, 0.0, 5.43]],
        lattice_parameters={
            "a": 5.43,
            "b": 5.43,
            "c": 5.43,
            "alpha": 90.0,
            "beta": 90.0,
            "gamma": 90.0,
        },
        atomic_positions=[AtomicPosition(element="Si", position=(0.0, 0.0, 0.0))],
        crystal_system="Cubic",
        space_group="Fd-3m",
        nelements=1,
        nsites=1,
    )
    return Material(
        name="silicon",
        chemical_formula="Si",
        source=MaterialSource.MATERIALS_PROJECT,
        mp_id="mp-149",
        structure=structure,
    )


def _select_fn(query: str, trace: Trace) -> Material:
    trace.identification.domain = "crystal"
    trace.identification.outcome = "selected"
    return _si_material()


def test_schema_file_matches_model() -> None:
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "adam_identification"
        / "schemas"
        / "trace.schema.json"
    )
    committed = json.loads(schema_path.read_text(encoding="utf-8"))
    assert committed == Trace.model_json_schema()
    text = json.dumps(committed)
    assert "candidates_shown_molecule" not in text
    assert "molecule_selection" not in text
    assert "candidates_shown_narrow" in text
    assert "adam_version" not in text
    assert "adam_identification_version" in text


def test_load_preserves_running(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    trace = Trace(query="silicon")
    assert trace.status == TraceStatus.RUNNING
    store.save(trace)
    loaded = TraceStore.load(tmp_path / "adam.json")
    assert loaded.status == TraceStatus.RUNNING
    assert loaded.finished_at is None


def test_get_provider_failure_writes_initialization_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise ConfigurationError("API key missing")

    monkeypatch.setattr("adam_identification.llm.get_provider", boom)
    with pytest.raises(ConfigurationError):
        run_identification("silicon", model="test-model", work_dir=tmp_path)
    loaded = TraceStore.load(tmp_path / "adam.json")
    assert loaded.status == TraceStatus.FAILED
    assert loaded.failures
    assert loaded.failures[0].stage == IdentificationStage.INITIALIZATION
    assert loaded.failures[0].exc_type == "ConfigurationError"


def test_cif_write_failure_is_output_conversion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("ase.io.write", fail_write)
    with pytest.raises(OSError):
        scoped_identify(
            "silicon",
            _select_fn,
            work_dir=tmp_path,
            write_artifacts=True,
        )
    loaded = TraceStore.load(tmp_path / "adam.json")
    assert loaded.status == TraceStatus.FAILED
    assert loaded.failures
    assert loaded.failures[0].stage == IdentificationStage.OUTPUT_CONVERSION
    assert loaded.status != TraceStatus.COMPLETED


def test_success_records_artifacts_at_both_paths(tmp_path: Path) -> None:
    work = tmp_path / "work"
    user = tmp_path / "user" / "silicon.cif"
    _material, trace = scoped_identify(
        "silicon",
        _select_fn,
        work_dir=work,
        extra_save_paths=[user],
        write_artifacts=True,
    )
    assert trace.status == TraceStatus.COMPLETED
    loaded = TraceStore.load(work / "adam.json")
    paths = {Path(art.path) for art in loaded.artifacts}
    assert user.resolve() in paths
    assert (work / "silicon.cif").resolve() in paths
    assert user.exists()
    assert (work / "silicon.cif").exists()
    assert loaded.identification.outcome == "selected"


def test_keyboard_interrupt_records_failure(tmp_path: Path) -> None:
    def boom(_query: str, _trace: Trace) -> Material:
        set_stage(IdentificationStage.LLM)
        assert current_stage() == IdentificationStage.LLM
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        scoped_identify("silicon", boom, work_dir=tmp_path)
    loaded = TraceStore.load(tmp_path / "adam.json")
    assert loaded.status == TraceStatus.FAILED
    assert loaded.failures
    assert loaded.failures[0].exc_type == "KeyboardInterrupt"
    assert loaded.failures[0].stage == IdentificationStage.LLM


def test_protocol_ambiguous_is_completed_without_failure(tmp_path: Path) -> None:
    def ambiguous(_query: str, trace: Trace) -> Material:
        trace.identification.outcome = "ambiguous"
        raise AmbiguousIdentificationError(
            "ambiguous",
            query=_query,
            formula="TiO2",
            domain="crystal",
        )

    with pytest.raises(AmbiguousIdentificationError) as exc_info:
        scoped_identify("TiO2", ambiguous, work_dir=tmp_path)
    assert exc_info.value.trace is not None
    loaded = TraceStore.load(tmp_path / "adam.json")
    assert loaded.status == TraceStatus.COMPLETED
    assert loaded.identification.outcome == "ambiguous"
    assert loaded.failures == []
    assert loaded.run_id == exc_info.value.trace.run_id


def test_session_linked_traces(monkeypatch: pytest.MonkeyPatch) -> None:
    identifier = MagicMock()

    def first_identify(query: str, *, trace: Trace) -> Material:
        trace.identification.domain = "crystal"
        trace.identification.extraction = {"formula": "BN", "domain": "crystal"}
        trace.identification.n_candidates_narrow = 2
        trace.identification.candidates_shown_narrow = [
            {"mp_id": "mp-1", "formula": "BN", "space_group": "P6_3/mmc"},
            {"mp_id": "mp-2", "formula": "BN", "space_group": "F-43m"},
        ]
        trace.identification.outcome = "ambiguous"
        raise AmbiguousIdentificationError(
            "ambiguous",
            query=query,
            formula="BN",
            domain="crystal",
            candidates=trace.identification.candidates_shown_narrow,
        )

    identifier.identify.side_effect = first_identify
    traces: list[Trace] = []
    from adam_identification import session as session_mod

    orig_scoped = session_mod.scoped_identify

    def wrapping(
        query: str,
        identify_fn: object,
        **kwargs: object,
    ) -> tuple[Material, Trace]:
        try:
            material, trace = orig_scoped(query, identify_fn, **kwargs)  # type: ignore[arg-type]
            traces.append(trace)
            return material, trace
        except Exception as exc:
            attached = getattr(exc, "trace", None)
            if isinstance(attached, Trace):
                traces.append(attached)
            raise

    monkeypatch.setattr(session_mod, "scoped_identify", wrapping)

    session = IdentificationSession(identifier)
    with pytest.raises(AmbiguousIdentificationError) as exc_info:
        session.identify("boron nitride")
    parent = exc_info.value.trace
    assert parent is not None
    assert parent.status == TraceStatus.COMPLETED
    assert parent.identification.outcome == "ambiguous"
    parent_dump = parent.model_dump()

    material = _si_material()
    identifier._mp.search_by_formula.return_value = [material]
    identifier._select_phase.return_value = {
        "decision": "select",
        "selected_index": 0,
        "selection_reason": "hexagonal BN",
        "suggested_candidates": [],
    }

    result = session.resume("h-BN P6_3/mmc")
    assert result is material
    assert len(traces) == 2
    child = traces[1]
    assert child.run_id != parent.run_id
    assert child.parent_run_id == parent.run_id
    assert parent.model_dump() == parent_dump
    assert child.status == TraceStatus.COMPLETED
    assert child.identification.outcome == "selected"


def test_resume_does_not_rewrite_parent_file(tmp_path: Path) -> None:
    parent_dir = tmp_path / "parent"
    child_dir = tmp_path / "child"

    def ambiguous(_query: str, trace: Trace) -> Material:
        trace.identification.outcome = "ambiguous"
        raise AmbiguousIdentificationError(
            "ambiguous", query=_query, formula="BN", domain="crystal"
        )

    with pytest.raises(AmbiguousIdentificationError) as exc_info:
        scoped_identify("boron nitride", ambiguous, work_dir=parent_dir)
    parent_id = exc_info.value.trace.run_id  # type: ignore[union-attr]
    parent_text = (parent_dir / "adam.json").read_text(encoding="utf-8")

    scoped_identify(
        "h-BN",
        _select_fn,
        parent_run_id=parent_id,
        work_dir=child_dir,
        write_artifacts=False,
    )
    assert (parent_dir / "adam.json").read_text(encoding="utf-8") == parent_text
    child = TraceStore.load(child_dir / "adam.json")
    assert child.parent_run_id == parent_id
    assert child.run_id != parent_id


def test_identifier_signature_has_required_trace_only() -> None:
    sig = inspect.signature(MaterialIdentifier.identify)
    assert "return_trace" not in sig.parameters
    assert sig.parameters["trace"].kind is inspect.Parameter.KEYWORD_ONLY


def test_llm_complete_records_success_and_failure() -> None:
    class FakeLLM(BaseLLM):
        provider = "google"
        requested_model = "test-model"

        async def _complete_once(self, messages: list[Message], **_kwargs: object) -> LLMResponse:
            return LLMResponse(content='{"ok": true}', model="test-model-actual")

    class BoomLLM(BaseLLM):
        provider = "openai"
        requested_model = "gpt-test"

        async def _complete_once(self, messages: list[Message], **_kwargs: object) -> LLMResponse:
            raise RuntimeError("provider down")

    trace = Trace(query="silicon")
    token = set_trace(trace)
    try:
        with llm_purpose("identification.extraction"):
            resp = asyncio.run(
                FakeLLM().complete(
                    [Message(role="user", content="hello")],
                    response_format="json",
                )
            )
        assert resp.content == '{"ok": true}'
        assert len(trace.llm_calls) == 1
        call = trace.llm_calls[0]
        assert call.purpose == "identification.extraction"
        assert call.provider == "google"
        assert call.requested_model == "test-model"
        assert call.provider_model == "test-model-actual"
        assert call.response == '{"ok": true}'
        assert call.error is None
        assert call.prompt_tokens is None
        assert call.messages[0].content == "hello"

        with llm_purpose("identification.narrow_selection"):
            with pytest.raises(RuntimeError, match="provider down"):
                asyncio.run(BoomLLM().complete([Message(role="user", content="pick")]))
        assert len(trace.llm_calls) == 2
        failed = trace.llm_calls[1]
        assert failed.purpose == "identification.narrow_selection"
        assert failed.provider == "openai"
        assert failed.error == "provider down"
        assert failed.exc_type == "RuntimeError"
        assert failed.response is None
    finally:
        from adam_identification.provenance.context import _current_trace

        _current_trace.reset(token)
