"""Shared identification run: Trace lifecycle, artifacts, and provider construction."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from adam_identification.exceptions import (
    AmbiguousIdentificationError,
    ClarificationNeededError,
    MaterialNotFoundError,
)
from adam_identification.models import Material
from adam_identification.provenance.context import (
    checkpoint,
    current_stage,
    set_stage,
    set_store,
    set_trace,
)
from adam_identification.provenance.store import TraceStore
from adam_identification.provenance.trace import (
    IdentificationStage,
    Trace,
    TraceStatus,
)

_PROTOCOL_ERRORS = (AmbiguousIdentificationError, ClarificationNeededError, MaterialNotFoundError)


def _bind(trace: Trace, store: TraceStore | None) -> tuple[Any, Any, Any]:
    """Bind ContextVars and return reset tokens."""
    t = set_trace(trace)
    s = set_store(store)
    st = set_stage(IdentificationStage.INITIALIZATION)
    return t, s, st


def _unbind(tokens: tuple[Any, Any, Any]) -> None:
    """Reset provenance ContextVars."""
    from adam_identification.provenance.context import (
        _current_stage,
        _current_store,
        _current_trace,
    )

    t, s, st = tokens
    _current_trace.reset(t)
    _current_store.reset(s)
    _current_stage.reset(st)


def _attach(exc: BaseException, trace: Trace) -> None:
    """Attach *trace* onto *exc* when the exception type allows it."""
    try:
        exc.trace = trace  # type: ignore[attr-defined]
    except Exception:
        pass


def _finish_failed(trace: Trace, exc: BaseException) -> None:
    """Record an observed failure and mark the trace FAILED."""
    if not trace.failures or trace.failures[-1].exc_type != type(exc).__name__:
        trace.record_failure(exc, stage=current_stage())
    trace.status = TraceStatus.FAILED
    trace.finished_at = datetime.now(UTC)
    _save_terminal(trace)


def _finish_protocol(trace: Trace) -> None:
    """Mark a protocol terminal outcome COMPLETED (no FailureRecord)."""
    trace.status = TraceStatus.COMPLETED
    trace.finished_at = datetime.now(UTC)
    _save_terminal(trace)


def _save_terminal(trace: Trace) -> None:
    """Checkpoint terminal state; persistence errors become a FailureRecord."""
    from adam_identification.provenance.context import current_store

    store = current_store()
    if store is None:
        return
    try:
        store.save(trace)
    except Exception as persist_exc:
        trace.record_failure(persist_exc, stage=IdentificationStage.PERSISTENCE)
        trace.status = TraceStatus.FAILED
        try:
            store.save(trace)
        except Exception:
            pass


def write_structure_artifacts(
    material: Material,
    *,
    work_dir: Path | None,
    extra_paths: list[Path] | None = None,
) -> None:
    """Write structure files, record them on the active trace, then checkpoint.

    Raises:
        Exception: Conversion or write failure (caller marks FAILED).
    """
    from adam_identification.output import to_ase
    from adam_identification.provenance.context import current_trace

    set_stage(IdentificationStage.OUTPUT_CONVERSION)
    dests: list[Path] = []
    if extra_paths:
        dests.extend(extra_paths)
        if work_dir is not None:
            work_dir.mkdir(parents=True, exist_ok=True)
            for path in extra_paths:
                dests.append(work_dir / path.name)
    elif work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
        formula = material.chemical_formula or "structure"
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in formula)
        ext = ".cif" if material.material_type == "crystal" else ".xyz"
        dests.append(work_dir / f"{safe}{ext}")

    if not dests:
        return

    import ase.io as ase_io

    atoms = to_ase(material)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in dests:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)

    trace = current_trace()
    for path in unique:
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()
        if ext == ".cif":
            fmt = "cif"
        elif ext in (".xyz", ".extxyz"):
            fmt = "extxyz"
        elif path.name in ("POSCAR", "CONTCAR") or ext == ".vasp":
            fmt = "vasp"
        else:
            fmt = None
        ase_io.write(str(path), atoms, format=fmt)
        if trace is not None:
            trace.record_artifact(str(path.resolve()), fmt=fmt or ext.lstrip(".") or None)
            checkpoint()


def _write_success_artifacts(
    material: Material,
    *,
    work_dir: Path | None,
    extra_save_paths: list[Path] | None,
    save_dir: Path | None,
    save_stem: str | None,
    write_artifacts: bool,
) -> None:
    """Write structure artifacts when the caller requested persistence."""
    paths = list(extra_save_paths or [])
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = save_stem or "structure"
        ext = ".cif" if material.material_type == "crystal" else ".xyz"
        paths.append(Path(save_dir) / f"{stem}{ext}")
    if write_artifacts and (work_dir is not None or paths):
        write_structure_artifacts(material, work_dir=work_dir, extra_paths=paths or None)


def scoped_identify(
    query: str,
    identify_fn: Callable[[str, Trace], Material],
    *,
    parent_run_id: str | None = None,
    store: TraceStore | None = None,
    work_dir: Path | None = None,
    extra_save_paths: list[Path] | None = None,
    write_artifacts: bool = False,
    save_dir: Path | None = None,
    save_stem: str | None = None,
) -> tuple[Material, Trace]:
    """Run ``identify_fn`` under a new Trace. Does not construct an LLM provider.

    Used by :class:`~adam_identification.session.IdentificationSession`.
    """
    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        if store is None:
            store = TraceStore(work_dir)
    trace = Trace(query=query, parent_run_id=parent_run_id)
    tokens = _bind(trace, store)
    try:
        if store is not None:
            store.save(trace)
        try:
            material = identify_fn(query, trace)
        except _PROTOCOL_ERRORS as exc:
            _attach(exc, trace)
            _finish_protocol(trace)
            raise
        except BaseException as exc:
            _attach(exc, trace)
            _finish_failed(trace, exc)
            raise
        trace.set_material(material)
        checkpoint()
        try:
            _write_success_artifacts(
                material,
                work_dir=work_dir,
                extra_save_paths=extra_save_paths,
                save_dir=save_dir,
                save_stem=save_stem,
                write_artifacts=write_artifacts,
            )
        except BaseException as exc:
            _finish_failed(trace, exc)
            raise
        trace.status = TraceStatus.COMPLETED
        trace.finished_at = datetime.now(UTC)
        _save_terminal(trace)
        return material, trace
    except BaseException as exc:
        if trace.status == TraceStatus.RUNNING:
            if isinstance(exc, _PROTOCOL_ERRORS):
                _finish_protocol(trace)
            else:
                _finish_failed(trace, exc)
        raise
    finally:
        _unbind(tokens)


def run_identification(
    query: str,
    *,
    provider: str = "google",
    model: str | None = None,
    mp_api_key: str | None = None,
    minimal_interaction: bool = False,
    work_dir: Path | None = None,
    extra_save_paths: list[Path] | None = None,
    parent_run_id: str | None = None,
    identify_fn: Callable[[str, Trace], Material] | None = None,
    write_artifacts: bool = True,
    save_dir: Path | None = None,
    save_stem: str | None = None,
) -> tuple[Material, Trace]:
    """Create a Trace, identify ``query``, write artifacts, return ``(material, trace)``.

    Constructs the Trace **before** :func:`~adam_identification.llm.get_provider`.
    Persist when ``work_dir`` is set. When ``identify_fn`` is supplied, skip
    provider and database client construction.
    """
    from adam_identification.database.materials_project import MaterialsProjectClient
    from adam_identification.database.pubchem import PubChemClient
    from adam_identification.identifier import MaterialIdentifier
    from adam_identification.llm import get_provider

    if work_dir is not None:
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    trace = Trace(query=query, parent_run_id=parent_run_id)
    store = TraceStore(work_dir) if work_dir is not None else None
    tokens = _bind(trace, store)
    try:
        if store is not None:
            store.save(trace)
        if identify_fn is None:
            try:
                llm = get_provider(provider, model)
                mp_client = MaterialsProjectClient(api_key=mp_api_key)
                pubchem_client = PubChemClient()
            except BaseException as exc:
                _attach(exc, trace)
                _finish_failed(trace, exc)
                raise
            identifier = MaterialIdentifier(
                llm, mp_client, pubchem_client, minimal_interaction=minimal_interaction
            )
            identify_fn = lambda q, tr: identifier.identify(q, trace=tr)  # noqa: E731

        try:
            material = identify_fn(query, trace)
        except _PROTOCOL_ERRORS as exc:
            _attach(exc, trace)
            _finish_protocol(trace)
            raise
        except BaseException as exc:
            _attach(exc, trace)
            _finish_failed(trace, exc)
            raise

        trace.set_material(material)
        checkpoint()
        try:
            _write_success_artifacts(
                material,
                work_dir=work_dir,
                extra_save_paths=extra_save_paths,
                save_dir=save_dir,
                save_stem=save_stem,
                write_artifacts=write_artifacts,
            )
        except BaseException as exc:
            _finish_failed(trace, exc)
            raise
        trace.status = TraceStatus.COMPLETED
        trace.finished_at = datetime.now(UTC)
        _save_terminal(trace)
        return material, trace
    except BaseException as exc:
        if trace.status == TraceStatus.RUNNING:
            if isinstance(exc, _PROTOCOL_ERRORS):
                _finish_protocol(trace)
            else:
                _finish_failed(trace, exc)
        raise
    finally:
        _unbind(tokens)
