"""ContextVars for the active Trace, TraceStore, LLM purpose, and pipeline stage."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

from adam_identification.provenance.trace import IdentificationStage

if TYPE_CHECKING:
    from adam_identification.provenance.store import TraceStore
    from adam_identification.provenance.trace import Trace

_current_trace: ContextVar[Trace | None] = ContextVar("_current_trace", default=None)
_current_store: ContextVar[TraceStore | None] = ContextVar("_current_store", default=None)
_current_purpose: ContextVar[str] = ContextVar("_current_purpose", default="unknown")
_current_stage: ContextVar[IdentificationStage] = ContextVar(
    "_current_stage", default=IdentificationStage.INITIALIZATION
)


def set_trace(trace: Trace | None) -> Token[Trace | None]:
    """Bind *trace* as the active provenance record."""
    return _current_trace.set(trace)


def current_trace() -> Trace | None:
    """Return the active :class:`~adam_identification.provenance.trace.Trace`, or ``None``."""
    return _current_trace.get()


def set_store(store: TraceStore | None) -> Token[TraceStore | None]:
    """Bind *store* for incremental checkpoints."""
    return _current_store.set(store)


def current_store() -> TraceStore | None:
    """Return the active :class:`~adam_identification.provenance.store.TraceStore`, or ``None``."""
    return _current_store.get()


def set_purpose(purpose: str) -> Token[str]:
    """Set the purpose label for the next LLM call."""
    return _current_purpose.set(purpose)


def current_purpose() -> str:
    """Return the current LLM purpose label (default ``"unknown"``)."""
    return _current_purpose.get()


def set_stage(stage: IdentificationStage) -> Token[IdentificationStage]:
    """Set the pipeline stage for interrupt / failure records."""
    return _current_stage.set(stage)


def current_stage() -> IdentificationStage:
    """Return the current :class:`IdentificationStage`."""
    return _current_stage.get()


def checkpoint() -> None:
    """Save the active trace when a store is bound. No-op otherwise."""
    trace = current_trace()
    store = current_store()
    if trace is None or store is None:
        return
    try:
        store.save(trace)
    except Exception:
        pass


@contextmanager
def llm_purpose(purpose: str) -> Generator[None, None, None]:
    """Set the LLM purpose for the duration of the block."""
    token = _current_purpose.set(purpose)
    try:
        yield
    finally:
        _current_purpose.reset(token)


@contextmanager
def identification_stage(stage: IdentificationStage) -> Generator[None, None, None]:
    """Set :func:`current_stage` for the duration of the block.

    On a clean exit the previous stage is restored. On an exception the stage
    is left set so a ``FailureRecord`` records this region (e.g. interrupt
    during an LLM call).
    """
    token = _current_stage.set(stage)
    try:
        yield
    except BaseException:
        raise
    else:
        _current_stage.reset(token)
