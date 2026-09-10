"""FAIR provenance for adam-identification runs."""

from adam_identification.provenance.context import (
    checkpoint,
    current_stage,
    current_store,
    current_trace,
    identification_stage,
    llm_purpose,
    set_stage,
    set_store,
    set_trace,
)
from adam_identification.provenance.lifecycle import run_identification, scoped_identify
from adam_identification.provenance.store import TraceStore
from adam_identification.provenance.trace import (
    ArtifactRecord,
    EnvironmentRecord,
    FailureRecord,
    IdentificationSection,
    IdentificationStage,
    LLMCallRecord,
    LLMMessage,
    MaterialIdentity,
    Trace,
    TraceStatus,
    discover_environment,
)

__all__ = [
    "ArtifactRecord",
    "EnvironmentRecord",
    "FailureRecord",
    "IdentificationSection",
    "IdentificationStage",
    "LLMCallRecord",
    "LLMMessage",
    "MaterialIdentity",
    "Trace",
    "TraceStatus",
    "TraceStore",
    "checkpoint",
    "current_stage",
    "current_store",
    "current_trace",
    "discover_environment",
    "identification_stage",
    "llm_purpose",
    "run_identification",
    "scoped_identify",
    "set_stage",
    "set_store",
    "set_trace",
]
