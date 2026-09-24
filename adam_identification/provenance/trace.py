"""FAIR provenance record for one identification run.

Serialized as ``adam.json`` by :class:`~adam_identification.provenance.store.TraceStore`.
DFT job fields (intent, folders, attempts, software probes, resources) are omitted.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from adam_identification.models import CrystalStructure, Material


class TraceStatus(StrEnum):
    """Lifecycle status of one identification run."""

    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class IdentificationStage(StrEnum):
    """Pipeline region recorded on :class:`FailureRecord` and ``current_stage``."""

    INITIALIZATION = "initialization"
    LLM = "llm"
    DATABASE_SEARCH = "database_search"
    SELECTION = "selection"
    OUTPUT_CONVERSION = "output_conversion"
    PERSISTENCE = "persistence"


class MaterialIdentity(BaseModel):
    """Compact identity of the identified material."""

    name: str | None = None
    chemical_formula: str | None = None
    source: str | None = None
    mp_id: str | None = None
    mc3d_id: str | None = None
    pc_cid: str | None = None
    material_type: Literal["crystal", "molecule"] | None = None
    space_group: str | None = None
    crystal_system: str | None = None


class LLMMessage(BaseModel):
    """One chat message sent to the provider."""

    role: str
    content: str


class LLMCallRecord(BaseModel):
    """Provenance for one LLM API call."""

    call_id: str = Field(default_factory=lambda: str(uuid4()))
    purpose: str = "unknown"
    provider: str = ""
    requested_model: str = ""
    provider_model: str | None = None
    requested_at: datetime
    completed_at: datetime | None = None
    messages: list[LLMMessage] = Field(default_factory=list)
    response: str | None = None
    error: str | None = None
    exc_type: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    response_format: str | None = None
    rate_limit_retries: int = 0


class EnvironmentRecord(BaseModel):
    """Package and interpreter versions for this run."""

    adam_identification_version: str
    python_version: str


class FailureRecord(BaseModel):
    """Observed failure during an identification run."""

    failure_id: str = Field(default_factory=lambda: str(uuid4()))
    at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stage: IdentificationStage
    exc_type: str
    message: str


class SelectedMaterialRecord(BaseModel):
    """Provider-neutral identity of the material identification selected.

    Older traces omit the object; loading them leaves the field ``None``.

    Attributes:
        provider: Database that supplied the material.
        provider_id: That database's identifier.
        method: Methodology such as ``"pbesol-v2"``. ``None`` when unreported.
            The Materials Project placeholder ``"unknown"`` is stored as ``None``.
    """

    provider: str
    provider_id: str
    method: str | None = None


def _selected_method(material: Material) -> str | None:
    """Return the methodology string to store, or ``None`` when unresolved."""
    props = material.get_properties(material.source)
    method = getattr(props, "method", None)
    if not isinstance(method, str):
        return None
    stripped = method.strip()
    if not stripped or stripped == "unknown":
        return None
    return stripped


def record_selected_material(section: IdentificationSection, material: Material) -> None:
    """Record the selected material on an identification section.

    Sets ``selected_id`` from :attr:`~adam_identification.models.Material.primary_id`.
    When that id is present, also sets ``selected_record``.
    """
    provider_id = material.primary_id
    section.selected_id = provider_id
    if provider_id is None:
        section.selected_record = None
        return
    section.selected_record = SelectedMaterialRecord(
        provider=material.source.value,
        provider_id=provider_id,
        method=_selected_method(material),
    )


class ArtifactRecord(BaseModel):
    """A structure file written for this run."""

    path: str
    kind: Literal["structure"] = "structure"
    format: str | None = None


class IdentificationSection(BaseModel):
    """Decision trail for crystals and molecules (shared field names).

    Narrow is the first candidate search and selection. Wide is the expanded
    search when the narrow decision was ``not_found`` (crystals today; molecules
    leave wide fields unset). Candidate dict keys may differ by database.
    """

    domain: Literal["crystal", "molecule", "unknown"] = "unknown"
    extraction: dict[str, Any] = Field(default_factory=dict)
    n_candidates_narrow: int | None = None
    candidates_shown_narrow: list[dict[str, Any]] = Field(default_factory=list)
    n_candidates_wide: int | None = None
    candidates_shown_wide: list[dict[str, Any]] = Field(default_factory=list)
    narrow_selection: dict[str, Any] | None = None
    wide_selection: dict[str, Any] | None = None
    selection_decision: str | None = None
    selection_reason: str | None = None
    clarification_message: str | None = None
    suggested_candidates: list[dict[str, Any]] | None = None
    outcome: Literal["selected", "not_found", "ambiguous", "clarify"] | None = None
    selected_id: str | None = None
    selected_record: SelectedMaterialRecord | None = None
    needs_review: bool = False


def discover_environment() -> EnvironmentRecord:
    """Return package and Python versions for a new :class:`Trace`."""
    version = "2.1.0"
    try:
        from importlib.metadata import version as pkg_version

        version = pkg_version("adam-identification")
    except Exception:
        from adam_identification import __version__ as fallback

        version = fallback
    return EnvironmentRecord(
        adam_identification_version=version,
        python_version=sys.version,
    )


class Trace(BaseModel):
    """Full provenance for one identification call."""

    schema_version: Literal["1"] = "1"
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    parent_run_id: str | None = None
    status: TraceStatus = TraceStatus.RUNNING
    query: str
    material: MaterialIdentity | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    environment: EnvironmentRecord = Field(default_factory=discover_environment)
    llm_calls: list[LLMCallRecord] = Field(default_factory=list)
    identification: IdentificationSection = Field(default_factory=IdentificationSection)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    failures: list[FailureRecord] = Field(default_factory=list)

    @property
    def needs_review(self) -> bool:
        """True when identification was resolved under uncertainty."""
        return self.identification.needs_review

    def record_llm_call(self, call: LLMCallRecord) -> None:
        """Append one :class:`LLMCallRecord`."""
        self.llm_calls.append(call)

    def record_failure(self, exc: BaseException, *, stage: IdentificationStage) -> None:
        """Append a :class:`FailureRecord` for an observed exception."""
        self.failures.append(
            FailureRecord(
                stage=stage,
                exc_type=type(exc).__name__,
                message=str(exc),
            )
        )

    def record_artifact(self, path: str, *, fmt: str | None = None) -> None:
        """Record a written structure file."""
        self.artifacts.append(ArtifactRecord(path=path, format=fmt))

    def set_material(self, material: Material) -> None:
        """Fill :attr:`material` from a resolved :class:`~adam_identification.models.Material`."""
        space_group: str | None = None
        crystal_system: str | None = None
        if isinstance(material.structure, CrystalStructure):
            space_group = material.structure.space_group
            crystal_system = material.structure.crystal_system
        source = material.source
        source_str = source.value if hasattr(source, "value") else str(source)
        self.material = MaterialIdentity(
            name=material.name,
            chemical_formula=material.chemical_formula,
            source=source_str,
            mp_id=material.mp_id,
            mc3d_id=material.mc3d_id,
            pc_cid=str(material.pc_cid) if material.pc_cid is not None else None,
            material_type=material.material_type,
            space_group=space_group,
            crystal_system=crystal_system,
        )
