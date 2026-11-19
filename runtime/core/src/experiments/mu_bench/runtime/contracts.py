"""Versioned contracts shared by every μ-Bench execution backend."""

from __future__ import annotations

try:
    from enum import StrEnum
except ImportError:  # Python 3.10 in the current H online-inference image.
    from enum import Enum

    class StrEnum(str, Enum):
        """Minimal stdlib StrEnum compatibility for Python 3.10."""

        def __str__(self) -> str:
            return str(self.value)
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


RUNTIME_PROTOCOL_VERSION = "mu-bench-runtime-v1.3-hybrid-control"
SESSION_SCHEMA_VERSION = "mu-bench-session-spec-v1"
RESULT_SCHEMA_VERSION = "mu-bench-session-result-v1"


class SessionPhase(StrEnum):
    """One immutable model session in Probe/Freeze/Final execution."""

    PROBE = "probe"
    FINAL = "final"
    SMOKE = "smoke"


class SessionState(StrEnum):
    """Infrastructure state; only SUCCEEDED is a valid model result."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    MODEL_ERROR = "model_error"
    INFRA_ERROR = "infra_error"
    UNSUPPORTED = "unsupported"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self not in {SessionState.QUEUED, SessionState.RUNNING}


class ExecutionMode(StrEnum):
    """How the adapter exposes the model's input frontier to the runner."""

    CAUSAL_STREAMING = "causal_streaming"
    WHOLE_TRACK_COMPAT = "whole_track_compat"


class CausalInputIsolation(StrEnum):
    """Strength of the guarantee that future input is hidden from the model."""

    ENFORCED = "enforced"
    AUDITED = "audited"
    NOT_ENFORCED = "not_enforced"


class MediaClockSpec(BaseModel):
    """Authoritative media-time configuration inside one SUT session."""

    tick_ms: int = Field(default=200, gt=0)
    input_sample_rate: int = Field(default=16_000, gt=0)
    input_channels: Literal[1] = 1
    sample_width_bytes: Literal[2] = 2
    pacing: Literal["realtime", "unpaced"] = "realtime"
    task_start_ms: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_integral_tick(self) -> MediaClockSpec:
        samples = self.input_sample_rate * self.tick_ms
        if samples % 1000:
            raise ValueError("media tick must contain an integral number of samples")
        return self

    @property
    def samples_per_tick(self) -> int:
        return self.input_sample_rate * self.tick_ms // 1000

    @property
    def bytes_per_tick(self) -> int:
        return self.samples_per_tick * self.sample_width_bytes


class ArtifactManifest(BaseModel):
    """Absolute GPFS paths used by one immutable session attempt."""

    input_wav: str
    output_dir: str
    result_json: str
    system_prompt_path: str | None = None
    frozen_event_plan_path: str | None = None
    voice_manifest_path: str | None = None
    trace_json: str | None = None

    @model_validator(mode="after")
    def require_absolute_paths(self) -> ArtifactManifest:
        for name, value in self.model_dump().items():
            if value is not None and not Path(value).is_absolute():
                raise ValueError(f"artifact path {name} must be absolute: {value}")
        return self


class BackendCapabilities(BaseModel):
    """Declared adapter surface used to select fair leaderboard tracks."""

    backend_name: str
    model_name: str
    incremental_audio_input: bool = True
    whole_track_audio_input: bool = False
    incremental_audio_output: bool = True
    dynamic_text_control_items: bool = False
    native_tool_calling: bool = False
    explicit_speaker_attribution: bool = False
    explicit_addressee_attribution: bool = False
    media_tick_ms: int = Field(default=200, gt=0)
    native_input_quantum_ms: int = Field(default=200, gt=0)
    effective_input_quantum_ms: int | None = Field(default=None, gt=0)
    execution_mode: ExecutionMode = ExecutionMode.CAUSAL_STREAMING
    causal_input_isolation: CausalInputIsolation = CausalInputIsolation.ENFORCED
    committed_output_timing: bool = True
    mid_session_cancel: bool = True
    supports_wall_clock_timing: bool = True
    notes: list[str] = Field(default_factory=list)


class SessionSpec(BaseModel):
    """Complete session-level input placed on GPFS by the orchestrator."""

    schema_version: Literal[SESSION_SCHEMA_VERSION] = SESSION_SCHEMA_VERSION
    runtime_protocol_version: Literal[RUNTIME_PROTOCOL_VERSION] = (
        RUNTIME_PROTOCOL_VERSION
    )
    session_id: str = Field(min_length=1)
    attempt_id: int = Field(default=1, ge=1)
    case_id: str = Field(min_length=1)
    phase: SessionPhase
    probe_index: int | None = Field(default=None, ge=0)
    model_name: str = Field(min_length=1)
    backend_name: str = Field(min_length=1)
    execution_mode: ExecutionMode = ExecutionMode.CAUSAL_STREAMING
    seed: int = 42
    media_clock: MediaClockSpec = Field(default_factory=MediaClockSpec)
    artifacts: ArtifactManifest
    frozen_event_ids: list[str] = Field(default_factory=list)
    frozen_plan_sha256: str | None = None
    required_capabilities: dict[str, bool] = Field(default_factory=dict)
    control_context: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_phase(self) -> SessionSpec:
        if self.phase == SessionPhase.PROBE and self.probe_index is None:
            raise ValueError("probe session requires probe_index")
        if self.phase != SessionPhase.PROBE and self.probe_index is not None:
            raise ValueError("probe_index is only valid for probe sessions")
        return self


class SessionSubmission(BaseModel):
    """Small HTTP control message pointing at a GPFS SessionSpec."""

    spec_path: str
    result_path: str
    idempotency_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_paths(self) -> SessionSubmission:
        for name in ("spec_path", "result_path"):
            value = getattr(self, name)
            if not Path(value).is_absolute():
                raise ValueError(f"{name} must be an absolute GPFS path")
        return self


class AssistantFragment(BaseModel):
    timestamp_ms: int = Field(ge=0)
    text: str


class AssistantInterval(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_interval(self) -> AssistantInterval:
        if self.end_ms < self.start_ms:
            raise ValueError("assistant interval ends before it starts")
        return self


class ToolObservation(BaseModel):
    timestamp_ms: int = Field(ge=0)
    kind: Literal["tool_call", "tool_result"]
    tool_name: str
    payload: dict[str, Any] = Field(default_factory=dict)


class SessionError(BaseModel):
    error_type: str
    message: str
    traceback: str | None = None
    retryable: bool = False


class SessionResult(BaseModel):
    """One terminal or in-progress result with separate media/wall clocks."""

    schema_version: Literal[RESULT_SCHEMA_VERSION] = RESULT_SCHEMA_VERSION
    runtime_protocol_version: Literal[RUNTIME_PROTOCOL_VERSION] = (
        RUNTIME_PROTOCOL_VERSION
    )
    session_id: str
    attempt_id: int = Field(ge=1)
    case_id: str
    phase: SessionPhase
    state: SessionState
    model_name: str
    backend_name: str
    execution_mode: ExecutionMode = ExecutionMode.CAUSAL_STREAMING
    causal_input_isolation: CausalInputIsolation = CausalInputIsolation.ENFORCED
    queued_at_unix: float | None = None
    started_at_unix: float | None = None
    completed_at_unix: float | None = None
    media_duration_ms: int = Field(default=0, ge=0)
    inference_wall_ms: float = Field(default=0.0, ge=0)
    input_wav: str | None = None
    assistant_wav: str | None = None
    assistant_text: str = ""
    assistant_fragments: list[AssistantFragment] = Field(default_factory=list)
    assistant_intervals: list[AssistantInterval] = Field(default_factory=list)
    tool_observations: list[ToolObservation] = Field(default_factory=list)
    capabilities: BackendCapabilities | None = None
    runtime_metrics: dict[str, Any] = Field(default_factory=dict)
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    error: SessionError | None = None

    @model_validator(mode="after")
    def validate_terminal_result(self) -> SessionResult:
        if self.state == SessionState.SUCCEEDED:
            if self.error is not None:
                raise ValueError("successful result cannot contain an error")
            if not self.assistant_wav:
                raise ValueError("successful result requires assistant_wav")
        if self.state in {
            SessionState.MODEL_ERROR,
            SessionState.INFRA_ERROR,
            SessionState.UNSUPPORTED,
        } and self.error is None:
            raise ValueError(f"{self.state} result requires error details")
        return self


class WorkerJobStatus(BaseModel):
    """HTTP-visible job status; result content remains on GPFS."""

    session_id: str
    state: SessionState
    spec_path: str
    result_path: str
    idempotency_key: str
    queued_at_unix: float
    started_at_unix: float | None = None
    completed_at_unix: float | None = None
    error: SessionError | None = None


class RJobWorkerEndpoint(BaseModel):
    """GPFS-persisted discovery record for one directly addressable RJob."""

    schema_version: Literal["mu-bench-rjob-endpoint-v1"] = (
        "mu-bench-rjob-endpoint-v1"
    )
    transport: Literal["rjob_http"] = "rjob_http"
    ready: bool
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65_535)
    base_url: str = Field(min_length=1)
    hostname: str = Field(min_length=1)
    pid: int = Field(gt=0)
    backend_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    api_key_file: str | None = None
    started_at_unix: float
    updated_at_unix: float
    rjob_name: str | None = None
    pod_name: str | None = None

    @model_validator(mode="after")
    def validate_endpoint(self) -> RJobWorkerEndpoint:
        expected = f"http://{self.host}:{self.port}"
        if self.base_url.rstrip("/") != expected:
            raise ValueError("RJob base_url disagrees with host and port")
        if self.api_key_file is not None and not Path(self.api_key_file).is_absolute():
            raise ValueError("RJob api_key_file must be an absolute GPFS path")
        return self
