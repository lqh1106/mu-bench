"""Whole-track compatibility wrapper for pre-v1.3 open-weight adapters."""

from __future__ import annotations

from pathlib import Path
import threading
import time
import traceback
from typing import Any, Protocol

from experiments.mu_bench.runtime.contracts import (
    AssistantFragment,
    AssistantInterval,
    BackendCapabilities,
    CausalInputIsolation,
    ExecutionMode,
    SessionError,
    SessionResult,
    SessionSpec,
    SessionState,
    ToolObservation,
)
from experiments.mu_bench.runtime.storage import GPFSArtifactStore


class LegacyTrackBackend(Protocol):
    """Existing adapter that internally consumes a complete local WAV."""

    def run_session(
        self,
        input_wav: Path,
        output_dir: Path,
        system_prompt: str | None,
        seed: int,
    ) -> dict[str, Any]:
        """Return legacy transcript/timing fields after local inference."""


class LegacyWholeTrackExecutor:
    """Bridge existing model code into SessionSpec without frame-level GPFS.

    This compatibility executor is not eligible for uniform wall-clock timing
    unless the wrapped model itself paces input locally.  New adapters should
    implement ModelMediaAdapter and use LocalMediaClockExecutor.
    """

    def __init__(
        self,
        *,
        store: GPFSArtifactStore,
        backend: LegacyTrackBackend,
        capabilities: BackendCapabilities,
    ) -> None:
        if capabilities.execution_mode != ExecutionMode.WHOLE_TRACK_COMPAT:
            raise ValueError(
                "LegacyWholeTrackExecutor requires execution_mode=whole_track_compat"
            )
        if not capabilities.whole_track_audio_input:
            raise ValueError(
                "LegacyWholeTrackExecutor requires whole_track_audio_input=true"
            )
        if capabilities.causal_input_isolation == CausalInputIsolation.ENFORCED:
            raise ValueError(
                "LegacyWholeTrackExecutor cannot claim enforced causal input isolation"
            )
        self.store = store
        self.backend = backend
        self._capabilities = capabilities

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def execute(
        self,
        spec: SessionSpec,
        *,
        cancel_event: threading.Event | None = None,
    ) -> SessionResult:
        started = time.time()
        wall_zero = time.monotonic()
        capabilities = self.capabilities
        if spec.execution_mode != capabilities.execution_mode:
            return SessionResult(
                session_id=spec.session_id,
                attempt_id=spec.attempt_id,
                case_id=spec.case_id,
                phase=spec.phase,
                state=SessionState.UNSUPPORTED,
                model_name=spec.model_name,
                backend_name=spec.backend_name,
                execution_mode=capabilities.execution_mode,
                causal_input_isolation=capabilities.causal_input_isolation,
                started_at_unix=started,
                completed_at_unix=time.time(),
                capabilities=capabilities,
                error=SessionError(
                    error_type="execution_mode_mismatch",
                    message=(
                        f"SessionSpec requests {spec.execution_mode.value}, "
                        f"legacy adapter exposes {capabilities.execution_mode.value}"
                    ),
                ),
            )
        if cancel_event is not None and cancel_event.is_set():
            return SessionResult(
                session_id=spec.session_id,
                attempt_id=spec.attempt_id,
                case_id=spec.case_id,
                phase=spec.phase,
                state=SessionState.CANCELLED,
                model_name=spec.model_name,
                backend_name=spec.backend_name,
                execution_mode=capabilities.execution_mode,
                causal_input_isolation=capabilities.causal_input_isolation,
                started_at_unix=started,
                completed_at_unix=time.time(),
                capabilities=capabilities,
                runtime_metrics={
                    "execution_mode": capabilities.execution_mode.value,
                    "causal_input_isolation": capabilities.causal_input_isolation.value,
                    "input_delivery": "complete_wav_path",
                    "wall_timing_eligible": False,
                },
            )
        try:
            input_wav = self.store.resolve(spec.artifacts.input_wav)
            output_dir = self.store.ensure_output_dir(spec.artifacts.output_dir)
            prompt = None
            if spec.artifacts.system_prompt_path:
                prompt = self.store.resolve(
                    spec.artifacts.system_prompt_path
                ).read_text(encoding="utf-8")
            payload = self.backend.run_session(
                input_wav,
                output_dir,
                prompt,
                spec.seed,
            )
            assistant_wav = self.store.resolve(payload["assistant_wav"])
            fragments = [
                AssistantFragment.model_validate(item)
                for item in payload.get("assistant_fragments", [])
            ]
            intervals = [
                AssistantInterval.model_validate(item)
                for item in payload.get("assistant_intervals", [])
            ]
            observations = [
                ToolObservation.model_validate(item)
                for item in payload.get("tool_observations", [])
            ]
            media_duration_ms = int(payload.get("media_duration_ms", 0))
            return SessionResult(
                session_id=spec.session_id,
                attempt_id=spec.attempt_id,
                case_id=spec.case_id,
                phase=spec.phase,
                state=SessionState.SUCCEEDED,
                model_name=spec.model_name,
                backend_name=spec.backend_name,
                execution_mode=capabilities.execution_mode,
                causal_input_isolation=capabilities.causal_input_isolation,
                started_at_unix=started,
                completed_at_unix=time.time(),
                media_duration_ms=media_duration_ms,
                inference_wall_ms=(time.monotonic() - wall_zero) * 1000,
                input_wav=str(input_wav),
                assistant_wav=str(assistant_wav),
                assistant_text=str(payload.get("assistant_text", "")),
                assistant_fragments=fragments,
                assistant_intervals=intervals,
                tool_observations=observations,
                capabilities=self.capabilities,
                runtime_metrics={
                    "frame_transport": "gpu_process_local_legacy_track",
                    "input_delivery": "complete_wav_path",
                    "execution_mode": capabilities.execution_mode.value,
                    "causal_input_isolation": capabilities.causal_input_isolation.value,
                    "future_audio_access": "possible",
                    "tick_ms": spec.media_clock.tick_ms,
                    "native_input_quantum_ms": capabilities.native_input_quantum_ms,
                    "effective_input_quantum_ms": capabilities.effective_input_quantum_ms
                    or capabilities.native_input_quantum_ms,
                    "committed_output_timing": capabilities.committed_output_timing,
                    "mid_session_cancel": capabilities.mid_session_cancel,
                    "wall_timing_eligible": False,
                },
                model_metadata={
                    key: value
                    for key, value in payload.items()
                    if key
                    not in {
                        "assistant_wav",
                        "assistant_text",
                        "assistant_fragments",
                        "assistant_intervals",
                        "tool_observations",
                    }
                },
            )
        except Exception as exc:  # noqa: BLE001
            return SessionResult(
                session_id=spec.session_id,
                attempt_id=spec.attempt_id,
                case_id=spec.case_id,
                phase=spec.phase,
                state=SessionState.MODEL_ERROR,
                model_name=spec.model_name,
                backend_name=spec.backend_name,
                execution_mode=capabilities.execution_mode,
                causal_input_isolation=capabilities.causal_input_isolation,
                started_at_unix=started,
                completed_at_unix=time.time(),
                inference_wall_ms=(time.monotonic() - wall_zero) * 1000,
                input_wav=spec.artifacts.input_wav,
                capabilities=self.capabilities,
                error=SessionError(
                    error_type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    retryable=False,
                ),
            )
