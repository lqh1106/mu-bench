"""GPU-local 200 ms media clock; frames never cross HTTP or GPFS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any, Protocol
import wave

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


@dataclass(frozen=True)
class AudioFrame:
    """One local PCM16 mono input frame on the authoritative media clock."""

    index: int
    start_ms: int
    end_ms: int
    pcm16: bytes
    final: bool = False


@dataclass
class AdapterSessionOutput:
    """Model-native output after all local frame callbacks have drained."""

    assistant_pcm16: bytes
    assistant_sample_rate: int
    assistant_fragments: list[AssistantFragment] = field(default_factory=list)
    assistant_intervals: list[AssistantInterval] = field(default_factory=list)
    tool_observations: list[ToolObservation] = field(default_factory=list)
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelMediaAdapter(Protocol):
    """A loaded model receiving media frames inside the GPU process."""

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return immutable backend capabilities."""

    def start(self, spec: SessionSpec, system_prompt: str | None) -> None:
        """Start one fresh model session."""

    def push_frame(self, frame: AudioFrame) -> None:
        """Consume one local benchmark tick; native buffering is allowed."""

    def finish(self, media_end_ms: int) -> AdapterSessionOutput:
        """Stop input, drain model playback, and return committed output."""

    def cancel(self) -> None:
        """Best-effort cancellation for the active session."""


def _read_pcm16_mono(path: Path, spec: SessionSpec) -> bytes:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != spec.media_clock.input_channels:
            raise ValueError("input WAV channel count does not match SessionSpec")
        if handle.getsampwidth() != spec.media_clock.sample_width_bytes:
            raise ValueError("input WAV sample width does not match SessionSpec")
        if handle.getframerate() != spec.media_clock.input_sample_rate:
            raise ValueError("input WAV sample rate does not match SessionSpec")
        return handle.readframes(handle.getnframes())


def _write_pcm16_mono(path: Path, pcm: bytes, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


class LocalMediaClockExecutor:
    """Run an entire Probe/Final locally while measuring media and wall clocks."""

    def __init__(
        self,
        *,
        store: GPFSArtifactStore,
        adapter: ModelMediaAdapter,
    ) -> None:
        self.store = store
        self.adapter = adapter
        self._active_lock = threading.Lock()

    @property
    def capabilities(self) -> BackendCapabilities:
        return self.adapter.capabilities

    def execute(
        self,
        spec: SessionSpec,
        *,
        cancel_event: threading.Event | None = None,
    ) -> SessionResult:
        if not self._active_lock.acquire(blocking=False):
            raise RuntimeError("GPU worker already has an active model session")
        started_unix = time.time()
        wall_zero = time.monotonic()
        tick_records: list[dict[str, Any]] = []
        try:
            capabilities = self.adapter.capabilities
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
                    started_at_unix=started_unix,
                    completed_at_unix=time.time(),
                    capabilities=capabilities,
                    error=SessionError(
                        error_type="execution_mode_mismatch",
                        message=(
                            f"SessionSpec requests {spec.execution_mode.value}, "
                            f"adapter exposes {capabilities.execution_mode.value}"
                        ),
                    ),
                )
            if (
                capabilities.execution_mode != ExecutionMode.CAUSAL_STREAMING
                or capabilities.causal_input_isolation
                != CausalInputIsolation.ENFORCED
            ):
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
                    started_at_unix=started_unix,
                    completed_at_unix=time.time(),
                    capabilities=capabilities,
                    error=SessionError(
                        error_type="causal_streaming_required",
                        message=(
                            "LocalMediaClockExecutor requires an adapter that "
                            "enforces causal streaming input isolation"
                        ),
                    ),
                )
            input_path = self.store.resolve(spec.artifacts.input_wav)
            output_dir = self.store.ensure_output_dir(spec.artifacts.output_dir)
            prompt = None
            if spec.artifacts.system_prompt_path is not None:
                prompt_path = self.store.resolve(spec.artifacts.system_prompt_path)
                prompt = prompt_path.read_text(encoding="utf-8")
            pcm = _read_pcm16_mono(input_path, spec)
            self.adapter.start(spec, prompt)
            chunk_bytes = spec.media_clock.bytes_per_tick
            media_end_ms = 0
            stalled_ticks = 0
            for index, offset in enumerate(range(0, len(pcm), chunk_bytes)):
                if cancel_event is not None and cancel_event.is_set():
                    self.adapter.cancel()
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
                        started_at_unix=started_unix,
                        completed_at_unix=time.time(),
                        media_duration_ms=media_end_ms,
                        inference_wall_ms=(time.monotonic() - wall_zero) * 1000,
                        input_wav=str(input_path),
                        capabilities=capabilities,
                        runtime_metrics={
                            "tick_ms": spec.media_clock.tick_ms,
                            "ticks_completed": index,
                            "stalled_tick_count": stalled_ticks,
                            "execution_mode": capabilities.execution_mode.value,
                            "causal_input_isolation": capabilities.causal_input_isolation.value,
                        },
                    )
                frame_pcm = pcm[offset : offset + chunk_bytes]
                duration_ms = round(
                    len(frame_pcm)
                    * 1000
                    / (
                        spec.media_clock.input_sample_rate
                        * spec.media_clock.sample_width_bytes
                    )
                )
                start_ms = index * spec.media_clock.tick_ms
                end_ms = start_ms + duration_ms
                target_wall = wall_zero + end_ms / 1000
                if spec.media_clock.pacing == "realtime":
                    delay = target_wall - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)
                callback_started = time.monotonic()
                self.adapter.push_frame(
                    AudioFrame(
                        index=index,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        pcm16=frame_pcm,
                        final=offset + chunk_bytes >= len(pcm),
                    )
                )
                callback_ms = (time.monotonic() - callback_started) * 1000
                if callback_ms > spec.media_clock.tick_ms:
                    stalled_ticks += 1
                tick_records.append(
                    {
                        "index": index,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "callback_wall_ms": round(callback_ms, 3),
                    }
                )
                media_end_ms = end_ms
            output = self.adapter.finish(media_end_ms)
            assistant_path = output_dir / "assistant.wav"
            _write_pcm16_mono(
                assistant_path,
                output.assistant_pcm16,
                output.assistant_sample_rate,
            )
            completed_unix = time.time()
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
                started_at_unix=started_unix,
                completed_at_unix=completed_unix,
                media_duration_ms=media_end_ms,
                inference_wall_ms=(time.monotonic() - wall_zero) * 1000,
                input_wav=str(input_path),
                assistant_wav=str(assistant_path),
                assistant_text="".join(
                    fragment.text for fragment in output.assistant_fragments
                ).strip(),
                assistant_fragments=output.assistant_fragments,
                assistant_intervals=output.assistant_intervals,
                tool_observations=output.tool_observations,
                capabilities=self.adapter.capabilities,
                runtime_metrics={
                    "tick_ms": spec.media_clock.tick_ms,
                    "ticks_completed": len(tick_records),
                    "stalled_tick_count": stalled_ticks,
                    "tick_records": tick_records,
                    "frame_transport": "gpu_process_local",
                    "input_delivery": "bounded_audio_frame",
                    "execution_mode": capabilities.execution_mode.value,
                    "causal_input_isolation": capabilities.causal_input_isolation.value,
                    "native_input_quantum_ms": capabilities.native_input_quantum_ms,
                    "effective_input_quantum_ms": capabilities.effective_input_quantum_ms
                    or capabilities.native_input_quantum_ms,
                    "committed_output_timing": capabilities.committed_output_timing,
                    "mid_session_cancel": capabilities.mid_session_cancel,
                },
                model_metadata=output.metadata,
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
                started_at_unix=started_unix,
                completed_at_unix=time.time(),
                inference_wall_ms=(time.monotonic() - wall_zero) * 1000,
                input_wav=spec.artifacts.input_wav,
                capabilities=capabilities,
                error=SessionError(
                    error_type=type(exc).__name__,
                    message=str(exc),
                    retryable=False,
                ),
            )
        finally:
            self._active_lock.release()
