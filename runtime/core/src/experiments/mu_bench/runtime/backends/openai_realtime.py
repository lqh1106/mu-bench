"""In-process OpenAI Realtime SessionExecutor.

This adapter owns one Realtime WebSocket session and exposes the same
SessionSpec/SessionResult boundary as the RJob worker. Case-level planning
remains outside this class; the executor only streams one immutable input track
and records actual wall-clock output timing.
"""

from __future__ import annotations

import audioop
from pathlib import Path
import threading
import time
from typing import Any, Callable
import wave

from experiments.mu_bench.runtime.contracts import (
    AssistantFragment,
    AssistantInterval,
    BackendCapabilities,
    CausalInputIsolation,
    ExecutionMode,
    SessionResult,
    SessionSpec,
    SessionState,
)
from experiments.mu_bench.runtime.storage import GPFSArtifactStore
from experiments.tau_multiuser_voice.version import (
    CLOSED_SOURCE_SYSTEM_PROMPT_POLICY,
    OPEN_WEIGHT_SYSTEM_PROMPT_POLICY,
)


class OpenAIRealtimeSessionExecutor:
    """Execute one immutable Realtime SessionSpec on the development machine."""

    def __init__(
        self,
        *,
        store: GPFSArtifactStore,
        runner_module: Any,
        agent_module: Any,
        audio_format: Any,
        model_name: str,
        tools_factory: Callable[[SessionSpec], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.store = store
        self.runner_module = runner_module
        self.agent_module = agent_module
        self.audio_format = audio_format
        self.model_name = model_name
        self.tools_factory = tools_factory or (lambda _spec: [])
        self._capabilities = BackendCapabilities(
            backend_name="openai_realtime_api_inprocess",
            model_name=model_name,
            incremental_audio_input=True,
            whole_track_audio_input=False,
            incremental_audio_output=True,
            dynamic_text_control_items=False,
            native_tool_calling=True,
            explicit_speaker_attribution=True,
            explicit_addressee_attribution=True,
            media_tick_ms=200,
            native_input_quantum_ms=20,
            effective_input_quantum_ms=20,
            execution_mode=ExecutionMode.CAUSAL_STREAMING,
            causal_input_isolation=CausalInputIsolation.ENFORCED,
            committed_output_timing=True,
            mid_session_cancel=True,
            supports_wall_clock_timing=True,
            notes=[
                "Realtime WebSocket runs on the networked development machine.",
                "Input/output pacing and timestamps use the benchmark wall clock.",
                "Closed-source sessions may receive a benchmark protocol prompt; identity is audio-only.",
                "The provider's native function-call surface is passed through by the adapter.",
            ],
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    @staticmethod
    def _read_input(path: Path, expected_rate: int) -> bytes:
        with wave.open(str(path), "rb") as source:
            if source.getnchannels() != 1 or source.getsampwidth() != 2:
                raise ValueError("Realtime SessionSpec input must be mono PCM16 WAV")
            if source.getframerate() != expected_rate:
                raise ValueError(
                    f"Realtime input sample rate {source.getframerate()} "
                    f"does not match SessionSpec {expected_rate}"
                )
            return source.readframes(source.getnframes())

    @staticmethod
    def _cancelled(spec: SessionSpec, started: float) -> SessionResult:
        return SessionResult(
            session_id=spec.session_id,
            attempt_id=spec.attempt_id,
            case_id=spec.case_id,
            phase=spec.phase,
            state=SessionState.CANCELLED,
            model_name=spec.model_name,
            backend_name=spec.backend_name,
            execution_mode=ExecutionMode.CAUSAL_STREAMING,
            causal_input_isolation=CausalInputIsolation.ENFORCED,
            started_at_unix=started,
            completed_at_unix=time.time(),
            input_wav=spec.artifacts.input_wav,
            capabilities=BackendCapabilities(
                backend_name=spec.backend_name,
                model_name=spec.model_name,
                incremental_audio_input=True,
                whole_track_audio_input=False,
                incremental_audio_output=True,
                dynamic_text_control_items=False,
                native_tool_calling=True,
                explicit_speaker_attribution=True,
                explicit_addressee_attribution=True,
                media_tick_ms=spec.media_clock.tick_ms,
                native_input_quantum_ms=20,
                effective_input_quantum_ms=20,
                execution_mode=ExecutionMode.CAUSAL_STREAMING,
                causal_input_isolation=CausalInputIsolation.ENFORCED,
                committed_output_timing=True,
                mid_session_cancel=True,
                supports_wall_clock_timing=True,
            ),
        )

    def execute(
        self,
        spec: SessionSpec,
        *,
        cancel_event: threading.Event | None = None,
    ) -> SessionResult:
        started_unix = time.time()
        wall_started = time.monotonic()
        input_path = self.store.resolve(spec.artifacts.input_wav)
        output_dir = self.store.ensure_output_dir(spec.artifacts.output_dir)
        prompt = None
        if spec.artifacts.system_prompt_path is not None:
            prompt = self.store.resolve(
                spec.artifacts.system_prompt_path
            ).read_text(encoding="utf-8")
        pcm = self._read_input(input_path, spec.media_clock.input_sample_rate)
        tick_ms = spec.media_clock.tick_ms
        samples_per_tick = spec.media_clock.samples_per_tick
        bytes_per_tick = samples_per_tick * spec.media_clock.sample_width_bytes
        tick_bytes = bytes_per_tick
        input_tick_count = (len(pcm) + bytes_per_tick - 1) // bytes_per_tick
        input_duration_ms = input_tick_count * tick_ms
        controls = list(spec.metadata.get("control_events") or [])
        if controls:
            raise ValueError(
                "spoken-only enrollment forbids model-visible text control events"
            )

        adapter, _ = self.agent_module.create_adapter(
            tick_duration_ms=tick_ms,
            send_audio_instant=True,
            reasoning_effort=None,
            audio_format=self.audio_format,
        )
        converter = self.runner_module.ULAWConverter()
        agent_ticks: list[bytes] = []
        fragments: list[AssistantFragment] = []
        intervals: list[AssistantInterval] = []
        tick_records: list[dict[str, Any]] = []
        assistant_speaking = False
        interval_start: int | None = None
        response_done_seen = False
        quiet_since_ms: int | None = None
        auto_response_enabled = False
        stalled_tick_count = 0
        max_tick_processing_ms = 0.0
        drain_ms = int(spec.metadata.get("final_drain_ms", 30_000))
        quiet_guard_ms = int(spec.metadata.get("completion_guard_ms", 600))
        max_ticks = input_tick_count + max(1, (drain_ms + tick_ms - 1) // tick_ms)

        try:
            adapter.connect(prompt, self.tools_factory(spec), modality="audio")
            self.runner_module.set_realtime_auto_response(adapter, enabled=False)
            wall_epoch = time.monotonic()
            for tick in range(max_ticks):
                if cancel_event is not None and cancel_event.is_set():
                    return self._cancelled(spec, started_unix)
                session_time_ms = tick * tick_ms
                if spec.media_clock.pacing == "realtime":
                    target = wall_epoch + session_time_ms / 1000.0
                    remaining = target - time.monotonic()
                    if remaining > 0:
                        time.sleep(remaining)
                if session_time_ms >= spec.media_clock.task_start_ms and not auto_response_enabled:
                    self.runner_module.set_realtime_auto_response(adapter, enabled=True)
                    auto_response_enabled = True
                start = tick * bytes_per_tick
                frame = pcm[start : start + bytes_per_tick]
                frame = frame.ljust(bytes_per_tick, b"\x00")
                if tick >= input_tick_count:
                    frame = b"\x00" * tick_bytes
                user_ulaw = converter.convert(frame, adapter.bytes_per_tick)
                tick_started = time.monotonic()
                result = adapter.run_tick(user_ulaw, tick_number=tick)
                processing_ms = (time.monotonic() - tick_started) * 1000
                max_tick_processing_ms = max(max_tick_processing_ms, processing_ms)
                if processing_ms > tick_ms:
                    stalled_tick_count += 1
                agent_ulaw = result.get_played_agent_audio()
                agent_ticks.append(agent_ulaw)
                event_types = [getattr(event, "type", "unknown") for event in result.events]
                if "response.done" in event_types:
                    response_done_seen = True
                fragment = str(result.proportional_transcript or "")
                if fragment:
                    fragments.append(
                        AssistantFragment(timestamp_ms=session_time_ms, text=fragment)
                    )
                agent_pcm = audioop.ulaw2lin(agent_ulaw, 2)
                now_speaking = audioop.rms(agent_pcm, 2) > 80
                if now_speaking and not assistant_speaking:
                    interval_start = session_time_ms
                elif not now_speaking and assistant_speaking and interval_start is not None:
                    intervals.append(
                        AssistantInterval(start_ms=interval_start, end_ms=session_time_ms)
                    )
                    interval_start = None
                assistant_speaking = now_speaking
                tick_records.append(
                    {
                        "tick": tick,
                        "session_time_ms": session_time_ms,
                        "assistant_speaking": now_speaking,
                        "assistant_transcript_delta": fragment,
                        "realtime_event_types": event_types,
                        "was_truncated": bool(result.was_truncated),
                    }
                )
                if tick >= input_tick_count:
                    if response_done_seen and not now_speaking:
                        quiet_since_ms = quiet_since_ms or session_time_ms
                        if session_time_ms - quiet_since_ms >= quiet_guard_ms:
                            break
                    else:
                        quiet_since_ms = None
        finally:
            if adapter.is_connected:
                adapter.disconnect()

        if interval_start is not None:
            intervals.append(
                AssistantInterval(start_ms=interval_start, end_ms=max_ticks * tick_ms)
            )
        assistant_pcm = audioop.ulaw2lin(b"".join(agent_ticks), 2)
        assistant_wav = output_dir / "assistant.wav"
        self.runner_module.write_mono_wav(assistant_wav, assistant_pcm, 8_000)
        self.runner_module.write_json(
            output_dir / "ticks.json",
            tick_records,
        )
        self.runner_module.write_json(
            output_dir / "control_events.json",
            [],
        )
        wall_ms = (time.monotonic() - wall_started) * 1000
        return SessionResult(
            session_id=spec.session_id,
            attempt_id=spec.attempt_id,
            case_id=spec.case_id,
            phase=spec.phase,
            state=SessionState.SUCCEEDED,
            model_name=spec.model_name,
            backend_name=spec.backend_name,
            execution_mode=self.capabilities.execution_mode,
            causal_input_isolation=self.capabilities.causal_input_isolation,
            started_at_unix=started_unix,
            completed_at_unix=time.time(),
            media_duration_ms=input_duration_ms,
            inference_wall_ms=wall_ms,
            input_wav=str(input_path),
            assistant_wav=str(assistant_wav.resolve()),
            assistant_text=" ".join(item.text for item in fragments).strip(),
            assistant_fragments=fragments,
            assistant_intervals=intervals,
            capabilities=self.capabilities,
            runtime_metrics={
                "frame_transport": "development_machine_process_local",
                "input_delivery": "bounded_audio_frame",
                "execution_mode": self.capabilities.execution_mode.value,
                "causal_input_isolation": self.capabilities.causal_input_isolation.value,
                "tick_ms": tick_ms,
                "native_input_quantum_ms": self.capabilities.native_input_quantum_ms,
                "effective_input_quantum_ms": self.capabilities.effective_input_quantum_ms
                or self.capabilities.native_input_quantum_ms,
                "max_tick_processing_ms": max_tick_processing_ms,
                "stalled_tick_count": stalled_tick_count,
                "wall_timing_eligible": stalled_tick_count == 0,
                "delivered_control_count": 0,
                "system_prompt_policy": spec.metadata.get(
                    "system_prompt_policy",
                    (
                        CLOSED_SOURCE_SYSTEM_PROMPT_POLICY
                        if spec.artifacts.system_prompt_path is not None
                        else OPEN_WEIGHT_SYSTEM_PROMPT_POLICY
                    ),
                ),
                "identity_delivery": "spoken_tts_self_introduction",
            },
            model_metadata={
                "provider_session_id": getattr(adapter.provider, "session_id", None),
            },
        )
