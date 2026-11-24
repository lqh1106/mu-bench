"""A persistent offline GPU session exposed to the same logical runner.

Only newly revealed samples cross the control link. Planning and ASR stay on
the networked development host. No resets, tool emulation or text injection.
"""
from __future__ import annotations

import audioop
import json
import math
from pathlib import Path
from types import SimpleNamespace

from .openweight import SessionClient


class OpenweightLogicalAdapter:
    def __init__(self, endpoint_file, *, tick_duration_ms, seed, max_media_ms, input_sample_rate=16000):
        endpoint = json.loads(Path(endpoint_file).read_text())
        protocol = endpoint.get("protocol")
        if (
            protocol is None
            and endpoint.get("schema_version") == "mu-bench-worker-endpoint-v2"
            and endpoint.get("native_quantum_ms") is not None
            and endpoint.get("api_key_file")
            and endpoint.get("base_url")
        ):
            # R4 GPU launchers persist service discovery as endpoint-v2.  The
            # service itself exposes the logical /sessions -> /advance ->
            # /finish protocol, but older discovery files did not carry an
            # explicit protocol string.  Accept this current endpoint schema
            # while still rejecting true whole-track compatibility endpoints.
            protocol = "mu-single-session-logical-v14.0"
        if protocol != "mu-single-session-logical-v14.0":
            raise ValueError("refusing a legacy whole-track worker endpoint")
        token = Path(endpoint["api_key_file"]).read_text().strip()
        self.client = SessionClient(endpoint["base_url"], token,
                                    timeout_seconds=endpoint.get("timeout_seconds", 1800))
        self.tick_ms = tick_duration_ms
        self.input_sample_rate = input_sample_rate
        self.bytes_per_tick = input_sample_rate * 2 * self.tick_ms // 1000
        self.seed = seed
        self.max_media_ms = max_media_ms
        self.provider = SimpleNamespace(session_id=None)
        self.is_connected = False
        self.paused = False
        self.audit_records = []
        self._in_state = self._out_state = None
        self._status = {}

    def connect(self, system_prompt, tools, modality="audio"):
        if system_prompt or tools:
            raise ValueError("openweight no-tool protocol accepts native defaults, never custom text")
        if self.is_connected:
            raise RuntimeError("one persistent session only")
        self._status = self.client.start(seed=self.seed, max_media_ms=self.max_media_ms)
        self.provider.session_id = self._status["session_id"]
        self.output_sample_rate = self._status["output_sample_rate"]
        self.is_connected = True
        self.audit_records.append({"type": "session_started", **self._status})

    def pause_logical_clock(self, reason, logical_time_ms):
        # A returned advance RPC is already a model/codec barrier. No model
        # computation runs in the server between advances. Avoid two extra
        # network round-trips on every empty simulator tick.
        if self.paused or self._status.get("logical_ms", 0) != logical_time_ms:
            raise RuntimeError("pause not at the acknowledged media boundary")
        self.paused = True
        self.audit_records.append({"type": "planning_pause", "logical_ms": logical_time_ms,
                                   "reason": reason, "barrier": "completed_synchronous_advance_rpc"})

    def resume_logical_clock(self, reason, logical_time_ms):
        if not self.paused or self._status.get("logical_ms", 0) != logical_time_ms:
            raise RuntimeError("resume cannot advance the logical clock")
        self.paused = False

    def run_tick(self, user_audio, tick_number):
        if self.paused or not self.is_connected:
            raise RuntimeError("session is paused or disconnected")
        if len(user_audio) != self.bytes_per_tick:
            raise ValueError("wrong logical audio tick size")
        pcm = user_audio
        rate = self._status["input_sample_rate"]
        if rate != self.input_sample_rate:
            pcm, self._in_state = audioop.ratecv(pcm, 2, 1, self.input_sample_rate, rate, self._in_state)
        expected = rate * self.tick_ms // 1000 * 2
        # ratecv's initial interpolation endpoint can be shorter by <1 source
        # sample. Preserve exact logical input duration; record this conversion.
        difference = expected - len(pcm)
        if difference < 0 or difference > math.ceil(rate / self.input_sample_rate) * 2:
            raise ValueError("unexpected sample-rate conversion length drift")
        pcm = pcm.ljust(expected, b"\0")
        self._status = self.client.step_pcm(pcm, tick_number * self.tick_ms)
        output_pcm = self._status.pop("assistant_pcm16")
        output_rate = self._status["output_sample_rate"]
        played = output_pcm
        expected_output = output_rate * self.tick_ms // 1000 * 2
        if len(played) != expected_output:
            raise ValueError("GPU returned a different logical output duration")
        events = self._status.get("events", [])
        self.audit_records.append({"type": "native_advance", "logical_ms": self._status["logical_ms"],
            "processed_end_ms": self._status["processed_end_ms"], "input_conversion_padding_bytes": difference,
            "events": events})
        return SimpleNamespace(
            events=[SimpleNamespace(type=e.get("type", "native_event")) for e in events],
            tool_calls=[], vad_events=[], was_truncated=False, truncated_audio_bytes=0,
            proportional_transcript="", get_played_agent_audio=lambda: played)

    def is_output_drained(self):
        # Continuous codecs produce silence frames indefinitely; an empty
        # queue is not a reachable terminal state for those models. The runner
        # additionally requires a played-silence guard and closure decision.
        return (self._status.get("pending_output_samples", 0) == 0
                or self._status.get("pending_output_rms_int16", float("inf")) <= 80)

    def disconnect(self):
        if self.is_connected:
            try:
                self.audit_records.append({"type": "session_finished", **self.client.finish()})
            except Exception as finish_error:
                # A failed finish must not leave a max_sessions=1 worker
                # permanently occupied. Abort only this known session; never
                # create a replacement context or replay model audio.
                try:
                    abort = self.client.abort() if hasattr(self.client, "abort") else None
                    self.audit_records.append({"type": "session_aborted_after_finish_error",
                                               **(abort or {"aborted": False})})
                except Exception as abort_error:
                    self.audit_records.append({"type": "session_abort_failed",
                                               "finish_error": type(finish_error).__name__,
                                               "abort_error": type(abort_error).__name__})
                raise
            finally:
                self.is_connected = False

    def force_abort(self, *, reason: str = "signal_or_supervisor_cleanup"):
        """Best-effort release for SIGTERM / hard abort paths."""
        if not getattr(self, "is_connected", False) and not getattr(self.client, "session_id", None):
            return {"aborted": False, "reason": reason, "already_disconnected": True}
        try:
            if hasattr(self.client, "abort") and self.client.session_id is not None:
                result = self.client.abort()
                self.audit_records.append({"type": "session_force_aborted", "reason": reason, **(result if isinstance(result, dict) else {})})
                self.is_connected = False
                return {"aborted": True, "reason": reason, "result": result}
            self.disconnect()
            return {"aborted": True, "reason": reason, "via": "disconnect"}
        except Exception as exc:
            self.is_connected = False
            return {"aborted": False, "reason": reason, "error_type": type(exc).__name__, "error": str(exc)}


class PlayedPrefixObserver:
    """Apply the existing audible-ASR validator to delivered audio only."""
    def __init__(self, *, artifact_dir, bridge, values, snapshot):
        self.root = Path(artifact_dir)
        self.bridge, self.values, self.snapshot = bridge, values, snapshot
        self.last_digest = None
        self.last_result = None
        self.index = 0

    def observe(self, observation=None):
        import hashlib
        import wave
        from .audible_asr_v3 import transcribe_response
        pcm, rate, task_start_ms, intervals = self.snapshot()
        digest = hashlib.sha256(pcm).hexdigest()
        if digest != self.last_digest:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / f"played_prefix_{self.index:05d}.wav"
            self.index += 1
            if path.exists():
                raise ValueError("ASR prefix artifacts must not be overwritten")
            with wave.open(str(path), "wb") as wav:
                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate); wav.writeframes(pcm)
            try:
                self.last_result = transcribe_response({"assistant_wav": str(path.resolve()),
                    "assistant_intervals": intervals}, task_start_ms, self.root, self.bridge, self.values)
            except Exception as exc:
                from .audible_asr_v3 import AudibleASRError
                if not isinstance(exc, AudibleASRError):
                    raise
                self.last_result = {
                    "asr_derived": True,
                    "asr_transcript_scope": "task_only_actual_assistant_waveform",
                    "status": "asr_failed",
                    "assistant_text": "",
                    "assistant_fragments": [],
                    "assistant_intervals": intervals,
                    "asr_failure": str(exc),
                    "asr_failure_policy": "report_failed_asr_result_after_retries",
                }
                (self.root / f"played_prefix_{self.index-1:05d}.asr_failed.json").write_text(
                    json.dumps(self.last_result, ensure_ascii=False, indent=2) + "\n")
                # Missing observation is a technical failure, never an empty answer.
                raise
            self.last_digest = digest
        if observation is None:
            return self.last_result
        return observation.model_copy(update={"assistant_committed_prefix": self.last_result.get("assistant_text", "")})
