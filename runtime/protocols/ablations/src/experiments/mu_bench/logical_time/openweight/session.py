"""One inference context, no hidden stepping while the planner is thinking.

The backend owns KV/codec/generator state until finish. The process actor in
service.py additionally isolates global Python/NumPy/Torch RNG state. Pause is
an acknowledgement barrier, not state serialization and not a model reset.
"""
from __future__ import annotations

import base64
from array import array
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Protocol
import wave


class SessionProtocolError(RuntimeError):
    pass


@dataclass
class NativeOutput:
    pcm16: bytes = b""
    # Native generated text is not necessarily the text of retained audio.
    # It is diagnostic-only. The development host may ASR committed audio.
    diagnostic_text: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)


class NativeBackend(Protocol):
    input_sample_rate: int
    output_sample_rate: int
    native_quantum_ms: int

    def start(self, *, seed: int, max_media_ms: int) -> None: ...
    def step(self, pcm16: bytes) -> NativeOutput: ...
    def finish(self) -> None: ...


def _samples(ms: int, rate: int) -> int:
    if ms * rate % 1000:
        raise SessionProtocolError("logical time must align to an audio sample")
    return ms * rate // 1000


def _gpu_memory_snapshot() -> dict[str, Any]:
    """Best-effort per-process CUDA allocator telemetry.

    This runs inside the native actor process, so PyTorch allocator counters
    describe the process that owns the model/cache rather than the HTTP parent.
    """
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return {"cuda_available": False}
        device = torch.cuda.current_device()
        stats = torch.cuda.memory_stats(device)
        return {
            "cuda_available": True,
            "cuda_device": int(device),
            "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "gpu_memory_active_bytes": int(stats.get("active_bytes.all.current", 0)),
            "gpu_memory_inactive_split_bytes": int(stats.get("inactive_split_bytes.all.current", 0)),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"cuda_available": None, "gpu_memory_error": f"{type(exc).__name__}: {exc}"}


class StatefulSession:
    """A synchronous context owner; never calls backend.step outside advance.

    A request may batch native frames, but advance ends exactly at its supplied
    audio boundary. Partial native frames remain buffered, including across a
    planning pause. Generated audio only becomes playable at the native frame
    end that produced it. Audio beyond the current logical horizon is withheld.
    """

    def __init__(self, backend: NativeBackend, *, session_id: str, seed: int,
                 max_media_ms: int, output_dir: str | Path | None = None):
        if max_media_ms <= 0:
            raise SessionProtocolError("max_media_ms must be positive")
        for attr in ("input_sample_rate", "output_sample_rate", "native_quantum_ms"):
            if getattr(backend, attr, 0) <= 0:
                raise SessionProtocolError(f"invalid backend {attr}")
        self.backend = backend
        self.session_id = session_id
        self.max_media_ms = max_media_ms
        self.logical_ms = self.processed_ms = self.sequence = 0
        self.paused = self.closed = self.failed = False
        self.pending_input = bytearray()
        self.pending_output: list[tuple[int, bytes]] = []
        self.output_cursor = 0
        self.input_pcm = bytearray()
        self.played_pcm = bytearray()
        self.audit: list[dict[str, Any]] = []
        self.responses: dict[str, tuple[str, dict[str, Any]]] = {}
        self.lock = threading.RLock()
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=False)
        backend.start(seed=seed, max_media_ms=max_media_ms)
        self._record("start", seed=seed, native_quantum_ms=backend.native_quantum_ms)

    def _record(self, kind: str, **fields: Any) -> None:
        event = {"type": kind, "logical_ms": self.logical_ms,
                 "processed_ms": self.processed_ms, "wall_monotonic": time.monotonic(),
                 **fields}
        self.audit.append(event)
        if self.output_dir:
            with (self.output_dir / "context_events.jsonl").open("a", encoding="utf-8") as out:
                out.write(json.dumps(event, ensure_ascii=False) + "\n")

    def status(self) -> dict[str, Any]:
        with self.lock:
            played_samples = _samples(self.logical_ms, self.backend.output_sample_rate)
            unplayed = sum(max(0, offset+len(chunk)//2-max(offset, played_samples))
                           for offset, chunk in self.pending_output)
            sum_squares = 0
            for offset, chunk in self.pending_output:
                tail = chunk[max(0, played_samples-offset)*2:]
                values = array("h", tail)
                sum_squares += sum(int(value)*int(value) for value in values)
            rms = (sum_squares/max(unplayed, 1))**0.5 / 32768
            return {"session_id": self.session_id, "logical_ms": self.logical_ms,
                    "processed_end_ms": self.processed_ms, "sequence": self.sequence,
                    "paused": self.paused, "closed": self.closed, "failed": self.failed,
                    "native_process_pid": os.getpid(),
                    "buffered_input_samples": len(self.pending_input) // 2,
                    "buffered_input_bytes": len(self.pending_input),
                    "pending_output_samples": unplayed, "unplayed_bytes": unplayed*2,
                    "pending_output_chunks": len(self.pending_output),
                    "pending_output_rms": rms,
                    "pending_output_rms_int16": rms*32768,
                    "response_completion_signal": "unavailable_use_committed_audio_silence_guard",
                    "input_sample_rate": self.backend.input_sample_rate,
                    "output_sample_rate": self.backend.output_sample_rate,
                    "native_quantum_ms": self.backend.native_quantum_ms,
                    "state_retention": "same_live_process_and_native_objects",
                    "retained_response_count": len(self.responses),
                    **_gpu_memory_snapshot(),
                    "restart_resume_supported": False, "wall_clock_timing_eligible": False}

    def append_context(self, *, payload, logical_ms, request_id):
        with self.lock:
            if self.closed or self.failed or logical_ms != self.logical_ms:
                raise SessionProtocolError('context must match the current live media boundary')
            if not isinstance(payload,dict) or payload.get('kind') not in ['trusted_routing','structured_user_input','ablation_protocol']:
                raise SessionProtocolError('unsupported auxiliary record')
            raw = json.dumps(payload,sort_keys=True).encode()
            if not request_id or len(raw) > 65536:
                raise SessionProtocolError('invalid context size or id')
            digest = hashlib.sha256(raw).hexdigest()
            if not hasattr(self,'context_receipts'):self.context_receipts={}
            if request_id in self.context_receipts:
                old = self.context_receipts[request_id]
                if old['payload_sha256'] != digest:raise SessionProtocolError('conflicting context id')
                return old
            if not hasattr(self.backend,'queue_context'):
                raise SessionProtocolError('backend does not support auxiliary records')
            self.backend.queue_context(payload)
            receipt = {'queued':True,'applied':False,'logical_ms':self.logical_ms,'sequence':self.sequence,
                       'payload_sha256':digest,'request_id':request_id}
            self.context_receipts[request_id]=receipt
            self._record('ablation_context_queued',payload=payload,**{k:v for k,v in receipt.items() if k!='logical_ms'})
            return receipt

    def pause(self) -> dict[str, Any]:
        with self.lock:
            if self.closed or self.failed:
                raise SessionProtocolError("session is terminal")
            self.paused = True
            self._record("planning_pause")
            return self.status()

    def resume(self) -> dict[str, Any]:
        with self.lock:
            if self.closed or self.failed:
                raise SessionProtocolError("session is terminal")
            self.paused = False
            self._record("planning_resume")
            return self.status()

    def advance(self, pcm16: bytes, *, sim_start_ms: int, request_id: str,
                sequence: int) -> dict[str, Any]:
        """Consume only newly visible PCM; exact repeated requests are idempotent."""
        if not request_id or len(pcm16) % 2:
            raise SessionProtocolError("request_id and complete PCM16 samples required")
        fingerprint = hashlib.sha256(
            json.dumps([sim_start_ms, sequence]).encode() + pcm16).hexdigest()
        with self.lock:
            if request_id in self.responses:
                old_fp, result = self.responses[request_id]
                if old_fp != fingerprint:
                    raise SessionProtocolError("request_id reused with different input")
                return result
            if self.closed or self.failed or self.paused:
                raise SessionProtocolError("session is closed, failed, or paused")
            if sim_start_ms != self.logical_ms or sequence != self.sequence:
                raise SessionProtocolError("out-of-order or discontinuous advance")
            sample_count = len(pcm16) // 2
            rate = self.backend.input_sample_rate
            if sample_count == 0 or sample_count * 1000 % rate:
                raise SessionProtocolError("advance must have a positive integral-ms duration")
            end_ms = sim_start_ms + sample_count * 1000 // rate
            if end_ms > self.max_media_ms:
                raise SessionProtocolError("session max_media_ms exceeded before native inference")
            self.pending_input.extend(pcm16)
            self.input_pcm.extend(pcm16)
            quantum = self.backend.native_quantum_ms
            frame_bytes = _samples(quantum, rate) * 2
            emitted_events: list[dict[str, Any]] = []
            try:
                while len(self.pending_input) >= frame_bytes:
                    frame = bytes(self.pending_input[:frame_bytes])
                    del self.pending_input[:frame_bytes]
                    result = self.backend.step(frame)
                    if len(result.pcm16) % 2:
                        raise SessionProtocolError("backend returned incomplete PCM16 samples")
                    self.processed_ms += quantum
                    available_sample = _samples(self.processed_ms, self.backend.output_sample_rate)
                    if any(event.get("type") == "audio_interrupt" for event in result.events):
                        retained = []
                        for offset, chunk in self.pending_output:
                            count = min(len(chunk)//2, max(0, available_sample - offset))
                            if count:
                                retained.append((offset, chunk[:count*2]))
                        self.pending_output = retained
                        self.output_cursor = min(self.output_cursor, available_sample)
                    if result.pcm16:
                        start = max(available_sample, self.output_cursor)
                        self.pending_output.append((start, result.pcm16))
                        self.output_cursor = start + len(result.pcm16) // 2
                    # Diagnostic content never enters the planner-facing result.
                    self._record("native_step", available_at_ms=self.processed_ms,
                                 diagnostic_text=result.diagnostic_text,
                                 emitted_samples=len(result.pcm16) // 2,
                                 native_events=result.events)
                    emitted_events.extend(result.events)
            except Exception as exc:
                self.failed = True
                self._record("native_failure", error_type=type(exc).__name__, error=str(exc))
                raise
            start_sample = _samples(sim_start_ms, self.backend.output_sample_rate)
            end_sample = _samples(end_ms, self.backend.output_sample_rate)
            played = bytearray((end_sample - start_sample) * 2)
            kept = []
            for offset, chunk in self.pending_output:
                stop = offset + len(chunk) // 2
                lo, hi = max(start_sample, offset), min(end_sample, stop)
                if hi > lo:
                    played[(lo-start_sample)*2:(hi-start_sample)*2] = chunk[(lo-offset)*2:(hi-offset)*2]
                if stop > end_sample:
                    kept.append((offset, chunk))
            self.pending_output = kept
            self.played_pcm.extend(played)
            self.logical_ms = end_ms
            self.sequence += 1
            self._record("advance_committed", request_id=request_id, input_sha256=hashlib.sha256(pcm16).hexdigest(),
                         played_sha256=hashlib.sha256(played).hexdigest(), sequence=sequence)
            response = {**self.status(), "input_end_ms": end_ms,
                        "output_start_ms": sim_start_ms, "committed_end_ms": end_ms,
                        "assistant_pcm16_b64": base64.b64encode(played).decode(),
                        "played_text": "", "requires_audible_asr": True,
                        "events": emitted_events}
            self.responses[request_id] = (fingerprint, response)
            return response

    def finish(self) -> dict[str, Any]:
        """Release state, with no padding, decoding, or hidden tail generation."""
        with self.lock:
            if not self.closed:
                dropped_unplayed_samples = sum(len(p)//2 for _, p in self.pending_output)
                unprocessed_input_samples = len(self.pending_input)//2
                try:
                    self.backend.finish()
                finally:
                    self.closed = True
                    self._record("finish", dropped_unplayed_samples=dropped_unplayed_samples,
                                 unprocessed_input_samples=unprocessed_input_samples)
                    if self.output_dir:
                        for name, rate, pcm in (("user.wav", self.backend.input_sample_rate, self.input_pcm),
                                                ("assistant.wav", self.backend.output_sample_rate, self.played_pcm)):
                            with wave.open(str(self.output_dir / name), "wb") as wav:
                                wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(rate)
                                wav.writeframes(bytes(pcm))
                    self.pending_input.clear()
                    self.pending_output.clear()
                    self.responses.clear()
                    self.input_pcm.clear()
                    self.played_pcm.clear()
            return self.status()
