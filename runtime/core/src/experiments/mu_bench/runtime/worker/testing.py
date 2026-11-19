"""Deterministic local adapter used for protocol and deployment smoke tests."""

from __future__ import annotations

from experiments.mu_bench.runtime.contracts import (
    AssistantFragment,
    AssistantInterval,
    BackendCapabilities,
    SessionSpec,
)
from experiments.mu_bench.runtime.worker.media_clock import (
    AdapterSessionOutput,
    AudioFrame,
)


class DeterministicDuplexAdapter:
    """Record every 200 ms callback and emit a short deterministic tone."""

    def __init__(self) -> None:
        self.frames: list[AudioFrame] = []
        self.spec: SessionSpec | None = None
        self.prompt = ""
        self.cancelled = False

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend_name="deterministic-local-smoke",
            model_name="deterministic-duplex-adapter",
            media_tick_ms=200,
            native_input_quantum_ms=200,
        )

    def start(self, spec: SessionSpec, system_prompt: str | None) -> None:
        self.frames = []
        self.spec = spec
        self.prompt = system_prompt
        self.cancelled = False

    def push_frame(self, frame: AudioFrame) -> None:
        self.frames.append(frame)

    def finish(self, media_end_ms: int) -> AdapterSessionOutput:
        sample_rate = 16_000
        tone_ms = 200
        sample_count = sample_rate * tone_ms // 1000
        amplitude = 1200
        pcm = b"".join(int(amplitude).to_bytes(2, "little", signed=True) for _ in range(sample_count))
        start_ms = max(0, media_end_ms - tone_ms)
        return AdapterSessionOutput(
            assistant_pcm16=pcm,
            assistant_sample_rate=sample_rate,
            assistant_fragments=[
                AssistantFragment(timestamp_ms=start_ms, text="smoke complete")
            ],
            assistant_intervals=[AssistantInterval(start_ms=start_ms, end_ms=media_end_ms)],
            metadata={
                "received_frame_count": len(self.frames),
                "received_frame_ms": [frame.end_ms - frame.start_ms for frame in self.frames],
            },
        )

    def cancel(self) -> None:
        self.cancelled = True
