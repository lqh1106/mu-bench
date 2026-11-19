"""Per-speaker PCM queues and sample-accurate mono microphone mixing."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel, Field


class TTSBackend(Protocol):
    """Injectable speech synthesizer; production adapters may call any provider."""

    sample_rate: int

    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Return mono signed 16-bit little-endian PCM."""


class QueuedAudio(BaseModel):
    """One immutable utterance scheduled on a speaker stem."""

    audio_id: str
    speaker_id: str
    text: str
    start_ms: int
    pcm_s16le: bytes
    sample_rate: int
    committed_until_ms: int = 0
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def duration_ms(self) -> int:
        samples = len(self.pcm_s16le) // 2
        return math.ceil(samples * 1000 / self.sample_rate)

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms


class SpeakerAudioQueue:
    """Maintain cancellable future audio separately for each speaker."""

    def __init__(self, speaker_id: str, sample_rate: int) -> None:
        self.speaker_id = speaker_id
        self.sample_rate = sample_rate
        self.items: list[QueuedAudio] = []

    def schedule(self, item: QueuedAudio) -> None:
        """Add a non-retroactive utterance to this speaker's stem."""
        if item.speaker_id != self.speaker_id:
            raise ValueError("audio scheduled on the wrong speaker queue")
        if item.sample_rate != self.sample_rate:
            raise ValueError("all queue audio must use the same sample rate")
        if any(existing.audio_id == item.audio_id for existing in self.items):
            raise ValueError(f"duplicate audio id: {item.audio_id}")
        self.items.append(item)
        self.items.sort(key=lambda audio: (audio.start_ms, audio.audio_id))

    def commit_through(self, sim_time_ms: int) -> None:
        """Mark audio before the simulation clock as immutable."""
        for item in self.items:
            if sim_time_ms > item.start_ms:
                item.committed_until_ms = min(
                    item.duration_ms, sim_time_ms - item.start_ms
                )

    def cancel_pending(self, audio_id: str, at_ms: int) -> None:
        """Cancel unplayed samples while preserving the committed prefix."""
        for item in list(self.items):
            if item.audio_id != audio_id:
                continue
            if at_ms <= item.start_ms:
                self.items.remove(item)
                return
            keep_ms = min(item.duration_ms, at_ms - item.start_ms)
            keep_samples = int(keep_ms * self.sample_rate / 1000)
            item.pcm_s16le = item.pcm_s16le[: keep_samples * 2]
            item.committed_until_ms = keep_ms
            return
        raise ValueError(f"unknown audio id: {audio_id}")


class MixedAudioWindow(BaseModel):
    """Mono SUT input plus oracle stems for one clock window."""

    start_ms: int
    end_ms: int
    mixed_pcm_s16le: bytes
    stems_pcm_s16le: dict[str, bytes]
    active_speakers: list[str]


class PCMTrackMixer:
    """Render speaker stems and a peak-safe single microphone signal."""

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

    def _render_stem(
        self,
        items: list[QueuedAudio],
        start_ms: int,
        end_ms: int,
    ) -> np.ndarray:
        sample_count = int((end_ms - start_ms) * self.sample_rate / 1000)
        output = np.zeros(sample_count, dtype=np.float32)
        for item in items:
            overlap_start = max(start_ms, item.start_ms)
            overlap_end = min(end_ms, item.end_ms)
            if overlap_start >= overlap_end:
                continue
            source = np.frombuffer(item.pcm_s16le, dtype="<i2").astype(np.float32)
            source_start = int(
                (overlap_start - item.start_ms) * self.sample_rate / 1000
            )
            target_start = int((overlap_start - start_ms) * self.sample_rate / 1000)
            length = min(
                int((overlap_end - overlap_start) * self.sample_rate / 1000),
                len(source) - source_start,
                len(output) - target_start,
            )
            if length > 0:
                output[target_start : target_start + length] += source[
                    source_start : source_start + length
                ]
        return output

    @staticmethod
    def _to_pcm(samples: np.ndarray) -> bytes:
        return np.clip(samples, -32768, 32767).astype("<i2").tobytes()

    def mix_window(
        self,
        queues: dict[str, SpeakerAudioQueue],
        *,
        start_ms: int,
        end_ms: int,
    ) -> MixedAudioWindow:
        """Mix one sample-aligned interval while retaining every speaker stem."""
        if end_ms <= start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        stems = {
            speaker_id: self._render_stem(queue.items, start_ms, end_ms)
            for speaker_id, queue in queues.items()
        }
        active = [
            speaker_id for speaker_id, samples in stems.items() if np.any(samples != 0)
        ]
        if stems:
            mixed = np.sum(list(stems.values()), axis=0)
        else:
            sample_count = int((end_ms - start_ms) * self.sample_rate / 1000)
            mixed = np.zeros(sample_count, dtype=np.float32)
        peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
        if peak > 32767:
            mixed *= 32767 / peak
        for queue in queues.values():
            queue.commit_through(end_ms)
        return MixedAudioWindow(
            start_ms=start_ms,
            end_ms=end_ms,
            mixed_pcm_s16le=self._to_pcm(mixed),
            stems_pcm_s16le={
                speaker_id: self._to_pcm(samples)
                for speaker_id, samples in stems.items()
            },
            active_speakers=active,
        )

    def write_session(
        self,
        queues: dict[str, SpeakerAudioQueue],
        output_dir: Path,
        end_ms: int,
    ) -> dict[str, Path]:
        """Write all oracle stems and the mono user microphone track."""
        output_dir.mkdir(parents=True, exist_ok=True)
        window = self.mix_window(queues, start_ms=0, end_ms=end_ms)
        paths: dict[str, Path] = {}
        for speaker_id, pcm in window.stems_pcm_s16le.items():
            path = output_dir / f"{speaker_id}_stem.wav"
            self._write_wav(path, pcm)
            paths[speaker_id] = path
        mixed_path = output_dir / "mixed_user.wav"
        self._write_wav(mixed_path, window.mixed_pcm_s16le)
        paths["mixed_user"] = mixed_path
        return paths

    def _write_wav(self, path: Path, pcm_s16le: bytes) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(pcm_s16le)


class DeterministicToneTTS:
    """Dependency-free test synthesizer; tones are not benchmark speech."""

    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Encode text length and voice id into a deterministic audible tone."""
        duration_seconds = min(2.5, max(0.35, len(text.split()) * 0.16))
        sample_count = int(duration_seconds * self.sample_rate)
        time_axis = np.arange(sample_count, dtype=np.float32) / self.sample_rate
        voice_hash = sum(voice_id.encode())
        frequency = 180 + voice_hash % 260
        envelope = np.minimum(1.0, np.arange(sample_count) / 240.0)
        envelope *= np.minimum(1.0, np.arange(sample_count, 0, -1) / 240.0)
        samples = 6000 * envelope * np.sin(2 * np.pi * frequency * time_axis)
        return samples.astype("<i2").tobytes()
