#!/usr/bin/env python3
"""Audio normalization and speech-interval helpers shared by GPU workers."""

from __future__ import annotations

import audioop
import math
from pathlib import Path
import wave

import numpy as np


def read_wav_mono(path: Path, target_rate: int) -> np.ndarray:
    """Read ordinary PCM WAV without depending on librosa inside GPU images."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels == 2:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
    elif channels != 1:
        raise ValueError(f"unsupported WAV channel count {channels}: {path}")
    if sample_width != 2:
        frames = audioop.lin2lin(frames, sample_width, 2)
        sample_width = 2
    if sample_rate != target_rate:
        frames, _ = audioop.ratecv(
            frames, sample_width, 1, sample_rate, target_rate, None
        )
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def write_wav_mono(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(np.asarray(audio), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as handle:
        return math.ceil(handle.getnframes() * 1000 / handle.getframerate())


def speech_intervals(
    path: Path,
    *,
    frame_ms: int = 20,
    min_speech_ms: int = 100,
    hangover_ms: int = 160,
) -> list[dict[str, int]]:
    """Return robust energy-based intervals in the WAV media clock."""
    audio = read_wav_mono(path, 16_000)
    frame = max(1, int(16_000 * frame_ms / 1000))
    rms = []
    for start in range(0, len(audio), frame):
        chunk = audio[start : start + frame]
        rms.append(float(np.sqrt(np.mean(np.square(chunk)))) if len(chunk) else 0.0)
    if not rms:
        return []
    values = np.asarray(rms, dtype=np.float32)
    noise = float(np.percentile(values, 20))
    high = float(np.percentile(values, 95))
    threshold = max(0.004, noise * 4.0, min(0.025, high * 0.12))
    voiced = values >= threshold
    hangover = max(0, hangover_ms // frame_ms)
    if hangover:
        expanded = voiced.copy()
        indices = np.flatnonzero(voiced)
        for index in indices:
            expanded[index : min(len(expanded), index + hangover + 1)] = True
        voiced = expanded
    intervals: list[dict[str, int]] = []
    start_index: int | None = None
    for index, active in enumerate([*voiced.tolist(), False]):
        if active and start_index is None:
            start_index = index
        elif not active and start_index is not None:
            start_ms = start_index * frame_ms
            end_ms = min(index * frame_ms, wav_duration_ms(path))
            if end_ms - start_ms >= min_speech_ms:
                intervals.append({"start_ms": start_ms, "end_ms": end_ms})
            start_index = None
    return intervals
