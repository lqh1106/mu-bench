"""Capability negotiation and fair-track admission for μ-Bench."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from experiments.mu_bench.runtime.contracts import (
    BackendCapabilities,
    CausalInputIsolation,
    ExecutionMode,
)


TrackName = Literal[
    "batch_semantic",
    "core_no_tool",
    "attribution",
    "native_tool",
    "wall_clock",
    "full",
]


@dataclass(frozen=True)
class CapabilityDecision:
    track: TrackName
    eligible: bool
    missing: tuple[str, ...]


def required_capabilities(track: TrackName) -> tuple[str, ...]:
    if track == "batch_semantic":
        # This compatibility track deliberately accepts a complete input WAV.
        # It is not a realtime leaderboard and therefore has no causal input
        # or wall-clock requirement.
        return ("whole_track_audio_input",)
    if track == "core_no_tool":
        return ("incremental_audio_input", "incremental_audio_output")
    if track == "attribution":
        return ("incremental_audio_input", "incremental_audio_output", "explicit_speaker_attribution", "explicit_addressee_attribution")
    if track == "native_tool":
        return ("incremental_audio_input", "incremental_audio_output", "native_tool_calling")
    if track == "wall_clock":
        return ("incremental_audio_input", "incremental_audio_output", "supports_wall_clock_timing")
    return (
        "incremental_audio_input",
        "incremental_audio_output",
        "explicit_speaker_attribution",
        "explicit_addressee_attribution",
        "native_tool_calling",
        "supports_wall_clock_timing",
    )


def check_capabilities(
    capabilities: BackendCapabilities,
    track: TrackName,
) -> CapabilityDecision:
    missing = [
        name for name in required_capabilities(track) if not getattr(capabilities, name)
    ]

    realtime_tracks = {"core_no_tool", "attribution", "native_tool", "wall_clock", "full"}
    if track in realtime_tracks:
        if capabilities.execution_mode != ExecutionMode.CAUSAL_STREAMING:
            missing.append("execution_mode=causal_streaming")
        if capabilities.causal_input_isolation != CausalInputIsolation.ENFORCED:
            missing.append("causal_input_isolation=enforced")

    if track in {"wall_clock", "full"}:
        if not capabilities.committed_output_timing:
            missing.append("committed_output_timing")
        if not capabilities.mid_session_cancel:
            missing.append("mid_session_cancel")

    return CapabilityDecision(track=track, eligible=not missing, missing=tuple(missing))
