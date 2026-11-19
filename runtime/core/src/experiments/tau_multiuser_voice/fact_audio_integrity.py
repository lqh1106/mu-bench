"""Fail-closed duration checks for reviewed, frozen user audio.

The compiler and simulator both represent duration as ceil(samples * 1000 / Hz).
Only *start times* are frame-aligned; normal end-of-frame silence in a rendered
track must not be confused with permission to crop a newly synthesized utterance
by an 80/200 ms frame. Inspect the original duration, not the repaired duration.

This guard covers duration integrity, not ASR correctness or equal-duration
waveform substitutions. Exact waveform identity still requires an immutable TTS
cache (or PCM hashes). It tolerates at most one millisecond of metadata rounding,
not one sample, since the existing diagnostics do not retain sample counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real


def _number(row: Mapping[str, object], key: str, event_id: str) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"reviewed frozen audio {event_id}: missing/invalid {key}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"reviewed frozen audio {event_id}: non-finite {key}")
    return value


def validate_reviewed_audio_drift(
    diagnostics: Sequence[Mapping[str, object]],
    *,
    max_delta_ms: float = 1.0,
    expected_event_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate actual ``simulator.frozen_audio_duration_diagnostics``.

    Call immediately after constructing each reviewed Probe/Final simulator,
    before any SUT session or playback starts. Pass the current plan's event IDs
    to fail closed on missing/reordered diagnostics. An empty prefix is invalid.

    Both excessive cropping and excessive zero-padding are rejected even when
    ``accepted=True`` and repaired alignment is perfect. A caller cannot widen
    the rounding allowance beyond 1 ms; a full media frame is not rounding.
    Returns a small serializable audit record; raises ValueError on any failure.
    """
    if (
        isinstance(max_delta_ms, bool)
        or not isinstance(max_delta_ms, Real)
        or not math.isfinite(float(max_delta_ms))
        or not 0 <= max_delta_ms <= 1
    ):
        raise ValueError("max_delta_ms must be finite and between 0 and 1 millisecond")
    if (
        not isinstance(diagnostics, Sequence)
        or isinstance(diagnostics, (str, bytes))
        or not diagnostics
    ):
        raise ValueError("reviewed frozen audio requires non-empty duration diagnostics")

    event_ids: list[str] = []
    tolerated: list[dict[str, object]] = []
    max_original_drift = 0.0
    for row in diagnostics:
        if not isinstance(row, Mapping):
            raise ValueError("reviewed frozen audio diagnostic must be a mapping")
        event_id = row.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("reviewed frozen audio diagnostic lacks event_id")
        if event_id in event_ids:
            raise ValueError(f"reviewed frozen audio duplicate event_id: {event_id}")
        event_ids.append(event_id)
        speaker = row.get("speaker_id")
        if not isinstance(speaker, str) or not speaker.strip():
            raise ValueError(f"reviewed frozen audio {event_id}: missing speaker_id")

        planned = _number(row, "planned_duration_ms", event_id)
        rendered = _number(row, "rendered_duration_ms", event_id)
        aligned = _number(row, "aligned_duration_ms", event_id)
        original = _number(row, "original_drift_ms", event_id)
        residual = _number(row, "alignment_error_ms", event_id)
        start = _number(row, "start_ms", event_id)
        end = _number(row, "planned_end_ms", event_id)
        sample_rate = _number(row, "sample_rate", event_id)
        if min(planned, rendered, aligned, sample_rate) <= 0 or start < 0:
            raise ValueError(f"reviewed frozen audio {event_id}: non-positive duration/rate")
        if sample_rate != int(sample_rate):
            raise ValueError(f"reviewed frozen audio {event_id}: non-integral sample_rate")
        # These fields originate from integer millisecond durations, not from
        # noisy physical-clock measurements. Do not let inconsistent metadata
        # hide cropping by changing only original_drift_ms.
        if (
            original != rendered - planned
            or residual != aligned - planned
            or end - start != planned
        ):
            raise ValueError(f"reviewed frozen audio {event_id}: inconsistent duration diagnostics")
        if row.get("accepted") is not True:
            raise ValueError(f"reviewed frozen audio {event_id}: simulator rejected alignment")
        if not isinstance(row.get("duration_repaired"), bool):
            raise ValueError(f"reviewed frozen audio {event_id}: missing duration_repaired")
        if row["duration_repaired"] != (rendered != planned):
            raise ValueError(f"reviewed frozen audio {event_id}: inconsistent duration_repaired")
        if abs(original) > max_delta_ms:
            effect = "cropping" if original > 0 else "padding"
            raise ValueError(
                f"reviewed frozen audio {event_id}: substantive {effect} / TTS drift; "
                f"planned={planned:g}ms rendered={rendered:g}ms "
                f"original_drift={original:g}ms tolerance={max_delta_ms:g}ms"
            )
        if abs(residual) > max_delta_ms:
            raise ValueError(f"reviewed frozen audio {event_id}: residual alignment drift {residual:g}ms")
        max_original_drift = max(max_original_drift, abs(original))
        if original != 0 or residual != 0:
            tolerated.append({"event_id": event_id, "original_drift_ms": original,
                              "alignment_error_ms": residual})

    if expected_event_ids is not None:
        if isinstance(expected_event_ids, (str, bytes)) or list(expected_event_ids) != event_ids:
            raise ValueError("reviewed frozen audio diagnostics do not match frozen event order")
    return {
        "schema_version": "reviewed-frozen-audio-integrity-v1",
        "passed": True,
        "event_ids": event_ids,
        "event_count": len(event_ids),
        "max_delta_ms": float(max_delta_ms),
        "max_abs_original_drift_ms": max_original_drift,
        "tolerated_rounding_events": tolerated,
        "scope": "duration_integrity_not_semantic_or_waveform_identity",
    }
