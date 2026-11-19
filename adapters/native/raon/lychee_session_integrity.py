"""Fail-closed audits of native Lychee input consumption and transport drain.

Arrival wall time is deliberately NOT converted into benchmark media time here.
The official async PCM callback tags the current emission round, not necessarily
the source audio-token round. Complete input consumption is a separate contract
from a verified output media clock.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
from typing import Any, Mapping, Sequence


class LycheeSessionIntegrityError(RuntimeError):
    def __init__(self, diagnostics: Mapping[str, Any]):
        self.diagnostics = dict(diagnostics)
        super().__init__("Lychee infrastructure-invalid session: " + "; ".join(diagnostics["issues"]))


def lychee_drain_timeout_seconds(
    input_samples: int, sample_rate: int, configured: str | float | None = None
) -> float:
    """Bound post-stop drain by input duration, optionally using an explicit cap."""
    if input_samples < 0 or sample_rate <= 0:
        raise ValueError("invalid Lychee input sample count/rate")
    seconds = float(configured) if configured is not None else max(180.0, input_samples / sample_rate * 8.0)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("LYCHEEFD_BENCH_DRAIN_TIMEOUT_SECONDS must be positive and finite")
    return seconds


def inspect_lychee_session_completion(
    events: Sequence[Mapping[str, Any]], *, input_samples: int, sample_rate: int,
    receiver_finished: bool, receive_errors: Sequence[str] = (),
) -> dict[str, Any]:
    if input_samples < 0 or sample_rate <= 0:
        raise ValueError("invalid Lychee input sample count/rate")
    issues = []
    rounds: dict[int, dict[str, Any]] = {}
    done_indices = []
    stage_indices = []
    native_errors = []
    total_ms = Decimal(0)

    def number(value: Any) -> Decimal:
        if value is None or isinstance(value, bool):
            raise ValueError("not a finite nonnegative number")
        try:
            result = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError("not a finite nonnegative number") from exc
        if not result.is_finite() or result < 0:
            raise ValueError("not a finite nonnegative number")
        return result

    for index, event in enumerate(events):
        kind = event.get("type") or event.get("event_type")
        if kind == "done":
            done_indices.append(index)
        if kind == "error" or event.get("frame_error"):
            native_errors.append(str(event.get("error") or event.get("frame_error") or event))
        if kind != "stage_timing":
            continue
        stage_indices.append(index)
        round_id = event.get("round_id")
        if not isinstance(round_id, int) or isinstance(round_id, bool) or round_id <= 0:
            issues.append("invalid_stage_round_id")
            continue
        if round_id in rounds:
            issues.append(f"duplicate_stage_round:{round_id}")
            continue
        queue = event.get("queue")
        if not isinstance(queue, Mapping):
            issues.append(f"missing_stage_queue:{round_id}")
            continue
        try:
            consumed = number(queue.get("consumed_ms"))
            pending = number(queue.get("pending_after_ms"))
            window = number(queue.get("infer_window_ms"))
        except ValueError:
            issues.append(f"invalid_stage_consumption:{round_id}")
            continue
        if window != Decimal(400) or consumed <= 0 or consumed > window:
            issues.append(f"invalid_native_400ms_quantum:{round_id}")
        rounds[round_id] = {"consumed_ms": float(consumed), "pending_after_ms": float(pending)}
        total_ms += consumed

    expected_ms = Decimal(input_samples) * 1000 / Decimal(sample_rate)
    # Official stage timing rounds milliseconds to 0.1 ms. At 16 kHz its
    # final partial quantum can differ from exact PCM duration by < one sample.
    tolerance_ms = max(Decimal(1000) / Decimal(sample_rate), Decimal("0.05"))
    if not receiver_finished:
        issues.append("receiver_not_finished_before_drain_deadline")
    if receive_errors:
        issues.append("receiver_error")
    if native_errors:
        issues.append("native_error_event")
    if len(done_indices) != 1:
        issues.append(f"expected_one_native_done:got_{len(done_indices)}")
    if rounds and sorted(rounds) != list(range(1, len(rounds) + 1)):
        issues.append("noncontiguous_consumption_rounds")
    if rounds and list(rounds) != sorted(rounds):
        issues.append("out_of_order_consumption_rounds")
    if input_samples and not rounds:
        issues.append("no_consumption_evidence")
    if abs(total_ms - expected_ms) > tolerance_ms:
        issues.append(f"input_consumption_mismatch:expected_ms={expected_ms}:consumed_ms={total_ms}")
    last_pending_ms = rounds[max(rounds)]["pending_after_ms"] if rounds else None
    if last_pending_ms is not None and Decimal(str(last_pending_ms)) > tolerance_ms:
        issues.append(f"native_input_still_pending_ms:{last_pending_ms}")
    if done_indices and stage_indices and done_indices[0] < max(stage_indices):
        issues.append("native_done_precedes_final_consumption")
    return {
        "schema_version": "lychee-native-completion-audit-v1",
        "input_samples": input_samples, "sample_rate": sample_rate,
        "input_duration_ms": float(expected_ms), "consumed_input_ms": float(total_ms),
        "consumption_round_count": len(rounds), "last_pending_ms": last_pending_ms,
        "rounds": rounds, "native_done_count": len(done_indices),
        "receiver_finished": receiver_finished, "receive_errors": list(receive_errors),
        "native_errors": native_errors, "issues": issues,
        "input_and_transport_complete": not issues,
        "safe_to_submit_next_native_session": not issues,
        "media_clock_verified": False,
        "eligible_for_probe_planning": False,
        "timing_limitation": "native async PCM round_id is emission round; source frame timing is not verified",
    }


def require_lychee_completion(diagnostics: Mapping[str, Any]) -> None:
    if not diagnostics["input_and_transport_complete"]:
        raise LycheeSessionIntegrityError(diagnostics)


def require_verified_lychee_media_clock() -> None:
    """Do not admit a drained but still wall-timestamped track to a planner."""
    raise LycheeSessionIntegrityError({"issues": [
        "media_clock_unverified: arrival wall time and async PCM emission round cannot be used as source media time"
    ], "media_clock_verified": False, "eligible_for_probe_planning": False})
