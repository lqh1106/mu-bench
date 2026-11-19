"""V14-only, media-time user-event pacing; never edits a case or its score rules.

The controller sees committed playback, not generated/unplayed text or time spent
inside an LLM/TTS/network call. A blocked trigger is not a successful challenge.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any

PACING_VERSION = "mu-v14-media-pacing-v1"


class CausalWindowUnrealized(RuntimeError):
    """A bounded, uncompleted challenge, not a model score or infrastructure fault."""

    def __init__(self, event_id: str, reason: str, evidence: dict[str, Any]):
        self.event_id = event_id
        self.reason = reason
        self.evidence = dict(evidence)
        super().__init__(f"pacing causal window unrealized for {event_id}: {reason}")

    def as_dict(self) -> dict[str, Any]:
        return {"schema": "mu-causal-window-unrealized-v1",
                "classification": "challenge_unrealized", "event_id": self.event_id,
                "reason": self.reason, "evidence": self.evidence,
                "valid_model_result": False, "event_completed": False,
                "may_continue_next_case": True}

# Reviewed semantic exceptions, scoped to authored events across their eight
# difficulty variants. Urgency=5 alone is deliberately not an exemption.
URGENT_EVENT_EXEMPTIONS = {
    ("airline_03_payer_traveler_cancel", "e_live"):
        "The affected traveler objects before an unauthorized cancellation can be committed.",
    ("airline_03_payer_traveler_cancel", "e_live_followup"):
        "The traveler's unresolved cancellation veto must not be delayed by recovery padding.",
    ("telecom_03_router_restart_pushback", "e_live"):
        "An active emergency video call makes an immediate router restart disruptive.",
    ("telecom_03_router_restart_pushback", "e_live_followup"):
        "The emergency call is still active; the repeated no-restart warning stays immediate.",
    ("vehicle_03_warning_provenance", "e_live"):
        "A passenger reports an unclosed vehicle door; safety warning outranks answer padding.",
    ("vehicle_03_warning_provenance", "e_live_followup"):
        "The unresolved relayed door warning retains safety priority on the returning turn.",
    ("emergency_incident_01_evacuation_accessibility", "e_live"):
        "The affected occupant cannot use stairs; an unsafe evacuation assumption must be corrected promptly.",
}


def _value(value: Any) -> Any:
    return getattr(value, "value", value)


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class PacingConfig:
    response_opportunity_ms: int = 5000
    quiet_guard_ms: int = 1200
    no_response_timeout_ms: int = 12000
    consecutive_interrupt_recovery_ms: int = 4000
    initial_interrupt_prefix_ms: int = 1200
    followup_interrupt_prefix_ms: int = 1800
    causal_trigger_timeout_ms: int = 30000

    def __post_init__(self):
        if any(v < 0 for v in asdict(self).values()):
            raise ValueError("pacing durations must be nonnegative")
        if self.no_response_timeout_ms < self.response_opportunity_ms:
            raise ValueError("no-response timeout must include the response opportunity")


@dataclass(frozen=True)
class EventPacingRule:
    event_id: str
    mode: str
    protected_causal: bool
    require_current_overlap: bool
    urgent_exemption: bool
    effective_overlap_allowed: bool
    min_current_response_played_ms: int
    original_trigger: dict[str, Any]
    original_event_sha256: str
    reasons: tuple[str, ...]
    risk_flags: tuple[str, ...]
    timing_metrics: tuple[str, ...]
    capability_assertion_ids: tuple[str, ...]


@dataclass(frozen=True)
class CasePacingPlan:
    case_id: str
    source_case_sha256: str
    config: PacingConfig
    events: dict[str, EventPacingRule]
    version: str = PACING_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateDecision:
    allow: bool
    reason: str
    trigger_path: str | None = None
    terminal: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)


def _event_rule(case: Any, event: Any, config: PacingConfig) -> EventPacingRule:
    kind, action = _value(event.trigger.kind), _value(event.action)
    timing = tuple(r.metric for r in case.timing_requirements if r.event_id == event.event_id)
    capabilities = tuple(a.assertion_id for a in case.capability_assertions
                         if event.event_id in a.source_event_ids)
    scenario_id = getattr(getattr(case, "variant", None), "scenario_id", "")
    urgent_reason = URGENT_EVENT_EXEMPTIONS.get((scenario_id, event.event_id))
    urgent = action == "revoke" or event.speech_act == "cancel_or_revoke" or bool(urgent_reason)
    # Explicit starts and scored yield/selectivity are not replaceable by turns.
    # overlap_allowed alone is permission, not evidence that overlap is required.
    protected = (kind in {"assistant_speech_start", "assistant_prefix", "tool_call", "tool_result"}
                 or action in {"interrupt", "backchannel", "side_talk", "revoke"}
                 or any(m in {"yield_latency_ms", "false_interruptions", "overlap_realized_ms"}
                        for m in timing))
    initial = kind == "conversation_start"
    if initial:
        mode, protected, require_overlap = "opening", False, False
    elif protected:
        mode = "urgent_causal" if urgent else "protected_causal"
        require_overlap = kind in {"assistant_speech_start", "assistant_prefix"}
    else:
        mode, require_overlap = "response_opportunity", False
    minimum = 0
    if require_overlap and not urgent:
        minimum = (config.followup_interrupt_prefix_ms if event.event_id == "e_live_followup"
                   else config.initial_interrupt_prefix_ms)
    reasons = []
    risks = []
    if initial:
        reasons.append("Preserve the opening request without adding a pre-answer wait.")
    elif protected:
        reasons.append("Retain the authored causal interaction; a clock fallback cannot substitute for it.")
        if urgent:
            reasons.append("Explicit cancellation/revocation or a reviewed urgent safety/consent event bypasses recovery padding, not causal validation.")
            if urgent_reason:
                reasons.append(urgent_reason)
        else:
            reasons.append("Require played-prefix opportunity and recovery between separate critical interventions.")
        if kind in {"tool_call", "tool_result"}:
            reasons.append("Keep the tool-event cause; do not invent an additional audio-overlap condition.")
    else:
        reasons.append("Wait for a bounded response opportunity, drained output and stable silence before ordinary speech.")
        if event.overlap_allowed:
            reasons.append("Withdraw optional overlap for this unscored ordinary contribution; preserve words, actor, intent and ordering.")
        if kind == "assistant_speech_stop":
            reasons.append("A brief acoustic pause or expired absolute fallback is not completion of the previous answer.")
        if kind == "event_completed":
            reasons.append("A preceding user's end alone does not give the assistant time to answer.")
    if event.trigger.fallback_at_ms is not None:
        risks.append("absolute_fallback_can_be_stale")
    if kind == "event_completed" and event.trigger.after_ms < config.response_opportunity_ms:
        risks.append("short_user_to_user_gap")
    if kind == "assistant_speech_stop" and event.trigger.after_ms < config.quiet_guard_ms:
        risks.append("acoustic_pause_may_preempt_answer")
    if protected and event.trigger.fallback_at_ms is not None:
        risks.append("causal_challenge_must_not_fallback")
    if not protected and event.overlap_allowed:
        risks.append("optional_overlap_removed")
    if require_overlap and minimum > event.trigger.after_ms:
        risks.append("short_answers_may_not_supply_required_live_window")
    return EventPacingRule(event_id=event.event_id, mode=mode,
        protected_causal=protected, require_current_overlap=require_overlap,
        urgent_exemption=urgent,
        effective_overlap_allowed=bool(event.overlap_allowed and protected),
        min_current_response_played_ms=minimum,
        original_trigger=event.trigger.model_dump(mode="json"),
        original_event_sha256=_sha(event.model_dump(mode="json")),
        reasons=tuple(reasons), risk_flags=tuple(risks), timing_metrics=timing,
        capability_assertion_ids=capabilities)


def build_case_pacing(case: Any, config: PacingConfig | None = None) -> CasePacingPlan:
    config = config or PacingConfig()
    return CasePacingPlan(case.case_id, _sha(case.model_dump(mode="json")), config,
                         {e.event_id: _event_rule(case, e, config) for e in case.events})


class PacingController:
    """Deterministic gate, fed once per completed task-media tick.

    ``assistant_speaking`` describes audio actually played during that tick.
    ``response_in_flight`` includes generation, tools and unplayed output. When
    available, pass actual ``played_duration_ms`` to handle native chunk sizes.
    Repeated calls at the same logical timestamp add no media evidence.
    """

    def __init__(self, plan: CasePacingPlan):
        self.plan = plan
        self.rules = dict(plan.events)
        self.now_ms = 0
        self.assistant_speaking = False
        self.response_in_flight = False
        self.pending_audio_ms = 0
        self.played_intervals: list[tuple[int, int]] = []
        self._last_speech_end_ms: int | None = None
        self._settled_response_ends: list[int] = []
        self._first_eligible: dict[str, int] = {}
        self._completed: dict[str, int] = {}
        self._last_critical_end_ms: int | None = None

    def register_event(self, event: Any) -> None:
        """Adaptive closure turns are ordinary response-opportunity events."""
        if event.event_id in self.rules:
            raise ValueError("pacing event already registered")
        self.rules[event.event_id] = EventPacingRule(event.event_id, "response_opportunity",
            False, False, False, False, 0, event.trigger.model_dump(mode="json"),
            _sha(event.model_dump(mode="json")),
            ("Adaptive closure waits for the outstanding answer to drain.",), (), (), ())

    def mark_event_completed(self, event_id: str, end_ms: int) -> None:
        if event_id not in self.rules:
            raise KeyError(event_id)
        if event_id not in self._completed:
            self._completed[event_id] = end_ms
            if self.rules[event_id].protected_causal:
                self._last_critical_end_ms = end_ms

    def observe_media(self, task_time_ms: int, assistant_speaking: bool,
                      response_in_flight: bool = False, pending_audio_ms: int = 0,
                      played_duration_ms: int | None = None) -> None:
        if task_time_ms < self.now_ms:
            raise ValueError("logical media time cannot move backwards")
        delta = task_time_ms - self.now_ms
        if played_duration_ms is not None and not 0 <= played_duration_ms <= delta:
            raise ValueError("played duration must fit the newly advanced media window")
        if played_duration_ms and not assistant_speaking:
            raise ValueError("silent ticks cannot add assistant played-prefix duration")
        played = (delta if assistant_speaking else 0) if played_duration_ms is None else played_duration_ms
        if played:
            start, end = task_time_ms - played, task_time_ms
            if self.played_intervals and self.played_intervals[-1][1] == start:
                self.played_intervals[-1] = (self.played_intervals[-1][0], end)
            else:
                self.played_intervals.append((start, end))
            self._last_speech_end_ms = end
        self.now_ms = task_time_ms
        self.assistant_speaking = assistant_speaking
        self.response_in_flight = response_in_flight
        self.pending_audio_ms = max(0, pending_audio_ms)
        end = self._last_speech_end_ms
        if (end is not None and not self.busy and task_time_ms - end >= self.plan.config.quiet_guard_ms
                and (not self._settled_response_ends or self._settled_response_ends[-1] != end)):
            self._settled_response_ends.append(end)

    @property
    def busy(self) -> bool:
        return self.assistant_speaking or self.response_in_flight or self.pending_audio_ms > 0

    @property
    def current_speech_start_ms(self) -> int | None:
        """Actual start of the current committed voiced interval, never future audio."""
        if (self.assistant_speaking and self.played_intervals
                and self.played_intervals[-1][1] == self.now_ms):
            return self.played_intervals[-1][0]
        return None

    def _played_since(self, after_ms: int) -> int:
        return sum(max(0, end - max(start, after_ms)) for start, end in self.played_intervals)

    def gate(self, event_id: str, now_ms: int, previous_user_end_ms: int | None,
             raw_trigger_path: str | None) -> GateDecision:
        rule, cfg = self.rules[event_id], self.plan.config
        previous_end = previous_user_end_ms if previous_user_end_ms is not None else 0
        if now_ms < self.now_ms:
            raise ValueError("gate cannot consume future media observations")
        evidence = {"logical_now_ms": now_ms, "previous_user_end_ms": previous_end,
            "assistant_played_since_user_ms": self._played_since(previous_end),
            "assistant_speaking": self.assistant_speaking,
            "response_in_flight": self.response_in_flight,
            "pending_audio_ms": self.pending_audio_ms,
            "raw_trigger_path": raw_trigger_path, "mode": rule.mode}
        def decision(allow, reason, path=None, terminal=False):
            return GateDecision(allow, reason, path, terminal, evidence)
        if previous_end > now_ms:
            return decision(False, "previous_user_audio_not_completed")
        if rule.mode == "opening":
            return decision(raw_trigger_path == "causal", "opening_preserved", raw_trigger_path)
        first = self._first_eligible.setdefault(event_id, now_ms)
        if rule.protected_causal:
            intervals = [(max(start, previous_end), end) for start, end in self.played_intervals
                         if end > previous_end]
            evidence.update({"first_eligible_ms": first,
                "deadline_ms": first + cfg.causal_trigger_timeout_ms,
                "wait_elapsed_ms": now_ms - first,
                "required_played_prefix_ms": rule.min_current_response_played_ms,
                "authored_after_ms": rule.original_trigger.get("after_ms", 0),
                "current_speech_start_ms": self.current_speech_start_ms,
                "observed_speech_after_user": bool(intervals),
                "played_intervals_after_user_ms": intervals[-32:],
                "played_intervals_truncated": len(intervals) > 32})
            # Timeout is a visible unrealized challenge, never synthetic user audio.
            expired = now_ms - first >= cfg.causal_trigger_timeout_ms
            if raw_trigger_path != "causal":
                return decision(False, "causal_trigger_unrealized" if expired else "await_genuine_causal_trigger",
                                terminal=expired)
            if rule.require_current_overlap and not self.assistant_speaking:
                return decision(False, "live_overlap_window_unrealized" if expired else "await_current_played_speech",
                                terminal=expired)
            if not rule.urgent_exemption:
                if self._played_since(previous_end) < rule.min_current_response_played_ms:
                    return decision(False, "played_prefix_window_unrealized" if expired else "allow_current_reply_prefix",
                                    terminal=expired)
                critical_end = self._last_critical_end_ms
                if critical_end is not None:
                    recovered = self._played_since(critical_end)
                    full_reply = any(end > critical_end for end in self._settled_response_ends)
                    evidence.update({"recovery_played_ms": recovered,
                                     "settled_reply_since_last_intervention": full_reply})
                    if recovered < cfg.consecutive_interrupt_recovery_ms and not full_reply:
                        return decision(False, "recovery_window_unrealized" if expired else "allow_intervention_recovery",
                                        terminal=expired)
            return decision(True, "protected_causal_window_ready", "causal")
        if raw_trigger_path is None:
            return decision(False, "await_authored_trigger")
        elapsed = now_ms - previous_end
        last_speech = self._last_speech_end_ms
        quiet_since = max(previous_end, last_speech if last_speech is not None else previous_end)
        quiet_ms = now_ms - quiet_since
        evidence.update({"response_opportunity_elapsed_ms": elapsed, "quiet_ms": quiet_ms})
        if elapsed < cfg.response_opportunity_ms:
            return decision(False, "allow_response_opportunity")
        if self.busy:
            return decision(False, "wait_response_and_playback_drain")
        if quiet_ms < cfg.quiet_guard_ms:
            return decision(False, "wait_stable_end_of_answer")
        if not self._played_since(previous_end):
            if elapsed < cfg.no_response_timeout_ms:
                return decision(False, "wait_for_first_answer_or_explicit_no_response_timeout")
            return decision(True, "no_response_timeout", "fallback")
        return decision(True, "ordinary_reply_opportunity_complete", raw_trigger_path)
