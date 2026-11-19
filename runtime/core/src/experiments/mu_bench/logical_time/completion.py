"""Evidence-bound, bounded completion of one live logical-time conversation.

This module does not stop/cancel a model, judge task success, or infer provider
response boundaries. Callers keep sending ordinary silence ticks while it waits.
An API adapter supplies its response/tool/queue state; a continuous native codec
supplies its own output-drained/silence state without inventing ``response.done``.

All times use the same session clock. ``evidence_revision`` advances on newly
played user/model content or semantic protocol activity, NOT on silence ticks.
The latest voiced-audio/user boundaries are additionally part of the proof key,
so text-identical but newly played native audio cannot reuse a stale end decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CompletionConfig:
    quiet_guard_ms: int = 2000
    minimum_response_opportunity_ms: int = 5000
    post_decision_settle_ms: int = 2000
    # 0 = disabled (openweight default). Closed-source / Grok runners set
    # 10_000 so pending-script events can scoreable-end after assistant speech.
    post_assistant_user_silence_ms: int = 0
    bilateral_silence_limit_ms: int = 0
    max_logical_ms: int = 900_000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
            # post_assistant_user_silence_ms may be 0 to disable quiet auto-end.
            if name in {"post_assistant_user_silence_ms", "bilateral_silence_limit_ms"}:
                if value < 0:
                    raise ValueError(f"{name} must be a non-negative integer")
                continue
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CompletionSnapshot:
    logical_ms: int
    all_events_completed: bool
    latest_user_audio_end_ms: int
    user_audio_active: bool
    assistant_speaking: bool
    last_assistant_audio_end_ms: int
    output_drained: bool
    evidence_revision: int
    pending_response: bool = False
    pending_tools: bool = False
    pending_protocol: bool = False

    def __post_init__(self):
        for name in ("logical_ms", "latest_user_audio_end_ms",
                     "last_assistant_audio_end_ms", "evidence_revision"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    @property
    def proof_key(self):
        return (self.evidence_revision, self.latest_user_audio_end_ms,
                self.last_assistant_audio_end_ms)


@dataclass(frozen=True)
class CompletionAssessment:
    can_request_closure: bool
    completed: bool
    terminal_incomplete: bool
    reasons: tuple[str, ...]
    quiet_ms: int
    user_response_opportunity_ms: int
    post_decision_elapsed_ms: int | None

    def as_dict(self):
        return asdict(self)


class CompletionGate:
    """Require playback, response opportunity, idle state AND fresh semantic end.

    ``record_decision`` accepts end only when the caller explicitly supplies
    ``outstanding_question=False``. This is the closure planner's semantic
    assertion based on its supplied played prefix, not a string/punctuation
    heuristic and not a substitute for the benchmark's task evaluator.

    A decision on eligible snapshot S never completes immediately: later normal
    ticks must reveal no new activity over the post-decision settle interval.
    A budget deadline is terminal *incomplete*, never successful closure.
    """

    def __init__(self, config: CompletionConfig | None = None):
        self.config = config or CompletionConfig()
        self.audit_records: list[dict] = []
        self._latest_snapshot: CompletionSnapshot | None = None
        self._end_proof: tuple[int, int, int] | None = None
        self._end_decision_ms: int | None = None
        self._terminal_assessment: CompletionAssessment | None = None

    def _invalidate_end(self, snapshot, reason):
        if self._end_proof is not None:
            self.audit_records.append({"type": "closure_end_invalidated",
                "logical_ms": snapshot.logical_ms, "reason": reason,
                "old_proof_key": self._end_proof, "new_proof_key": snapshot.proof_key})
        self._end_proof = None
        self._end_decision_ms = None

    def evaluate(self, snapshot: CompletionSnapshot) -> CompletionAssessment:
        previous = self._latest_snapshot
        if previous is not None:
            if snapshot.logical_ms < previous.logical_ms:
                raise ValueError("completion clock cannot move backwards")
            if snapshot.evidence_revision < previous.evidence_revision:
                raise ValueError("played evidence revision cannot move backwards")
        self._latest_snapshot = snapshot
        if self._terminal_assessment is not None:
            return self._terminal_assessment

        quiet_ms = max(0, snapshot.logical_ms - snapshot.last_assistant_audio_end_ms)
        opportunity_ms = max(0, snapshot.logical_ms - snapshot.latest_user_audio_end_ms)
        reasons = []
        if not snapshot.all_events_completed:
            reasons.append("user_events_pending")
        if snapshot.user_audio_active or snapshot.latest_user_audio_end_ms > snapshot.logical_ms:
            reasons.append("user_audio_not_finished")
        if snapshot.assistant_speaking:
            reasons.append("assistant_still_speaking")
        if not snapshot.output_drained:
            reasons.append("assistant_output_not_drained")
        if snapshot.pending_response:
            reasons.append("model_response_pending")
        if snapshot.pending_tools:
            reasons.append("tool_activity_pending")
        if snapshot.pending_protocol:
            reasons.append("protocol_activity_pending")
        # Any live activity invalidates an earlier semantic decision even when
        # the response has not yet emitted new transcript/audio bytes.
        if self._end_proof is not None:
            if self._end_proof != snapshot.proof_key:
                self._invalidate_end(snapshot, "new_played_evidence")
            elif reasons:
                self._invalidate_end(snapshot, "new_activity")
        if opportunity_ms < self.config.minimum_response_opportunity_ms:
            reasons.append("minimum_user_response_opportunity_not_elapsed")
        if quiet_ms < self.config.quiet_guard_ms:
            reasons.append("assistant_quiet_guard_not_elapsed")

        elapsed = (None if self._end_decision_ms is None else
                   snapshot.logical_ms - self._end_decision_ms)
        # Scoreable end while scripted events are still pending: the assistant
        # has spoken, nobody is speaking, and 10s have passed since whichever
        # side spoke last. A user who talks after the assistant resets this.
        latest_speech_end = max(
            snapshot.last_assistant_audio_end_ms, snapshot.latest_user_audio_end_ms)
        quiet_since_latest = max(0, snapshot.logical_ms - latest_speech_end)
        floor_idle = (
            not snapshot.user_audio_active
            and snapshot.latest_user_audio_end_ms <= snapshot.logical_ms
            and not snapshot.assistant_speaking
            and snapshot.output_drained
            and not snapshot.pending_response
            and not snapshot.pending_tools
            and not snapshot.pending_protocol
        )
        if (self.config.bilateral_silence_limit_ms > 0
                and floor_idle
                and quiet_since_latest >= self.config.bilateral_silence_limit_ms):
            result = CompletionAssessment(False, True, False,
                ("bilateral_silence_limit_reached",), quiet_ms, opportunity_ms, elapsed)
            self._terminal_assessment = result
            self.audit_records.append({
                "type": "bilateral_silence_limit_end",
                "logical_ms": snapshot.logical_ms,
                "quiet_since_latest_ms": quiet_since_latest,
                "limit_ms": self.config.bilateral_silence_limit_ms,
                "all_events_completed": snapshot.all_events_completed,
                "assessment": result.as_dict()})
            return result
        if (self.config.post_assistant_user_silence_ms > 0
                and not snapshot.all_events_completed
                and snapshot.last_assistant_audio_end_ms > 0
                and floor_idle
                and quiet_since_latest >= self.config.post_assistant_user_silence_ms):
            result = CompletionAssessment(False, True, False,
                ("incomplete_events_quiet_after_assistant_speech",),
                quiet_ms, opportunity_ms, elapsed)
            self._terminal_assessment = result
            self.audit_records.append({
                "type": "incomplete_events_quiet_auto_end",
                "logical_ms": snapshot.logical_ms,
                "quiet_since_latest_ms": quiet_since_latest,
                "assessment": result.as_dict()})
            return result
        if snapshot.logical_ms >= self.config.max_logical_ms:
            result = CompletionAssessment(False, False, True,
                ("logical_budget_exhausted_before_confirmed_completion", *reasons),
                quiet_ms, opportunity_ms, elapsed)
            self._terminal_assessment = result
            self.audit_records.append({"type": "conversation_incomplete",
                "logical_ms": snapshot.logical_ms, "assessment": result.as_dict()})
            return result

        eligible = not reasons
        if self._end_proof is None:
            if eligible:
                reasons.append("fresh_semantic_closure_required")
            return CompletionAssessment(eligible, False, False, tuple(reasons),
                                        quiet_ms, opportunity_ms, elapsed)
        if eligible and elapsed >= self.config.post_decision_settle_ms:
            result = CompletionAssessment(False, True, False,
                ("played_conversation_completed",), quiet_ms, opportunity_ms, elapsed)
            self._terminal_assessment = result
            self.audit_records.append({"type": "conversation_completed",
                "logical_ms": snapshot.logical_ms, "proof_key": snapshot.proof_key,
                "assessment": result.as_dict()})
            return result
        if eligible:
            reasons.append("post_closure_observation_window_not_elapsed")
        return CompletionAssessment(False, False, False, tuple(reasons),
                                    quiet_ms, opportunity_ms, elapsed)

    def record_decision(self, snapshot: CompletionSnapshot, *, action: str,
                        outstanding_question: bool | None = None, reason: str = "") -> bool:
        """Record a decision from exactly the latest eligible played observation.

        Returns True only for an accepted provisional end. ``respond`` and an
        end lacking a definite no-outstanding-question assertion cannot finish.
        A stale or ineligible end is rejected without turning it into success.
        """
        if action not in ("end", "respond"):
            raise ValueError("closure action must be end or respond")
        if self._terminal_assessment is not None:
            raise RuntimeError("cannot change a terminal completion decision")
        if snapshot != self._latest_snapshot:
            raise ValueError("closure decision must use the latest evaluated snapshot")
        assessment = self.evaluate(snapshot)
        accepted = (action == "end" and outstanding_question is False
                    and assessment.can_request_closure)
        self._invalidate_end(snapshot, "new_closure_decision")
        if accepted:
            self._end_proof = snapshot.proof_key
            self._end_decision_ms = snapshot.logical_ms
        self.audit_records.append({"type": "closure_decision",
            "logical_ms": snapshot.logical_ms, "proof_key": snapshot.proof_key,
            "action": action, "outstanding_question": outstanding_question,
            "accepted_provisional_end": accepted, "reason": reason,
            "eligibility": assessment.as_dict()})
        return accepted
