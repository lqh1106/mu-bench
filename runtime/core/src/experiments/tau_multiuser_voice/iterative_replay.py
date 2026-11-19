"""Model-conditioned Probe/Freeze/Final planning for μ-Bench.

The planner never mutates the benchmark oracle.  Each probe observes a fresh
SUT session containing only already-frozen user events.  The next user event is
then authored by that event's persistent user session from a causally available
assistant prefix, aligned to the benchmark media clock, and frozen before the
next probe starts.  After the declared case prefix, an event orchestrator may
request one bounded adaptive tail event based on an additional complete Probe.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Callable, Literal, Protocol

from pydantic import BaseModel, Field, ValidationError, model_validator

from experiments.tau_multiuser_voice.audio import TTSBackend
from experiments.tau_multiuser_voice.models import (
    MultiUserVoiceCase,
    ObservationKind,
    ScriptedEvent,
    TimedTranscriptFragment,
    TranscriptTurn,
    TriggerSpec,
    TriggerKind,
    UserAction,
    UserProposal,
    WorldObservation,
)


class ProbeAssistantFragment(BaseModel):
    """One task-relative committed assistant transcript fragment."""

    timestamp_ms: int = Field(ge=0)
    text: str


class ProbeAssistantInterval(BaseModel):
    """One actually played assistant speech interval in a probe."""

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_interval(self) -> ProbeAssistantInterval:
        if self.end_ms < self.start_ms:
            raise ValueError("assistant interval ends before it starts")
        return self


class ProbeToolObservation(BaseModel):
    """Task-relative tool call/result evidence exposed to event triggers."""

    timestamp_ms: int = Field(ge=0)
    kind: str
    tool_name: str
    result: str | None = None


class ProbeRunResult(BaseModel):
    """Evidence returned after a fresh probe has completely drained playback."""

    probe_index: int = Field(ge=0)
    frozen_event_ids: list[str]
    session_id: str | None = None
    assistant_fragments: list[ProbeAssistantFragment] = Field(default_factory=list)
    assistant_intervals: list[ProbeAssistantInterval] = Field(default_factory=list)
    tool_observations: list[ProbeToolObservation] = Field(default_factory=list)
    task_media_end_ms: int = Field(ge=0)
    response_done_seen: bool = False
    playback_drained: bool = False
    quiet_guard_ms: int = Field(default=0, ge=0)
    completion_reason: str
    artifact_dir: str | None = None

    @property
    def complete(self) -> bool:
        """Whether planning may safely consume this probe."""
        return self.response_done_seen and self.playback_drained

    def committed_prefix_at(self, timestamp_ms: int) -> str:
        """Return only assistant text committed no later than one media time."""
        return "".join(
            fragment.text
            for fragment in self.assistant_fragments
            if fragment.timestamp_ms <= timestamp_ms
        ).strip()


class FrozenUserEvent(BaseModel):
    """One immutable user event placed on the Final task-relative timeline."""

    event_id: str
    actor_id: str
    action: UserAction
    utterance: str
    addressee: str | None
    intent: str
    thread_id: str
    speech_act: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    planning_anchor_ms: int = Field(ge=0)
    requested_delay_ms: int = Field(default=0, ge=0)
    planner_target_start_ms: int | None = Field(default=None, ge=0)
    timing_adjustment_ms: int = 0
    planning_source: str
    assistant_prefix_used: str = ""
    probe_index: int | None = Field(default=None, ge=0)
    trigger_path: str = "causal"
    urgency: int = Field(default=3, ge=0, le=5)
    overlap_allowed: bool = False
    max_delay_ms: int = Field(default=3000, ge=0)
    required: bool = True
    adaptive_tail: bool = False

    @model_validator(mode="after")
    def validate_event(self) -> FrozenUserEvent:
        if self.end_ms <= self.start_ms:
            raise ValueError("frozen event must contain non-empty audio")
        return self


class ProbePlanRecord(BaseModel):
    """Auditable link between one probe and the next frozen event."""

    probe_index: int = Field(ge=0)
    input_event_ids: list[str]
    planned_event_id: str = ""
    observation_time_ms: int = Field(ge=0)
    observation_kind: ObservationKind
    assistant_prefix_used: str
    probe_complete_assistant_response: str = ""
    proposal: UserProposal | None = None
    frozen_start_ms: int = Field(ge=0)
    timing_adjustment_ms: int = 0
    completion_reason: str
    probe_artifact_dir: str | None = None
    probe_session_id: str | None = None
    phase: Literal["event", "tail_decision"] = "event"
    tail_decision: TailDecision | None = None


class TailDecision(BaseModel):
    """Central event-orchestrator decision after a complete final-prefix Probe."""

    action: Literal["end", "respond"] = "end"
    actor_id: str | None = None
    intent: str = ""
    addressee: str = "assistant"
    thread_id: str = "main"
    speech_act: str = "answer"
    urgency: int = Field(default=3, ge=0, le=5)
    overlap_allowed: bool = False
    max_delay_ms: int = Field(default=3000, ge=0)
    requested_delay_ms: int = Field(default=0, ge=0)
    reason: str = ""

    @model_validator(mode="after")
    def validate_decision(self) -> TailDecision:
        if self.action == "respond":
            if not self.actor_id:
                raise ValueError("tail response decision requires actor_id")
            if not self.intent.strip():
                raise ValueError("tail response decision requires intent")
        elif self.actor_id is not None:
            raise ValueError("end tail decision cannot select an actor")
        return self


class FrozenEventPlan(BaseModel):
    """Complete pre-rendering contract consumed by the Final session.

    The case events form the required prefix.  Optional ``tail_*`` events are
    appended only after the extra final-prefix Probe is judged by the event
    orchestrator.
    """

    case_id: str
    execution_protocol: str = "mu-bench-v1.3-spoken-self-introduction-adaptive-tail"
    tick_ms: int = Field(default=200, gt=0)
    event_alignment_ms: int = Field(default=80, gt=0)
    events: list[FrozenUserEvent]
    probes: list[ProbePlanRecord] = Field(default_factory=list)
    complete: bool = False
    tail_probe_complete: bool = False
    plan_sha256: str = ""

    @model_validator(mode="after")
    def validate_plan(self) -> FrozenEventPlan:
        event_ids = [event.event_id for event in self.events]
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("frozen event ids must be unique")
        if any(event.start_ms % self.event_alignment_ms for event in self.events):
            raise ValueError("frozen event start is not media-clock aligned")
        return self

    def with_checksum(self) -> FrozenEventPlan:
        """Return a copy containing a deterministic checksum of plan semantics."""
        payload = self.model_dump(mode="json", exclude={"plan_sha256"})
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return self.model_copy(update={"plan_sha256": digest})


class ProbeExecutor(Protocol):
    """Run a fresh model session containing exactly one frozen user prefix."""

    def run_probe(
        self,
        *,
        case: MultiUserVoiceCase,
        frozen_events: list[FrozenUserEvent],
        probe_index: int,
    ) -> ProbeRunResult:
        """Wait until the SUT response is done and playback has drained."""


class UserEventPlanner(Protocol):
    """One persistent, private planning session per benchmark participant."""

    def propose(
        self,
        observation: WorldObservation,
        event: ScriptedEvent | None = None,
    ) -> UserProposal:
        """Return the next causally grounded user proposal."""


class TailEventOrchestrator(Protocol):
    """Decide whether a completed final-prefix Probe needs a user reply."""

    def decide_tail(
        self,
        *,
        case: MultiUserVoiceCase,
        frozen_events: list[FrozenUserEvent],
        probe: ProbeRunResult,
        round_index: int,
    ) -> TailDecision:
        """Return ``end`` or a semantic contract for one adaptive tail event."""


class EndOfCaseTailOrchestrator:
    """Safe default for callers that do not provide a model-backed orchestrator."""

    def decide_tail(
        self,
        *,
        case: MultiUserVoiceCase,
        frozen_events: list[FrozenUserEvent],
        probe: ProbeRunResult,
        round_index: int,
    ) -> TailDecision:
        del case, frozen_events, probe, round_index
        return TailDecision(action="end", reason="no adaptive reply requested")


class ModelTailEventOrchestrator:
    """Use a JSON planner to make the one decision the case cannot pre-script.

    The planner receives only the public frozen timeline and the complete
    assistant response from the extra Probe.  It never receives undeclared
    future case events or private participant goals.
    """

    def __init__(self, backend: object) -> None:
        self.backend = backend
        # Keep a compact, JSON-serialisable audit trail of the tail planner's
        # raw candidate and any bounded schema repair.  The final decision is
        # still the only payload used to build a frozen event.
        self.audit_log: list[dict[str, object]] = []

    @staticmethod
    def _normalise_actor_id(
        payload: dict[str, object], case: MultiUserVoiceCase
    ) -> dict[str, object]:
        """Map an unambiguous planner alias to the canonical oracle user ID.

        The tail planner is a benchmark-side LLM and may copy the public
        participant ID or display name even though the event schema requires
        the canonical ``user_id``. Accept only exact, unique aliases; never
        guess from free-form prose or silently choose among duplicate names.
        """
        if payload.get("action") != "respond":
            return payload
        aliases: dict[str, set[str]] = {}
        for user in case.users:
            for value in (
                user.user_id,
                case.public_participant_id(user.user_id),
                user.display_name,
            ):
                key = str(value).strip().casefold()
                if key:
                    aliases.setdefault(key, set()).add(user.user_id)

        raw_values: list[object] = []
        for key in (
            "actor_id",
            "user_id",
            "user",
            "participant_id",
            "actor",
            "speaker",
            "selected_user",
            "selected_participant",
            "participant_name",
            "name",
            "requester",
        ):
            if key in payload:
                raw_values.append(payload[key])
        participant = payload.get("participant")
        if isinstance(participant, dict):
            for key in ("user_id", "participant_id", "display_name", "name"):
                if key in participant:
                    raw_values.append(participant[key])

        resolved: set[str] = set()
        for raw_value in raw_values:
            if isinstance(raw_value, str):
                resolved.update(aliases.get(raw_value.strip().casefold(), set()))
        if len(resolved) != 1:
            return payload
        return {**payload, "actor_id": next(iter(resolved))}

    def _validate_or_repair(
        self,
        *,
        messages: list[dict[str, str]],
        payload: dict[str, object],
        case: MultiUserVoiceCase,
        round_index: int,
    ) -> TailDecision:
        raw_payload = dict(payload)
        normalized = self._normalise_actor_id(payload, case)
        audit_entry: dict[str, object] = {
            "round_index": round_index,
            "raw_payload": raw_payload,
            "normalized_payload": dict(normalized),
            "repair_attempted": False,
            "accepted": False,
        }
        try:
            decision = TailDecision.model_validate(normalized)
            audit_entry["accepted"] = True
            self.audit_log.append(audit_entry)
            return decision
        except ValidationError as first_error:
            audit_entry["initial_validation_error"] = str(first_error)
            # Give the planner one bounded formatting correction for the two
            # observed schema-only failure modes.  Do not coerce semantic
            # labels such as ``standard`` to an arbitrary numeric urgency:
            # the planner must choose an explicit value while preserving every
            # other event field.  Unknown or repeated schema errors remain hard
            # failures instead of being hidden behind model defaults.
            missing_actor = (
                normalized.get("action") == "respond"
                and not normalized.get("actor_id")
            )
            validation_errors = first_error.errors()
            invalid_urgency = any(
                tuple(error.get("loc", ())) == ("urgency",)
                for error in validation_errors
            )
            unrelated_errors = [
                error
                for error in validation_errors
                if tuple(error.get("loc", ())) != ("urgency",)
                and not (
                    missing_actor and tuple(error.get("loc", ())) == ()
                )
            ]
            if (not missing_actor and not invalid_urgency) or unrelated_errors:
                self.audit_log.append(audit_entry)
                raise

            repair_fields: set[str] = set()
            corrections: list[str] = []
            if missing_actor:
                repair_fields.add("actor_id")
                corrections.append(
                    "Set actor_id exactly to one canonical user_id from the "
                    "supplied list; do not use participant_id, display_name, "
                    "or a free-form name, and do not omit the field."
                )
            if invalid_urgency:
                repair_fields.add("urgency")
                corrections.append(
                    "Set urgency to a JSON integer from 0 through 5; do not "
                    "return a label such as low, standard, normal, or high."
                )
            allowed_ids = [user.user_id for user in case.users]
            audit_entry["repair_attempted"] = True
            audit_entry["repair_fields"] = sorted(repair_fields)
            repair_messages = [*messages]
            repair_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous JSON failed the declared schema. Return "
                        "only one corrected JSON object. "
                        "The rejected candidate was: "
                        f"{json.dumps(normalized, ensure_ascii=False)}. "
                        "Canonical user_id values are: "
                        f"{json.dumps(allowed_ids, ensure_ascii=False)}. "
                        + " ".join(corrections)
                        + " Correct only the named schema fields. Preserve the "
                        "action, selected actor when already present, intent, "
                        "addressee, thread_id, speech_act, overlap_allowed, "
                        "max_delay_ms, and requested_delay_ms exactly."
                    ),
                }
            )
            repaired = self.backend.generate_json(repair_messages)  # type: ignore[attr-defined]
            if not isinstance(repaired, dict):
                self.audit_log.append(audit_entry)
                raise ValueError(
                    "tail event orchestrator repair did not return an object"
                ) from first_error
            repaired = self._normalise_actor_id(repaired, case)
            audit_entry["repaired_payload"] = dict(repaired)

            # A schema retry must not become a second semantic planning turn.
            # Compare the event-bearing fields using their declared defaults,
            # allowing changes only to the fields that actually failed.
            event_fields = (
                "action",
                "actor_id",
                "intent",
                "addressee",
                "thread_id",
                "speech_act",
                "urgency",
                "overlap_allowed",
                "max_delay_ms",
                "requested_delay_ms",
            )

            def resolved_value(candidate: dict[str, object], field: str) -> object:
                if field in candidate:
                    return candidate[field]
                return TailDecision.model_fields[field].default

            changed_fields = [
                field
                for field in event_fields
                if field not in repair_fields
                and resolved_value(normalized, field)
                != resolved_value(repaired, field)
            ]
            if changed_fields:
                audit_entry["repair_changed_fields"] = changed_fields
                self.audit_log.append(audit_entry)
                raise ValueError(
                    "tail event orchestrator schema repair changed non-repair "
                    "event fields: " + ", ".join(changed_fields)
                ) from first_error
            try:
                decision = TailDecision.model_validate(repaired)
                audit_entry["accepted"] = True
                self.audit_log.append(audit_entry)
                return decision
            except ValidationError as second_error:
                audit_entry["repair_validation_error"] = str(second_error)
                self.audit_log.append(audit_entry)
                raise ValueError(
                    "tail event orchestrator returned an invalid decision after "
                    "one bounded schema repair"
                ) from second_error

    def decide_tail(
        self,
        *,
        case: MultiUserVoiceCase,
        frozen_events: list[FrozenUserEvent],
        probe: ProbeRunResult,
        round_index: int,
    ) -> TailDecision:
        participants = [
            {
                "user_id": user.user_id,
                "participant_id": case.public_participant_id(user.user_id),
                "display_name": user.display_name,
                "role": user.role,
            }
            for user in case.users
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the μ-Bench event orchestrator. Decide whether the "
                    "complete assistant response after the final frozen user "
                    "prefix ends the interaction, or whether one participant "
                    "must make one additional reply. Use only the supplied "
                    "public timeline and response. Return one JSON object. "
                    "Choose action=end when the assistant has completed its "
                    "answer and no user-specific confirmation or correction is "
                    "needed. Choose action=respond only when the response "
                    "explicitly leaves a user-facing question, confirmation, "
                    "or repair unresolved. Never invent a new task or private "
                    "fact. If responding, select a listed user and provide a "
                    "short semantic intent; a separate persistent user session "
                    "will author the wording. Return exactly one JSON object "
                    "with keys action, actor_id, intent, addressee, thread_id, "
                    "speech_act, urgency, overlap_allowed, max_delay_ms, "
                    "requested_delay_ms, and reason. For action=end, actor_id "
                    "must be null. For action=respond, actor_id is required "
                    "and must exactly equal one participant's canonical "
                    "user_id value, never a display name or participant_id. "
                    "Do not omit actor_id. urgency must be a JSON integer from "
                    "0 through 5, not a word such as low, standard, normal, or "
                    "high. overlap_allowed must be a JSON boolean. "
                    "max_delay_ms and requested_delay_ms must be non-negative "
                    "JSON integers."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "round_index": round_index,
                        "participants": participants,
                        "frozen_public_timeline": [
                            {
                                "event_id": event.event_id,
                                "participant_id": case.public_participant_id(
                                    event.actor_id
                                ),
                                "utterance": event.utterance,
                                "addressee": event.addressee,
                                "intent": event.intent,
                                "thread_id": event.thread_id,
                            }
                            for event in frozen_events
                        ],
                        "assistant_complete_response": probe.committed_prefix_at(
                            probe.task_media_end_ms
                        ),
                        "assistant_fragments": [
                            fragment.model_dump(mode="json")
                            for fragment in probe.assistant_fragments
                        ],
                        "assistant_intervals": [
                            interval.model_dump(mode="json")
                            for interval in probe.assistant_intervals
                        ],
                        "tool_observations": [
                            item.model_dump(mode="json")
                            for item in probe.tool_observations
                        ],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        payload = self.backend.generate_json(messages)  # type: ignore[attr-defined]
        if not isinstance(payload, dict):
            raise ValueError("tail event orchestrator did not return an object")
        return self._validate_or_repair(
            messages=messages,
            payload=payload,
            case=case,
            round_index=round_index,
        )


def validate_iterative_replay_case(case: MultiUserVoiceCase) -> list[str]:
    """Return structural incompatibilities with Probe/Freeze/Final execution."""
    errors: list[str] = []
    if not case.events:
        return ["case has no user events"]
    first = case.events[0]
    if first.trigger.kind != TriggerKind.CONVERSATION_START:
        errors.append("first event must be released at conversation_start")
    if not first.fixed_utterance:
        errors.append("opening event requires a deterministic seed utterance")
    declared: set[str] = set()
    for event in case.events:
        if not event.intent and not event.fixed_utterance:
            errors.append(f"{event.event_id} has no semantic planning anchor")
        dependency = event.trigger.event_id
        fallback_dependency = event.trigger.fallback_after_event_id
        for referenced in (dependency, fallback_dependency):
            if referenced is not None and referenced not in declared:
                errors.append(
                    f"{event.event_id} depends on non-earlier event {referenced}"
                )
        declared.add(event.event_id)
    return errors


class IterativeReplayCompiler:
    """Compile one case through N fresh probes followed by an immutable plan."""

    def __init__(
        self,
        *,
        probe_executor: ProbeExecutor,
        user_sessions: dict[str, UserEventPlanner],
        tts: TTSBackend,
        tick_ms: int = 200,
        event_alignment_ms: int = 80,
        checkpoint: Callable[[FrozenEventPlan], None] | None = None,
        tail_orchestrator: TailEventOrchestrator | None = None,
    ) -> None:
        self.probe_executor = probe_executor
        self.user_sessions = user_sessions
        self.tts = tts
        self.tick_ms = tick_ms
        self.event_alignment_ms = event_alignment_ms
        self.checkpoint = checkpoint
        # Adaptive tail planning is enabled by the benchmark runners, which
        # provide a concrete event orchestrator.  Library callers that do not
        # provide one retain the legacy prefix-only compiler contract.
        self.tail_orchestrator = tail_orchestrator

    def _checkpoint(
        self,
        *,
        case: MultiUserVoiceCase,
        frozen: list[FrozenUserEvent],
        records: list[ProbePlanRecord],
        tail_probe_complete: bool = False,
    ) -> FrozenEventPlan:
        plan = FrozenEventPlan(
            case_id=case.case_id,
            tick_ms=self.tick_ms,
            event_alignment_ms=self.event_alignment_ms,
            events=list(frozen),
            probes=list(records),
            complete=len(frozen) >= len(case.events) and tail_probe_complete,
            tail_probe_complete=tail_probe_complete,
        ).with_checksum()
        if self.checkpoint is not None:
            self.checkpoint(plan)
        return plan

    def _align(self, timestamp_ms: int) -> int:
        return (
            math.ceil(timestamp_ms / self.event_alignment_ms)
            * self.event_alignment_ms
        )

    def _duration_ms(self, utterance: str, voice_id: str) -> int:
        pcm = self.tts.synthesize(utterance, voice_id)
        samples = len(pcm) // 2
        return max(1, math.ceil(samples * 1000 / self.tts.sample_rate))

    @staticmethod
    def _last_frozen_end(frozen: list[FrozenUserEvent]) -> int:
        return max((event.end_ms for event in frozen), default=0)

    def _fallback_anchor(
        self, event: ScriptedEvent, frozen: list[FrozenUserEvent]
    ) -> int:
        by_id = {item.event_id: item for item in frozen}
        candidates = [event.trigger.fallback_at_ms or 0]
        if event.trigger.fallback_after_event_id in by_id:
            candidates.append(by_id[event.trigger.fallback_after_event_id].end_ms)
        if event.trigger.event_id in by_id:
            candidates.append(by_id[event.trigger.event_id].end_ms)
        candidates.append(self._last_frozen_end(frozen))
        return max(candidates)

    def _resolve_anchor(
        self,
        event: ScriptedEvent,
        frozen: list[FrozenUserEvent],
        probe: ProbeRunResult,
    ) -> tuple[int, ObservationKind, str, str | None, str | None]:
        trigger = event.trigger
        not_before = self._last_frozen_end(frozen)
        anchor: int | None = None
        observation_kind = ObservationKind.CLOCK
        tool_name: str | None = None
        tool_result: str | None = None

        if trigger.kind == TriggerKind.CONVERSATION_START:
            anchor = trigger.after_ms
            observation_kind = ObservationKind.CONVERSATION_START
        elif trigger.kind == TriggerKind.EVENT_COMPLETED:
            prior = next(
                (item for item in frozen if item.event_id == trigger.event_id), None
            )
            if prior is not None:
                anchor = prior.end_ms + trigger.after_ms
                observation_kind = ObservationKind.USER_SPEECH
        elif trigger.kind == TriggerKind.ELAPSED:
            anchor = trigger.after_ms
        elif trigger.kind == TriggerKind.ASSISTANT_SPEECH_START:
            for interval in probe.assistant_intervals:
                candidate = interval.start_ms + trigger.after_ms
                if interval.start_ms >= not_before and (
                    not event.overlap_allowed or candidate < interval.end_ms
                ):
                    anchor = candidate
                    observation_kind = ObservationKind.ASSISTANT_SPEECH_PROGRESS
                    break
        elif trigger.kind == TriggerKind.ASSISTANT_SPEECH_STOP:
            for interval in probe.assistant_intervals:
                if interval.end_ms >= not_before:
                    anchor = interval.end_ms + trigger.after_ms
                    observation_kind = ObservationKind.ASSISTANT_SPEECH_STOP
                    break
        elif trigger.kind == TriggerKind.ASSISTANT_PREFIX and trigger.pattern:
            accumulated = ""
            for fragment in probe.assistant_fragments:
                accumulated += fragment.text
                if (
                    fragment.timestamp_ms >= not_before
                    and trigger.pattern.casefold() in accumulated.casefold()
                ):
                    anchor = fragment.timestamp_ms + trigger.after_ms
                    observation_kind = ObservationKind.ASSISTANT_SPEECH_PROGRESS
                    break
        elif trigger.kind in {TriggerKind.TOOL_CALL, TriggerKind.TOOL_RESULT}:
            expected_kind = (
                "tool_call"
                if trigger.kind == TriggerKind.TOOL_CALL
                else "tool_result"
            )
            for observation in probe.tool_observations:
                if (
                    observation.timestamp_ms >= not_before
                    and observation.kind == expected_kind
                    and (
                        trigger.tool_name is None
                        or observation.tool_name == trigger.tool_name
                    )
                ):
                    anchor = observation.timestamp_ms + trigger.after_ms
                    observation_kind = (
                        ObservationKind.TOOL_CALL
                        if expected_kind == "tool_call"
                        else ObservationKind.TOOL_RESULT
                    )
                    tool_name = observation.tool_name
                    tool_result = observation.result
                    break

        trigger_path = "causal"
        if anchor is None:
            anchor = self._fallback_anchor(event, frozen)
            trigger_path = "fallback"
        return anchor, observation_kind, trigger_path, tool_name, tool_result

    def _world_observation(
        self,
        *,
        event: ScriptedEvent,
        frozen: list[FrozenUserEvent],
        probe: ProbeRunResult,
        anchor_ms: int,
        kind: ObservationKind,
        tool_name: str | None,
        tool_result: str | None,
    ) -> WorldObservation:
        prefix = probe.committed_prefix_at(anchor_ms)
        complete_response = probe.committed_prefix_at(probe.task_media_end_ms)
        public = [
            TranscriptTurn(
                speaker=item.actor_id,
                text=item.utterance,
                start_ms=item.start_ms,
                end_ms=item.end_ms,
            )
            for item in frozen
            if item.end_ms <= anchor_ms
        ]
        assistant_speaking = any(
            interval.start_ms <= anchor_ms < interval.end_ms
            for interval in probe.assistant_intervals
        )
        return WorldObservation(
            sim_time_ms=anchor_ms,
            kind=kind,
            assistant_speaking=assistant_speaking,
            assistant_committed_prefix=prefix,
            probe_complete_assistant_response=complete_response,
            probe_assistant_fragments=[
                TimedTranscriptFragment(
                    timestamp_ms=fragment.timestamp_ms,
                    text=fragment.text,
                )
                for fragment in probe.assistant_fragments
            ],
            probe_assistant_intervals_ms=[
                (interval.start_ms, interval.end_ms)
                for interval in probe.assistant_intervals
            ],
            probe_task_media_end_ms=probe.task_media_end_ms,
            public_transcript=public,
            completed_event_ids={
                item.event_id for item in frozen if item.end_ms <= anchor_ms
            },
            tool_name=tool_name,
            tool_result=tool_result,
            active_thread_ids={item.thread_id for item in frozen},
        )

    def _opening_event(
        self, case: MultiUserVoiceCase, event: ScriptedEvent
    ) -> FrozenUserEvent:
        utterance = event.fixed_utterance or event.intent
        profile = next(user for user in case.users if user.user_id == event.actor_id)
        start_ms = self._align(event.trigger.after_ms)
        duration_ms = self._duration_ms(utterance, profile.voice.voice_id)
        return FrozenUserEvent(
            event_id=event.event_id,
            actor_id=event.actor_id,
            action=event.action,
            utterance=utterance,
            addressee=event.addressee,
            intent=event.intent,
            thread_id=event.thread_id,
            speech_act=event.speech_act,
            start_ms=start_ms,
            end_ms=start_ms + duration_ms,
            planning_anchor_ms=start_ms,
            planning_source="case_opening_seed",
            trigger_path="causal",
        )

    def _adaptive_tail_event(
        self,
        *,
        event_id: str,
        previous_event_id: str,
        decision: TailDecision,
    ) -> ScriptedEvent:
        """Turn an orchestrator contract into a normal benchmark event."""
        return ScriptedEvent(
            event_id=event_id,
            actor_id=decision.actor_id or "",
            action=UserAction.SPEAK,
            trigger=TriggerSpec(
                kind=TriggerKind.EVENT_COMPLETED,
                event_id=previous_event_id,
            ),
            intent=decision.intent,
            addressee=decision.addressee,
            fixed_utterance=None,
            fixed_utterance_exact=False,
            required=False,
            urgency=decision.urgency,
            overlap_allowed=decision.overlap_allowed,
            max_delay_ms=decision.max_delay_ms,
            thread_id=decision.thread_id,
            speech_act=decision.speech_act,
            notes=["adaptive_tail_event_orchestrated_after_complete_probe"],
        )

    def compile(self, case: MultiUserVoiceCase) -> FrozenEventPlan:
        """Run every required probe and freeze all case events for Final."""
        errors = validate_iterative_replay_case(case)
        if errors:
            raise ValueError(
                f"case {case.case_id} is not iterative-replay compatible: "
                + "; ".join(errors)
            )
        expected_users = {user.user_id for user in case.users}
        if set(self.user_sessions) != expected_users:
            raise ValueError("user_sessions must contain every case participant")

        frozen = [self._opening_event(case, case.events[0])]
        records: list[ProbePlanRecord] = []
        probe_session_ids: set[str] = set()
        self._checkpoint(case=case, frozen=frozen, records=records)
        profiles = {user.user_id: user for user in case.users}
        for probe_index, event in enumerate(case.events[1:]):
            probe = self.probe_executor.run_probe(
                case=case,
                frozen_events=list(frozen),
                probe_index=probe_index,
            )
            if probe.frozen_event_ids != [item.event_id for item in frozen]:
                raise RuntimeError("probe did not replay the exact frozen prefix")
            if not probe.complete:
                raise RuntimeError(
                    f"probe {probe_index} incomplete: {probe.completion_reason}"
                )
            if case.iterative_replay.require_fresh_session_per_probe:
                if not probe.session_id:
                    raise RuntimeError(
                        f"probe {probe_index} did not expose a session id"
                    )
                if probe.session_id in probe_session_ids:
                    raise RuntimeError(
                        f"probe {probe_index} reused session {probe.session_id}"
                    )
                probe_session_ids.add(probe.session_id)
            anchor, kind, trigger_path, tool_name, tool_result = self._resolve_anchor(
                event, frozen, probe
            )
            observation = self._world_observation(
                event=event,
                frozen=frozen,
                probe=probe,
                anchor_ms=anchor,
                kind=kind,
                tool_name=tool_name,
                tool_result=tool_result,
            )
            if event.fixed_utterance_exact:
                proposal = UserProposal(
                    proposal_id=f"event:{event.event_id}",
                    actor_id=event.actor_id,
                    action=event.action,
                    utterance=event.fixed_utterance or event.intent,
                    addressee=event.addressee,
                    urgency=event.urgency,
                    intent=event.intent,
                    overlap_allowed=event.overlap_allowed,
                    requested_delay_ms=0,
                    max_wait_ms=event.max_delay_ms,
                    triggered_by_event_id=event.event_id,
                    required_event=event.required,
                    thread_id=event.thread_id,
                    speech_act=event.speech_act,
                )
                planning_source = "fixed_exact_case_line"
            else:
                proposal = self.user_sessions[event.actor_id].propose(
                    observation, event
                )
                if not proposal.utterance:
                    raise RuntimeError(
                        f"Probe user simulator authored no dialogue for "
                        f"{event.event_id}"
                    )
                planning_source = "persistent_user_llm"
            if proposal.actor_id != event.actor_id:
                raise RuntimeError(f"planner changed speaker for {event.event_id}")
            if proposal.action != event.action:
                raise RuntimeError(f"planner changed action for {event.event_id}")
            if proposal.addressee != event.addressee:
                raise RuntimeError(f"planner changed addressee for {event.event_id}")
            if proposal.thread_id != event.thread_id:
                raise RuntimeError(f"planner changed thread for {event.event_id}")
            if (
                proposal.target_start_ms is None
                and proposal.requested_delay_ms > event.max_delay_ms
            ):
                raise RuntimeError(f"planner exceeded max delay for {event.event_id}")

            timing_adjustment_ms = 0
            if proposal.target_start_ms is not None:
                lower = self._align(anchor)
                upper_raw = min(
                    anchor + event.max_delay_ms,
                    probe.task_media_end_ms,
                )
                upper = (
                    upper_raw // self.event_alignment_ms
                ) * self.event_alignment_ms
                if upper < lower:
                    raise RuntimeError(
                        f"no aligned insertion point exists in the event window for "
                        f"{event.event_id}"
                    )
                nearest = round(
                    proposal.target_start_ms / self.event_alignment_ms
                ) * self.event_alignment_ms
                start_ms = min(max(nearest, lower), upper)
                timing_adjustment_ms = start_ms - proposal.target_start_ms
                planning_source += "_absolute_probe_timing"
                if timing_adjustment_ms:
                    planning_source += "_scheduler_aligned"
            else:
                start_ms = self._align(anchor + proposal.requested_delay_ms)
            profile = profiles[event.actor_id]
            duration_ms = self._duration_ms(
                proposal.utterance or "", profile.voice.voice_id
            )
            frozen_event = FrozenUserEvent(
                event_id=event.event_id,
                actor_id=event.actor_id,
                action=event.action,
                utterance=proposal.utterance or "",
                addressee=event.addressee,
                intent=event.intent,
                thread_id=event.thread_id,
                speech_act=event.speech_act,
                start_ms=start_ms,
                end_ms=start_ms + duration_ms,
                planning_anchor_ms=anchor,
                requested_delay_ms=max(0, start_ms - anchor),
                planner_target_start_ms=proposal.target_start_ms,
                timing_adjustment_ms=timing_adjustment_ms,
                planning_source=planning_source,
                assistant_prefix_used=probe.committed_prefix_at(start_ms),
                probe_index=probe_index,
                trigger_path=trigger_path,
            )
            records.append(
                ProbePlanRecord(
                    probe_index=probe_index,
                    input_event_ids=[item.event_id for item in frozen],
                    planned_event_id=event.event_id,
                    observation_time_ms=anchor,
                    observation_kind=kind,
                    assistant_prefix_used=probe.committed_prefix_at(start_ms),
                    probe_complete_assistant_response=(
                        observation.probe_complete_assistant_response
                    ),
                    proposal=proposal,
                    frozen_start_ms=start_ms,
                    timing_adjustment_ms=timing_adjustment_ms,
                    completion_reason=probe.completion_reason,
                    probe_artifact_dir=probe.artifact_dir,
                    probe_session_id=probe.session_id,
                )
            )
            frozen.append(frozen_event)
            self._checkpoint(case=case, frozen=frozen, records=records)

        if (
            not case.iterative_replay.tail_probe_enabled
            or self.tail_orchestrator is None
        ):
            return self._checkpoint(
                case=case,
                frozen=frozen,
                records=records,
                tail_probe_complete=True,
            )

        # The case declaration controls the required prefix only.  The final
        # prefix Probe is deliberately repeated here so an event orchestrator
        # can inspect the complete response and decide whether the interaction
        # is over or one participant still needs to answer.
        for tail_round in range(case.iterative_replay.tail_probe_max_rounds):
            probe_index = len(records)
            probe = self.probe_executor.run_probe(
                case=case,
                frozen_events=list(frozen),
                probe_index=probe_index,
            )
            if probe.frozen_event_ids != [item.event_id for item in frozen]:
                raise RuntimeError("tail probe did not replay the exact frozen prefix")
            if not probe.complete:
                raise RuntimeError(
                    f"tail probe {probe_index} incomplete: {probe.completion_reason}"
                )
            if case.iterative_replay.require_fresh_session_per_probe:
                if not probe.session_id:
                    raise RuntimeError(
                        f"tail probe {probe_index} did not expose a session id"
                    )
                if probe.session_id in probe_session_ids:
                    raise RuntimeError(
                        f"tail probe {probe_index} reused session {probe.session_id}"
                    )
                probe_session_ids.add(probe.session_id)

            decision = self.tail_orchestrator.decide_tail(
                case=case,
                frozen_events=list(frozen),
                probe=probe,
                round_index=tail_round,
            )
            if decision.action == "end":
                records.append(
                    ProbePlanRecord(
                        probe_index=probe_index,
                        input_event_ids=[item.event_id for item in frozen],
                        observation_time_ms=probe.task_media_end_ms,
                        observation_kind=ObservationKind.ASSISTANT_SPEECH_STOP,
                        assistant_prefix_used=probe.committed_prefix_at(
                            probe.task_media_end_ms
                        ),
                        probe_complete_assistant_response=probe.committed_prefix_at(
                            probe.task_media_end_ms
                        ),
                        frozen_start_ms=probe.task_media_end_ms,
                        completion_reason=probe.completion_reason,
                        probe_artifact_dir=probe.artifact_dir,
                        probe_session_id=probe.session_id,
                        phase="tail_decision",
                        tail_decision=decision,
                    )
                )
                return self._checkpoint(
                    case=case,
                    frozen=frozen,
                    records=records,
                    tail_probe_complete=True,
                )

            if decision.actor_id not in expected_users:
                raise RuntimeError(
                    f"tail orchestrator selected unknown participant: {decision.actor_id}"
                )
            previous_event_id = frozen[-1].event_id
            event = self._adaptive_tail_event(
                event_id=f"tail_{tail_round:02d}",
                previous_event_id=previous_event_id,
                decision=decision,
            )
            anchor = max(
                probe.task_media_end_ms,
                self._last_frozen_end(frozen),
                max(
                    (interval.end_ms for interval in probe.assistant_intervals),
                    default=0,
                ),
            )
            observation = self._world_observation(
                event=event,
                frozen=frozen,
                probe=probe,
                anchor_ms=anchor,
                kind=ObservationKind.ASSISTANT_SPEECH_STOP,
                tool_name=None,
                tool_result=None,
            )
            proposal = self.user_sessions[event.actor_id].propose(observation, event)
            if not proposal.utterance:
                raise RuntimeError(
                    f"tail user simulator authored no dialogue for {event.event_id}"
                )
            if proposal.actor_id != event.actor_id or proposal.action != event.action:
                raise RuntimeError(f"tail planner changed speaker/action for {event.event_id}")
            if proposal.addressee != event.addressee:
                raise RuntimeError(f"tail planner changed addressee for {event.event_id}")
            if proposal.thread_id != event.thread_id:
                raise RuntimeError(f"tail planner changed thread for {event.event_id}")
            if (
                proposal.target_start_ms is None
                and proposal.requested_delay_ms > event.max_delay_ms
            ):
                raise RuntimeError(f"tail planner exceeded max delay for {event.event_id}")
            timing_adjustment_ms = 0
            planning_source = "persistent_user_llm_adaptive_tail"
            if proposal.target_start_ms is not None:
                # ``target_start_ms`` is an absolute task-media timestamp,
                # just like it is for ordinary events.  Do not silently
                # replace it with requested_delay_ms: doing so moves a tail
                # reply to the anchor (often one 80 ms tick later) and makes
                # the frozen Final track disagree with the planner's causal
                # timing decision.
                lower = self._align(anchor)
                upper = (
                    (anchor + event.max_delay_ms) // self.event_alignment_ms
                ) * self.event_alignment_ms
                if upper < lower:
                    raise RuntimeError(
                        f"no aligned insertion point exists in the event window for "
                        f"{event.event_id}"
                    )
                nearest = round(
                    proposal.target_start_ms / self.event_alignment_ms
                ) * self.event_alignment_ms
                start_ms = min(max(nearest, lower), upper)
                timing_adjustment_ms = start_ms - proposal.target_start_ms
                planning_source += "_absolute_probe_timing"
                if timing_adjustment_ms:
                    planning_source += "_scheduler_aligned"
            else:
                start_ms = self._align(anchor + proposal.requested_delay_ms)
            profile = profiles[event.actor_id]
            duration_ms = self._duration_ms(
                proposal.utterance, profile.voice.voice_id
            )
            frozen_event = FrozenUserEvent(
                event_id=event.event_id,
                actor_id=event.actor_id,
                action=event.action,
                utterance=proposal.utterance,
                addressee=event.addressee,
                intent=event.intent,
                thread_id=event.thread_id,
                speech_act=event.speech_act,
                start_ms=start_ms,
                end_ms=start_ms + duration_ms,
                planning_anchor_ms=anchor,
                requested_delay_ms=max(0, start_ms - anchor),
                planner_target_start_ms=proposal.target_start_ms,
                timing_adjustment_ms=timing_adjustment_ms,
                planning_source=planning_source,
                assistant_prefix_used=probe.committed_prefix_at(anchor),
                probe_index=probe_index,
                trigger_path="causal",
                urgency=event.urgency,
                overlap_allowed=event.overlap_allowed,
                max_delay_ms=event.max_delay_ms,
                required=event.required,
                adaptive_tail=True,
            )
            records.append(
                ProbePlanRecord(
                    probe_index=probe_index,
                    input_event_ids=[item.event_id for item in frozen],
                    planned_event_id=event.event_id,
                    observation_time_ms=anchor,
                    observation_kind=ObservationKind.ASSISTANT_SPEECH_STOP,
                    assistant_prefix_used=probe.committed_prefix_at(start_ms),
                    probe_complete_assistant_response=probe.committed_prefix_at(
                        probe.task_media_end_ms
                    ),
                    proposal=proposal,
                    frozen_start_ms=start_ms,
                    completion_reason=probe.completion_reason,
                    probe_artifact_dir=probe.artifact_dir,
                    probe_session_id=probe.session_id,
                    phase="tail_decision",
                    tail_decision=decision,
                )
            )
            frozen.append(frozen_event)
            self._checkpoint(case=case, frozen=frozen, records=records)

        raise RuntimeError(
            "adaptive tail probe budget exhausted without an end decision"
        )
