"""Causal execution of benchmark-critical scripted events."""

from __future__ import annotations

from experiments.tau_multiuser_voice.event_actor_contract import validate_user_event_actor
from experiments.tau_multiuser_voice.models import (
    MultiUserVoiceCase,
    ObservationKind,
    ScriptedEvent,
    TriggerKind,
    UserProposal,
    WorldObservation,
)


class EventGraphExecutor:
    """Release only events whose trigger is observable at the current time."""

    def __init__(self, case: MultiUserVoiceCase) -> None:
        # model_copy and dynamically supplied cases can bypass Pydantic's
        # original actor check. The user event graph never owns assistant TTS.
        for event in case.events:
            validate_user_event_actor(event.actor_id, case=case, event_id=event.event_id)
        self.case = case
        self._events: list[ScriptedEvent] = list(case.events)
        self._event_by_id: dict[str, ScriptedEvent] = {
            event.event_id: event for event in self._events
        }
        self._completed_at_ms: dict[str, int] = {}
        self._triggered_ids: set[str] = set()
        self._scheduled_ids: set[str] = set()
        self._condition_seen_at_ms: dict[str, int] = {}
        self._condition_seen_kind: dict[str, ObservationKind] = {}
        self._trigger_paths: dict[str, str] = {}
        self._trigger_observation_kinds: dict[str, ObservationKind] = {}

    @property
    def completed_event_ids(self) -> set[str]:
        return set(self._completed_at_ms)

    def mark_completed(self, event_id: str, sim_time_ms: int) -> None:
        """Record the first time an event became committed on the user track."""
        if event_id not in self._event_by_id:
            raise ValueError(f"unknown event id: {event_id}")
        self._completed_at_ms.setdefault(event_id, sim_time_ms)

    def register_event(self, event: ScriptedEvent) -> None:
        """Register an evaluator-created event for a frozen replay."""
        # Adaptive closure events do not pass through MultiUserVoiceCase's
        # validator. Reject invalid actors before mutating graph state.
        validate_user_event_actor(event.actor_id, case=self.case, event_id=event.event_id)
        existing = self._event_by_id.get(event.event_id)
        if existing is not None:
            if existing.model_dump(mode="json") != event.model_dump(mode="json"):
                raise ValueError(
                    f"event id already registered with different data: {event.event_id}"
                )
            return
        self._events.append(event)
        self._event_by_id[event.event_id] = event

    def mark_scheduled(self, event_id: str) -> None:
        """Latch an event while its audio waits to become committed."""
        self._scheduled_ids.add(event_id)

    def trigger_path(self, event_id: str) -> str | None:
        """Return whether the event was released by its causal trigger or fallback."""
        return self._trigger_paths.get(event_id)

    def trigger_observation_kind(self, event_id: str) -> ObservationKind | None:
        """Return the observation that first satisfied the causal condition."""
        return self._trigger_observation_kinds.get(event_id)

    def _matching_trigger_path(
        self, event: ScriptedEvent, observation: WorldObservation
    ) -> str | None:
        from .interaction_contract import result_can_trigger
        release = self.case.task_contract_revision.get('semantic_event_release', {}).get(event.event_id)
        if release:
            anchor = self._completed_at_ms.get(release['anchor_event_id'])
            if anchor is None: return None
            if observation.sim_time_ms >= anchor + release['timeout_ms']:
                return 'fallback'
            if (observation.kind == ObservationKind.TOOL_RESULT
                    and observation.tool_name in release.get('tool_result_names', [])):
                import json
                try: result = json.loads(observation.tool_result or 'null')
                except (TypeError, ValueError): result = None
                if isinstance(result, dict) and not result.get('error'):
                    self._condition_seen_kind[event.event_id] = ObservationKind.TOOL_RESULT
                    return 'causal'
        trigger = event.trigger
        kind = trigger.kind
        if event.event_id in self._condition_seen_at_ms and kind not in {
            TriggerKind.EVENT_COMPLETED,
            TriggerKind.ELAPSED,
        }:
            first_seen = self._condition_seen_at_ms[event.event_id]
            return (
                "causal"
                if observation.sim_time_ms >= first_seen + trigger.after_ms
                else None
            )
        if kind == TriggerKind.CONVERSATION_START:
            matched = observation.kind == ObservationKind.CONVERSATION_START
        elif kind == TriggerKind.ASSISTANT_SPEECH_START:
            matched = observation.kind == ObservationKind.ASSISTANT_SPEECH_START
        elif kind == TriggerKind.ASSISTANT_SPEECH_STOP:
            matched = observation.kind == ObservationKind.ASSISTANT_SPEECH_STOP
        elif kind == TriggerKind.ASSISTANT_PREFIX:
            matched = bool(
                trigger.pattern
                and trigger.pattern.casefold()
                in observation.assistant_committed_prefix.casefold()
            )
        elif kind == TriggerKind.EVENT_COMPLETED:
            if trigger.event_id not in self._completed_at_ms:
                return None
            base_time = self._completed_at_ms[trigger.event_id]
            matched = observation.sim_time_ms >= base_time + trigger.after_ms
            return "causal" if matched else None
        elif kind == TriggerKind.TOOL_CALL:
            matched = (
                observation.kind == ObservationKind.TOOL_CALL
                and observation.tool_name == trigger.tool_name
            )
        elif kind == TriggerKind.TOOL_RESULT:
            matched = (
                observation.kind == ObservationKind.TOOL_RESULT
                and observation.tool_name == trigger.tool_name
                and result_can_trigger(self.case, event, observation)
            )
        elif kind == TriggerKind.ELAPSED:
            matched = observation.sim_time_ms >= trigger.after_ms
            return "causal" if matched else None
        else:
            matched = False
        if matched:
            first_seen = self._condition_seen_at_ms.setdefault(
                event.event_id, observation.sim_time_ms
            )
            self._condition_seen_kind.setdefault(event.event_id, observation.kind)
            return (
                "causal"
                if observation.sim_time_ms >= first_seen + trigger.after_ms
                else None
            )
        if (
            trigger.fallback_at_ms is not None
            and observation.sim_time_ms >= trigger.fallback_at_ms
            and (
                trigger.fallback_after_event_id is None
                or trigger.fallback_after_event_id in self._completed_at_ms
            )
        ):
            return "fallback"
        return None

    def ready_events(self, observation: WorldObservation) -> list[ScriptedEvent]:
        """Return newly released events in case declaration order."""
        ready = []
        for event in self._events:
            if (
                event.event_id in self._completed_at_ms
                or event.event_id in self._scheduled_ids
            ):
                continue
            trigger_path = self._matching_trigger_path(event, observation)
            if trigger_path is not None:
                self._triggered_ids.add(event.event_id)
                self._trigger_paths.setdefault(event.event_id, trigger_path)
                self._trigger_observation_kinds.setdefault(
                    event.event_id,
                    self._condition_seen_kind.get(event.event_id, observation.kind),
                )
            if event.event_id in self._triggered_ids:
                ready.append(event)
        return ready

    @staticmethod
    def fixed_proposal(event: ScriptedEvent) -> UserProposal:
        """Convert an exact scripted event to a scheduler proposal."""
        if not event.fixed_utterance:
            raise ValueError("dynamic event requires a user session proposal")
        return UserProposal(
            proposal_id=f"event:{event.event_id}",
            actor_id=event.actor_id,
            action=event.action,
            utterance=event.fixed_utterance,
            addressee=event.addressee,
            urgency=event.urgency,
            intent=event.intent,
            overlap_allowed=event.overlap_allowed,
            max_wait_ms=event.max_delay_ms,
            triggered_by_event_id=event.event_id,
            required_event=event.required,
        )
