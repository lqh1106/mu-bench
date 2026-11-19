"""Independent persistent LLM sessions for simulated users."""

from __future__ import annotations

import json
from typing import Protocol

from experiments.tau_multiuser_voice.models import (
    MultiUserVoiceCase,
    ScriptedEvent,
    UserAction,
    UserProfile,
    UserProposal,
    UserRuntimeState,
    WorldObservation,
)


class ProposalBackend(Protocol):
    """Minimal adapter for any text LLM that can return a JSON object."""

    def generate_json(self, messages: list[dict[str, str]]) -> dict:
        """Generate a proposal object from one persistent session history."""


class PersistentUserSession:
    """One isolated conversation and private state per simulated person."""

    def __init__(
        self,
        *,
        case: MultiUserVoiceCase,
        profile: UserProfile,
        backend: ProposalBackend,
        timing_alignment_ms: int = 80,
    ) -> None:
        self.case = case
        self.profile = profile
        self.backend = backend
        self.timing_alignment_ms = timing_alignment_ms
        active_grants = {
            grant.grant_id for grant in profile.permissions if grant.active_initially
        }
        self.runtime = UserRuntimeState(
            user_id=profile.user_id,
            goal_progress={goal.goal_id: "pending" for goal in profile.goals},
            active_grant_ids=active_grants,
            present=profile.present_initially,
        )
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt()}
        ]
        self._proposal_index = 0

    def _system_prompt(self) -> str:
        private_payload = {
            "case_summary": self.case.summary,
            "your_profile": self.profile.model_dump(mode="json"),
            "rules": [
                "You control only this person.",
                "Never claim another user's private goals or private knowledge.",
                "In ordinary online mode, use only public committed conversation.",
                "In Probe/Freeze mode, the completed unscored probe response may be used retrospectively to author and place the next event; a separate Final session will test that frozen event.",
                "Return one JSON proposal and no prose.",
                "WAIT is valid when there is no natural reason to speak.",
            ],
        }
        return json.dumps(private_payload, ensure_ascii=False)

    def _proposal_prompt(
        self,
        observation: WorldObservation,
        event: ScriptedEvent | None,
    ) -> str:
        event_payload = event.model_dump(mode="json") if event else None
        schema = {
            "action": [action.value for action in UserAction],
            "utterance": "string or null",
            "addressee": "assistant, a user_id, or null",
            "urgency": "integer 0..5",
            "intent": "short string",
            "overlap_allowed": "boolean",
            "requested_delay_ms": (
                "non-negative integer, no greater than max_wait_ms, and a "
                f"multiple of {self.timing_alignment_ms} ms"
            ),
            "target_start_ms": (
                "absolute task-media timestamp for this event, as a non-negative "
                f"multiple of {self.timing_alignment_ms} ms; in Probe/Freeze mode "
                "choose it from the completed assistant transcript/timeline within "
                "the permitted window"
            ),
            "max_wait_ms": "non-negative integer",
            "thread_id": "existing or new thread identifier",
            "speech_act": "request, quote, correction, backchannel, answer, or statement",
        }
        payload = {
            "observation": observation.model_dump(mode="json"),
            "required_event": event_payload,
            "planning_rules": [
                "Preserve required_event.actor_id, action, addressee, intent, thread_id, and speech_act.",
                "required_event.intent and notes form an abstract event contract, not dialogue to repeat verbatim.",
                "Generate this user's natural next utterance from the complete Probe response, the user's persistent private state, and the event contract; make it connect coherently to what was actually said.",
                "Do not speak benchmark metadata such as event ids, thread ids, trigger names, oracle terms, or evaluator language.",
                "Do not introduce facts, permissions, tool results, or future assistant content that is absent from the case and completed Probe.",
                "For Probe/Freeze mode, target_start_ms must be between observation.sim_time_ms and observation.sim_time_ms + required_event.max_delay_ms, and must be aligned exactly.",
                "Use probe_assistant_fragments and probe_assistant_intervals_ms to choose a semantically and temporally appropriate insertion point; do not invent transcript content.",
            ],
            "output_schema": schema,
        }
        return json.dumps(payload, ensure_ascii=False)

    def propose(
        self,
        observation: WorldObservation,
        event: ScriptedEvent | None = None,
    ) -> UserProposal:
        """Generate and validate one causally grounded proposal."""
        if event and event.actor_id != self.profile.user_id:
            raise ValueError("event assigned to the wrong user session")
        if event and event.fixed_utterance and event.fixed_utterance_exact:
            return UserProposal(
                proposal_id=f"event:{event.event_id}",
                actor_id=self.profile.user_id,
                action=event.action,
                utterance=event.fixed_utterance,
                addressee=event.addressee,
                urgency=event.urgency,
                intent=event.intent,
                overlap_allowed=event.overlap_allowed,
                max_wait_ms=event.max_delay_ms,
                triggered_by_event_id=event.event_id,
                required_event=event.required,
                thread_id=event.thread_id,
                speech_act=event.speech_act,
            )

        prompt = self._proposal_prompt(observation, event)
        self.messages.append({"role": "user", "content": prompt})
        raw = self.backend.generate_json(list(self.messages))
        self.messages.append(
            {
                "role": "assistant",
                "content": json.dumps(raw, ensure_ascii=False),
            }
        )
        self._proposal_index += 1
        proposal = UserProposal(
            proposal_id=(f"{self.profile.user_id}:proposal:{self._proposal_index}"),
            actor_id=self.profile.user_id,
            action=raw.get("action", UserAction.WAIT),
            utterance=raw.get("utterance"),
            addressee=raw.get("addressee", "assistant"),
            urgency=raw.get("urgency", 0),
            intent=raw.get("intent", ""),
            overlap_allowed=raw.get("overlap_allowed", False),
            requested_delay_ms=raw.get("requested_delay_ms", 0),
            target_start_ms=raw.get("target_start_ms"),
            max_wait_ms=raw.get("max_wait_ms", 3000),
            triggered_by_event_id=event.event_id if event else None,
            required_event=event.required if event else False,
            thread_id=raw.get("thread_id", event.thread_id if event else "main"),
            speech_act=raw.get(
                "speech_act", event.speech_act if event else "statement"
            ),
        )
        if event:
            proposal.action = event.action
            proposal.addressee = event.addressee
            proposal.urgency = max(proposal.urgency, event.urgency)
            proposal.intent = event.intent
            proposal.overlap_allowed = event.overlap_allowed
            proposal.thread_id = event.thread_id
            proposal.speech_act = event.speech_act
        return proposal


class ScriptedUserSession:
    """Offline deterministic session used by tests and smoke runs."""

    def __init__(self, profile: UserProfile) -> None:
        self.profile = profile
        self.runtime = UserRuntimeState(
            user_id=profile.user_id,
            goal_progress={goal.goal_id: "pending" for goal in profile.goals},
            active_grant_ids={
                grant.grant_id
                for grant in profile.permissions
                if grant.active_initially
            },
            present=profile.present_initially,
        )
        self.messages: list[dict[str, str]] = []

    def propose(
        self,
        observation: WorldObservation,
        event: ScriptedEvent | None = None,
    ) -> UserProposal:
        """Emit a deterministic semantic fallback for tests and smoke runs."""
        del observation
        if event is None:
            return UserProposal(
                proposal_id=f"{self.profile.user_id}:wait",
                actor_id=self.profile.user_id,
                action=UserAction.WAIT,
            )
        return UserProposal(
            proposal_id=f"event:{event.event_id}",
            actor_id=self.profile.user_id,
            action=event.action,
            utterance=event.fixed_utterance or event.intent,
            addressee=event.addressee,
            urgency=event.urgency,
            intent=event.intent,
            overlap_allowed=event.overlap_allowed,
            max_wait_ms=event.max_delay_ms,
            triggered_by_event_id=event.event_id,
            required_event=event.required,
            thread_id=event.thread_id,
            speech_act=event.speech_act,
        )
