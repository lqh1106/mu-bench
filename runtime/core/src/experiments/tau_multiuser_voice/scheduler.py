"""Deterministic shared-floor scheduling for multiple independent users."""

from __future__ import annotations

import hashlib
import math

from experiments.tau_multiuser_voice.models import (
    RejectedProposal,
    ScheduleDecision,
    ScheduledProposal,
    UserAction,
    UserProfile,
    UserProposal,
    UserRuntimeState,
)


class FloorScheduler:
    """Choose who reaches the single microphone without authoring utterances."""

    _SECURITY_ACTIONS = {
        UserAction.CORRECT,
        UserAction.CONFIRM,
        UserAction.DENY,
        UserAction.DELEGATE,
        UserAction.REVOKE,
    }

    def __init__(
        self,
        *,
        tick_ms: int = 200,
        max_simultaneous_speakers: int = 2,
        overlap_score_margin: float = 12.0,
        seed: int = 42,
    ) -> None:
        if tick_ms <= 0:
            raise ValueError("tick_ms must be positive")
        self.tick_ms = tick_ms
        self.max_simultaneous_speakers = max_simultaneous_speakers
        self.overlap_score_margin = overlap_score_margin
        self.seed = seed

    def _stable_jitter(self, proposal_id: str) -> float:
        payload = f"{self.seed}:{proposal_id}".encode()
        value = int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
        return value / (2**32) * 0.001

    def _score(
        self,
        proposal: UserProposal,
        profile: UserProfile,
        runtime: UserRuntimeState,
        sim_time_ms: int,
    ) -> float:
        score = float(proposal.urgency * 10)
        if proposal.required_event:
            score += 100
        if proposal.action in self._SECURITY_ACTIONS:
            score += 15
        if proposal.action == UserAction.INTERRUPT:
            score += 10 * profile.persona.interrupt_tendency
        if runtime.last_spoke_ms is not None:
            wait_ms = max(0, sim_time_ms - runtime.last_spoke_ms)
            score += min(wait_ms / 1000, 10)
            if wait_ms < 800:
                score -= 10
        if proposal.utterance and proposal.utterance in runtime.memory[-4:]:
            score -= 25
        if runtime.currently_speaking:
            score -= 30
        return score + self._stable_jitter(proposal.proposal_id)

    def _aligned_start(self, sim_time_ms: int, delay_ms: int) -> int:
        requested = sim_time_ms + delay_ms
        return math.ceil(requested / self.tick_ms) * self.tick_ms

    def schedule(
        self,
        *,
        proposals: list[UserProposal],
        profiles: dict[str, UserProfile],
        runtimes: dict[str, UserRuntimeState],
        sim_time_ms: int,
    ) -> ScheduleDecision:
        """Score proposals and select one or a controlled overlapping pair."""
        ranked: list[tuple[float, UserProposal]] = []
        rejected: list[RejectedProposal] = []
        seen_actors: set[str] = set()

        for proposal in proposals:
            if proposal.actor_id not in profiles or proposal.actor_id not in runtimes:
                rejected.append(
                    RejectedProposal(
                        proposal_id=proposal.proposal_id,
                        actor_id=proposal.actor_id,
                        reason="unknown_actor",
                    )
                )
                continue
            if proposal.actor_id in seen_actors:
                rejected.append(
                    RejectedProposal(
                        proposal_id=proposal.proposal_id,
                        actor_id=proposal.actor_id,
                        reason="one_proposal_per_actor_per_update",
                    )
                )
                continue
            seen_actors.add(proposal.actor_id)
            if not runtimes[proposal.actor_id].present and not proposal.required_event:
                rejected.append(
                    RejectedProposal(
                        proposal_id=proposal.proposal_id,
                        actor_id=proposal.actor_id,
                        reason="actor_not_present",
                    )
                )
                continue
            if proposal.action == UserAction.WAIT or not proposal.utterance:
                rejected.append(
                    RejectedProposal(
                        proposal_id=proposal.proposal_id,
                        actor_id=proposal.actor_id,
                        reason="wait_or_empty",
                    )
                )
                continue
            ranked.append(
                (
                    self._score(
                        proposal,
                        profiles[proposal.actor_id],
                        runtimes[proposal.actor_id],
                        sim_time_ms,
                    ),
                    proposal,
                )
            )

        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            return ScheduleDecision(
                sim_time_ms=sim_time_ms,
                selected=[],
                rejected=rejected,
            )

        top_score, top = ranked[0]
        selected = [
            ScheduledProposal(
                proposal=top,
                start_ms=self._aligned_start(sim_time_ms, top.requested_delay_ms),
                score=top_score,
                mode="exclusive",
            )
        ]

        for score, proposal in ranked[1:]:
            scripted_overlap = (
                top.required_event
                and proposal.required_event
                and top.overlap_allowed
                and proposal.overlap_allowed
            )
            can_overlap = (
                len(selected) < self.max_simultaneous_speakers
                and top.overlap_allowed
                and proposal.overlap_allowed
                and (scripted_overlap or top_score - score <= self.overlap_score_margin)
            )
            if can_overlap:
                selected[0].mode = "overlap"
                selected.append(
                    ScheduledProposal(
                        proposal=proposal,
                        start_ms=self._aligned_start(
                            sim_time_ms, proposal.requested_delay_ms
                        ),
                        score=score,
                        mode="overlap",
                    )
                )
            else:
                rejected.append(
                    RejectedProposal(
                        proposal_id=proposal.proposal_id,
                        actor_id=proposal.actor_id,
                        reason="floor_lost",
                    )
                )

        return ScheduleDecision(
            sim_time_ms=sim_time_ms,
            selected=selected,
            rejected=rejected,
        )
