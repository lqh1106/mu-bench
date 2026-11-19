"""Event-driven multi-user simulator core for a single mixed microphone."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel

from experiments.tau_multiuser_voice.audio import (
    MixedAudioWindow,
    PCMTrackMixer,
    QueuedAudio,
    SpeakerAudioQueue,
    TTSBackend,
)
from experiments.tau_multiuser_voice.authorization import AuthorizationOracle
from experiments.tau_multiuser_voice.event_actor_contract import validate_user_event_actor
from experiments.tau_multiuser_voice.event_graph import EventGraphExecutor
from experiments.tau_multiuser_voice.models import (
    MultiUserEpisodeTrace,
    MultiUserVoiceCase,
    ScheduleDecision,
    ScriptedEvent,
    SpokenEventTrace,
    UserProfile,
    UserProposal,
    UserRuntimeState,
    WorldObservation,
    TriggerSpec,
    TriggerKind,
)
from experiments.tau_multiuser_voice.scheduler import FloorScheduler

if TYPE_CHECKING:
    from experiments.tau_multiuser_voice.iterative_replay import FrozenEventPlan


class UserSession(Protocol):
    """Common surface of scripted and LLM-backed persistent users."""

    profile: UserProfile
    runtime: UserRuntimeState

    def propose(
        self,
        observation: WorldObservation,
        event: ScriptedEvent | None = None,
    ) -> UserProposal:
        """Propose a single structured user action."""


class SimulatorTickResult(BaseModel):
    """Artifacts produced by one 200 ms policy update."""

    observation: WorldObservation
    proposals: list[UserProposal]
    schedule: ScheduleDecision
    audio: MixedAudioWindow


class MultiUserVoiceSimulator:
    """Coordinate isolated users, exact events, TTS queues, and mono mixing."""

    def __init__(
        self,
        *,
        case: MultiUserVoiceCase,
        sessions: dict[str, UserSession],
        tts: TTSBackend,
        tick_ms: int = 200,
        scheduler: FloorScheduler | None = None,
        frozen_plan: FrozenEventPlan | None = None,
        allow_frozen_audio_duration_drift: bool = False,
    ) -> None:
        expected_users = {user.user_id for user in case.users}
        if set(sessions) != expected_users:
            raise ValueError("sessions must contain exactly one entry per case user")
        if any(sessions[user_id].profile.user_id != user_id for user_id in sessions):
            raise ValueError("session map key does not match profile user id")

        self.case = case
        self.sessions = sessions
        self.tts = tts
        self.tick_ms = tick_ms
        self.scheduler = scheduler or FloorScheduler(tick_ms=tick_ms, seed=case.seed)
        self.event_graph = EventGraphExecutor(case)
        self.authorization = AuthorizationOracle(case)
        self.mixer = PCMTrackMixer(sample_rate=tts.sample_rate)
        self.queues = {
            user_id: SpeakerAudioQueue(user_id, tts.sample_rate) for user_id in sessions
        }
        self.trace = MultiUserEpisodeTrace(case_id=case.case_id)
        self._event_end_ms: dict[str, int] = {}
        self._event_ready_ms: dict[str, int] = {}
        self._event_rejection_counts: dict[str, int] = {}
        self._effects_applied: set[str] = set()
        self._event_by_id = {event.event_id: event for event in case.events}
        self._frozen_replay = frozen_plan is not None
        # A frozen plan normally must reproduce the exact TTS duration.  When
        # a Probe or Final replays that plan, however, the current TTS backend
        # may be a different build/version and its diagnostic re-synthesis can
        # drift.  In that replay mode we align diagnostic PCM to the frozen
        # duration; Final's immutable mixed track remains authoritative input.
        self._allow_frozen_audio_duration_drift = allow_frozen_audio_duration_drift
        # Preserve an auditable record of the original TTS duration, the
        # frozen-plan duration, and any diagnostic PCM repair.  This is kept
        # outside the public episode trace so evaluator semantics remain
        # unchanged when a replay backend drifts.
        self.frozen_audio_duration_diagnostics: list[dict[str, object]] = []
        if frozen_plan is not None:
            self._load_frozen_plan(frozen_plan)

    def _load_frozen_plan(self, plan: FrozenEventPlan) -> None:
        """Pre-render an immutable user track for a Probe or Final session."""
        if plan.case_id != self.case.case_id:
            raise ValueError("frozen plan belongs to a different case")
        case_events = {event.event_id: event for event in self.case.events}
        declared_order = [event.event_id for event in self.case.events]
        plan_order = [event.event_id for event in plan.events]
        base_count = min(len(plan_order), len(declared_order))
        if plan_order[:base_count] != declared_order[:base_count]:
            raise ValueError("frozen plan must preserve declaration-order case events")
        if any(
            event_id not in case_events and not event_id.startswith("tail_")
            for event_id in plan_order
        ):
            raise ValueError("frozen plan contains an unknown non-tail event")
        previous_event_id: str | None = None
        for frozen in plan.events:
            event = case_events.get(frozen.event_id)
            if event is None:
                if previous_event_id is None:
                    raise ValueError("adaptive tail event cannot be the first event")
                event = ScriptedEvent(
                    event_id=frozen.event_id,
                    actor_id=frozen.actor_id,
                    action=frozen.action,
                    trigger=TriggerSpec(
                        kind=TriggerKind.EVENT_COMPLETED,
                        event_id=previous_event_id,
                    ),
                    intent=frozen.intent,
                    addressee=frozen.addressee,
                    fixed_utterance=frozen.utterance,
                    fixed_utterance_exact=True,
                    required=frozen.required,
                    urgency=frozen.urgency,
                    overlap_allowed=frozen.overlap_allowed,
                    max_delay_ms=frozen.max_delay_ms,
                    thread_id=frozen.thread_id,
                    speech_act=frozen.speech_act,
                    notes=["reconstructed_adaptive_tail_event"],
                )
                case_events[event.event_id] = event
                self._event_by_id[event.event_id] = event
                self.event_graph.register_event(event)
            if frozen.actor_id != event.actor_id:
                raise ValueError(f"frozen speaker changed for {frozen.event_id}")
            if frozen.action != event.action:
                raise ValueError(f"frozen action changed for {frozen.event_id}")
            if frozen.addressee != event.addressee:
                raise ValueError(f"frozen addressee changed for {frozen.event_id}")
            if event.fixed_utterance_exact and frozen.utterance != event.fixed_utterance:
                raise ValueError(f"exact event text changed for {frozen.event_id}")
            profile = self.profiles[frozen.actor_id]
            pcm = self.tts.synthesize(frozen.utterance, profile.voice.voice_id)
            rendered_duration_ms = QueuedAudio(
                audio_id=f"duration:{frozen.event_id}",
                speaker_id=frozen.actor_id,
                text=frozen.utterance,
                start_ms=frozen.start_ms,
                pcm_s16le=pcm,
                sample_rate=self.tts.sample_rate,
            ).duration_ms
            planned_duration_ms = frozen.end_ms - frozen.start_ms
            duration_repaired = rendered_duration_ms != planned_duration_ms
            if duration_repaired and self._allow_frozen_audio_duration_drift:
                # The benchmark voicebank is PCM16 at 16 kHz, but use the
                # backend's declared rate so this remains correct for other
                # offline TTS implementations.  A final one-sample adjustment
                # handles ceil() in QueuedAudio.duration_ms.
                target_samples = max(
                    0,
                    round(planned_duration_ms * self.tts.sample_rate / 1000),
                )
                pcm = pcm[: target_samples * 2].ljust(target_samples * 2, b"\x00")
                aligned_duration_ms = QueuedAudio(
                    audio_id=f"duration-aligned:{frozen.event_id}",
                    speaker_id=frozen.actor_id,
                    text=frozen.utterance,
                    start_ms=frozen.start_ms,
                    pcm_s16le=pcm,
                    sample_rate=self.tts.sample_rate,
                ).duration_ms
                while aligned_duration_ms < planned_duration_ms:
                    pcm += b"\x00\x00"
                    aligned_duration_ms = QueuedAudio(
                        audio_id=f"duration-aligned:{frozen.event_id}",
                        speaker_id=frozen.actor_id,
                        text=frozen.utterance,
                        start_ms=frozen.start_ms,
                        pcm_s16le=pcm,
                        sample_rate=self.tts.sample_rate,
                    ).duration_ms
                while aligned_duration_ms > planned_duration_ms and len(pcm) >= 2:
                    pcm = pcm[:-2]
                    aligned_duration_ms = QueuedAudio(
                        audio_id=f"duration-aligned:{frozen.event_id}",
                        speaker_id=frozen.actor_id,
                        text=frozen.utterance,
                        start_ms=frozen.start_ms,
                        pcm_s16le=pcm,
                        sample_rate=self.tts.sample_rate,
                    ).duration_ms
            else:
                aligned_duration_ms = rendered_duration_ms
            item = QueuedAudio(
                audio_id=f"frozen:{frozen.event_id}",
                speaker_id=frozen.actor_id,
                text=frozen.utterance,
                start_ms=frozen.start_ms,
                pcm_s16le=pcm,
                sample_rate=self.tts.sample_rate,
                metadata={
                    "proposal_id": f"frozen:{frozen.event_id}",
                    "event_id": frozen.event_id,
                    "planning_source": frozen.planning_source,
                    "rendered_duration_ms": str(rendered_duration_ms),
                    "planned_duration_ms": str(planned_duration_ms),
                    "duration_repaired": str(duration_repaired),
                },
            )
            alignment_error_ms = item.end_ms - frozen.end_ms
            self.frozen_audio_duration_diagnostics.append(
                {
                    "event_id": frozen.event_id,
                    "speaker_id": frozen.actor_id,
                    "start_ms": frozen.start_ms,
                    "planned_end_ms": frozen.end_ms,
                    "planned_duration_ms": planned_duration_ms,
                    "rendered_duration_ms": rendered_duration_ms,
                    "aligned_duration_ms": aligned_duration_ms,
                    "original_drift_ms": rendered_duration_ms - planned_duration_ms,
                    "alignment_error_ms": alignment_error_ms,
                    "duration_repaired": duration_repaired,
                    "strict_would_reject": (
                        abs(rendered_duration_ms - planned_duration_ms) > 1
                    ),
                    "strict_tolerance_ms": 1,
                    "repair_policy": (
                        "align_to_frozen_plan"
                        if self._allow_frozen_audio_duration_drift
                        else "strict"
                    ),
                    "accepted": abs(alignment_error_ms) <= 1,
                    "sample_rate": self.tts.sample_rate,
                }
            )
            if abs(alignment_error_ms) > 1:
                if not self._allow_frozen_audio_duration_drift:
                    raise ValueError(
                        f"frozen audio duration drifted for {frozen.event_id}: "
                        f"plan={frozen.end_ms - frozen.start_ms}ms, "
                        f"rendered={item.duration_ms}ms"
                    )
                raise ValueError(
                    f"unable to align frozen diagnostic audio for {frozen.event_id}: "
                    f"plan={frozen.end_ms - frozen.start_ms}ms, "
                    f"rendered={item.duration_ms}ms"
                )
            self.queues[frozen.actor_id].schedule(item)
            self.event_graph.mark_scheduled(frozen.event_id)
            # Preserve the frozen timeline even when the diagnostic PCM had
            # to be length-aligned.  Final input timing is defined by the
            # immutable pre-rendered track and the plan, never by a fresh TTS
            # regeneration.
            self._event_end_ms[frozen.event_id] = frozen.end_ms
            self.trace.spoken_events.append(
                SpokenEventTrace(
                    event_id=frozen.event_id,
                    proposal_id=f"frozen:{frozen.event_id}",
                    speaker_id=frozen.actor_id,
                    addressee=frozen.addressee,
                    utterance=frozen.utterance,
                    start_ms=frozen.start_ms,
                    end_ms=frozen.end_ms,
                    ready_ms=frozen.planning_anchor_ms,
                    trigger_path=frozen.trigger_path,
                    trigger_observation_kind=None,
                    thread_id=frozen.thread_id,
                    speech_act=frozen.speech_act,
                )
            )
            previous_event_id = frozen.event_id

    @property
    def profiles(self) -> dict[str, UserProfile]:
        return {user_id: session.profile for user_id, session in self.sessions.items()}

    @property
    def runtimes(self) -> dict[str, UserRuntimeState]:
        return {user_id: session.runtime for user_id, session in self.sessions.items()}

    def _commit_finished_events(self, sim_time_ms: int) -> None:
        for event_id, end_ms in self._event_end_ms.items():
            if end_ms <= sim_time_ms:
                self.event_graph.mark_completed(event_id, end_ms)
                if event_id in self._effects_applied:
                    continue
                event = self._event_by_id[event_id]
                for effect in event.grant_effects:
                    grants = self.sessions[
                        effect.target_user_id
                    ].runtime.active_grant_ids
                    if effect.active:
                        grants.add(effect.grant_id)
                    else:
                        grants.discard(effect.grant_id)
                for effect in event.presence_effects:
                    self.sessions[
                        effect.target_user_id
                    ].runtime.present = effect.present
                self._effects_applied.add(event_id)

    def _wake_dynamic_users(
        self,
        observation: WorldObservation,
        actors_with_events: set[str],
    ) -> list[UserProposal]:
        proposals = []
        for user_id, session in self.sessions.items():
            if user_id in actors_with_events:
                continue
            if not session.runtime.present:
                continue
            should_wake = (
                user_id in observation.mentioned_user_ids
                or observation.kind in session.profile.wake_on
            )
            if should_wake:
                proposals.append(session.propose(observation))
        return proposals

    def _validate_exact_event(self, proposal: UserProposal) -> None:
        event_id = proposal.triggered_by_event_id
        if not event_id:
            return
        event = self._event_by_id[event_id]
        if event.fixed_utterance_exact and event.fixed_utterance:
            if proposal.utterance != event.fixed_utterance:
                raise ValueError(f"event {event_id} changed its fixed utterance")
        if proposal.actor_id != event.actor_id:
            raise ValueError(f"event {event_id} changed its speaker")

    def update(self, observation: WorldObservation) -> SimulatorTickResult:
        """Advance the policy clock once using only currently committed evidence."""
        self._commit_finished_events(observation.sim_time_ms)
        for user_id, queue in self.queues.items():
            self.sessions[user_id].runtime.currently_speaking = any(
                item.start_ms <= observation.sim_time_ms < item.end_ms
                for item in queue.items
            )

        if self._frozen_replay:
            end_ms = observation.sim_time_ms + self.tick_ms
            audio = self.mixer.mix_window(
                self.queues,
                start_ms=observation.sim_time_ms,
                end_ms=end_ms,
            )
            self._commit_finished_events(end_ms)
            return SimulatorTickResult(
                observation=observation,
                proposals=[],
                schedule=ScheduleDecision(
                    sim_time_ms=observation.sim_time_ms,
                    selected=[],
                    rejected=[],
                ),
                audio=audio,
            )

        ready_events = self.event_graph.ready_events(observation)
        # Validate the entire batch before calling any user LLM/TTS. This is
        # also a boundary guard for custom graph implementations; rejecting a
        # malformed event must not silently skip or complete that event.
        for event in ready_events:
            validate_user_event_actor(event.actor_id, case=self.case, event_id=event.event_id)
        for event in ready_events:
            self._event_ready_ms.setdefault(event.event_id, observation.sim_time_ms)
        proposals = [
            self.sessions[event.actor_id].propose(observation, event)
            for event in ready_events
        ]
        proposals.extend(
            self._wake_dynamic_users(
                observation, {event.actor_id for event in ready_events}
            )
        )
        for proposal in proposals:
            self._validate_exact_event(proposal)

        schedule = self.scheduler.schedule(
            proposals=proposals,
            profiles=self.profiles,
            runtimes=self.runtimes,
            sim_time_ms=observation.sim_time_ms,
        )
        proposals_by_id = {proposal.proposal_id: proposal for proposal in proposals}
        for rejected in schedule.rejected:
            proposal = proposals_by_id.get(rejected.proposal_id)
            if proposal is not None and proposal.triggered_by_event_id:
                event_id = proposal.triggered_by_event_id
                self._event_rejection_counts[event_id] = (
                    self._event_rejection_counts.get(event_id, 0) + 1
                )
        for selected in schedule.selected:
            proposal = selected.proposal
            profile = self.profiles[proposal.actor_id]
            pcm = self.tts.synthesize(proposal.utterance or "", profile.voice.voice_id)
            audio_id = f"audio:{proposal.proposal_id}"
            item = QueuedAudio(
                audio_id=audio_id,
                speaker_id=proposal.actor_id,
                text=proposal.utterance or "",
                start_ms=selected.start_ms,
                pcm_s16le=pcm,
                sample_rate=self.tts.sample_rate,
                metadata={
                    "proposal_id": proposal.proposal_id,
                    "event_id": proposal.triggered_by_event_id or "",
                },
            )
            self.queues[proposal.actor_id].schedule(item)
            runtime = self.sessions[proposal.actor_id].runtime
            runtime.last_spoke_ms = selected.start_ms
            runtime.pending_audio_ids.append(audio_id)
            runtime.memory.append(proposal.utterance or "")
            runtime.active_thread_ids.add(proposal.thread_id)
            if proposal.triggered_by_event_id:
                event_id = proposal.triggered_by_event_id
                self.event_graph.mark_scheduled(event_id)
                self._event_end_ms[event_id] = item.end_ms
            self.trace.spoken_events.append(
                SpokenEventTrace(
                    event_id=proposal.triggered_by_event_id,
                    proposal_id=proposal.proposal_id,
                    speaker_id=proposal.actor_id,
                    addressee=proposal.addressee,
                    utterance=proposal.utterance or "",
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                    ready_ms=(
                        self._event_ready_ms.get(proposal.triggered_by_event_id)
                        if proposal.triggered_by_event_id
                        else observation.sim_time_ms
                    ),
                    trigger_path=(
                        self.event_graph.trigger_path(proposal.triggered_by_event_id)
                        if proposal.triggered_by_event_id
                        else None
                    ),
                    trigger_observation_kind=(
                        self.event_graph.trigger_observation_kind(
                            proposal.triggered_by_event_id
                        )
                        if proposal.triggered_by_event_id
                        else None
                    ),
                    schedule_rejection_count=(
                        self._event_rejection_counts.get(
                            proposal.triggered_by_event_id, 0
                        )
                        if proposal.triggered_by_event_id
                        else 0
                    ),
                    thread_id=proposal.thread_id,
                    speech_act=proposal.speech_act,
                )
            )

        end_ms = observation.sim_time_ms + self.tick_ms
        audio = self.mixer.mix_window(
            self.queues,
            start_ms=observation.sim_time_ms,
            end_ms=end_ms,
        )
        self._commit_finished_events(end_ms)
        return SimulatorTickResult(
            observation=observation,
            proposals=proposals,
            schedule=schedule,
            audio=audio,
        )

    def write_audio(
        self, output_dir: Path, end_ms: int | None = None
    ) -> dict[str, Path]:
        """Persist mono SUT input and oracle speaker stems."""
        if end_ms is None:
            end_ms = max(
                (item.end_ms for queue in self.queues.values() for item in queue.items),
                default=self.tick_ms,
            )
        return self.mixer.write_session(self.queues, output_dir, end_ms)
