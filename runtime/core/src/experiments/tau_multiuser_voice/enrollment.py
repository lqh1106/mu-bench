"""In-session speaker enrollment for a single shared microphone."""

from __future__ import annotations

import math
from pathlib import Path

from experiments.tau_multiuser_voice.audio import (
    MixedAudioWindow,
    PCMTrackMixer,
    QueuedAudio,
    SpeakerAudioQueue,
    TTSBackend,
)
from experiments.tau_multiuser_voice.models import (
    AssistantActivationTrace,
    MultiUserVoiceCase,
    VoiceEnrollmentTrace,
)


class InSessionVoiceEnrollment:
    """Render the spoken Nova preamble and user introductions before task time zero."""

    def __init__(
        self,
        *,
        case: MultiUserVoiceCase,
        tts: TTSBackend,
        tick_ms: int = 200,
        separate_session_protocol: bool = False,
    ) -> None:
        self.case = case
        self.tts = tts
        self.tick_ms = tick_ms
        self.separate_session_protocol = separate_session_protocol
        self.session_protocol_audio = None
        self.mixer = PCMTrackMixer(sample_rate=tts.sample_rate)
        self.queues = {
            user.user_id: SpeakerAudioQueue(user.user_id, tts.sample_rate)
            for user in case.users
        }
        self.traces: list[VoiceEnrollmentTrace] = []
        self.registration_audio_segments = []
        self.activation_trace: AssistantActivationTrace | None = None
        self.enrollment_complete_ms = 0
        self.task_start_ms = 0
        self._build()

    def _align(self, timestamp_ms: int) -> int:
        return math.ceil(timestamp_ms / self.tick_ms) * self.tick_ms

    def _build(self) -> None:
        protocol = self.case.voice_enrollment
        enrolled_users = (
            [
                user
                for user in self.case.users
                if user.voice.enrollment_status == "enrolled"
            ]
            if protocol.mode != "none"
            else []
        )
        participant_aliases = {
            user.user_id: (
                f"{self.case.public_participant_id(user.user_id)} "
                f"{user.display_name}"
            )
            for user in self.case.users
        }
        cursor_ms = self._align(protocol.pre_roll_ms)
        if self.separate_session_protocol and enrolled_users:
            import hashlib
            from .registration_audio import spoken_codes
            from .scene_audio import render_source
            text = spoken_codes(protocol.session_preamble_template.format(
                assistant_name=self.case.assistant_activation.assistant_name))
            pcm, voice = render_source(self.case, self.tts, text, short_segments=True)
            narrator = '__session_protocol_narrator__'
            item = QueuedAudio(audio_id='session-protocol', speaker_id=narrator,
                text=text, start_ms=cursor_ms, pcm_s16le=pcm, sample_rate=self.tts.sample_rate,
                metadata={'phase': 'session_protocol', 'enrolled_participant': 'false'})
            queue = SpeakerAudioQueue(narrator, self.tts.sample_rate)
            queue.schedule(item)
            self.queues[narrator] = queue
            self.session_protocol_audio = {'utterance': text, 'session_start_ms': item.start_ms,
                'session_end_ms': item.end_ms, 'speaker_id': narrator, 'narrator': voice,
                'enrolled_participant_added': False, 'pcm_sha256': hashlib.sha256(pcm).hexdigest()}
            cursor_ms = self._align(item.end_ms + protocol.inter_speaker_gap_ms)
        for index, user in enumerate(enrolled_users):
            participant_id = self.case.public_participant_id(user.user_id)
            utterance = protocol.render_enrollment_utterance(
                user_id=participant_id,
                display_name=user.display_name,
                role=user.role,
                permissions=user.permissions,
                oracle_user_id=user.user_id,
                participant_aliases=participant_aliases,
                assistant_name=self.case.assistant_activation.assistant_name,
                include_session_preamble=index == 0 and not self.separate_session_protocol,
            )
            from experiments.tau_multiuser_voice.registration_audio import spoken_codes, synthesize
            utterance = spoken_codes(utterance)
            pcm, parts = synthesize(self.tts, utterance, user.voice.voice_id)
            self.registration_audio_segments.append({'user_id': user.user_id, 'voice_id': user.voice.voice_id, 'segments': parts})
            item = QueuedAudio(
                audio_id=f"enrollment:{user.user_id}",
                speaker_id=user.user_id,
                text=utterance,
                start_ms=cursor_ms,
                pcm_s16le=pcm,
                sample_rate=self.tts.sample_rate,
                metadata={"phase": "enrollment", "user_id": user.user_id},
            )
            self.queues[user.user_id].schedule(item)
            self.traces.append(
                VoiceEnrollmentTrace(
                    user_id=user.user_id,
                    display_name=user.display_name,
                    role=user.role,
                    voice_id=user.voice.voice_id,
                    utterance=utterance,
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                    enrollment_status=user.voice.enrollment_status,
                )
            )
            cursor_ms = item.end_ms
            if index + 1 < len(enrolled_users):
                cursor_ms = self._align(cursor_ms + protocol.inter_speaker_gap_ms)
        if self.traces:
            cursor_ms += protocol.post_roll_ms
        self.enrollment_complete_ms = self._align(cursor_ms)

        activation = self.case.assistant_activation
        if not activation.enabled:
            self.task_start_ms = self.enrollment_complete_ms
            return
        enrolled_ids = {trace.user_id for trace in self.traces}
        caller_user_id = activation.caller_user_id
        if caller_user_id is None:
            caller_user_id = next(
                (
                    user.user_id
                    for user in self.case.users
                    if user.user_id in enrolled_ids
                ),
                self.case.users[0].user_id,
            )
        caller = next(
            user for user in self.case.users if user.user_id == caller_user_id
        )
        activation_start_ms = self._align(
            self.enrollment_complete_ms + activation.pre_roll_ms
        )
        utterance = protocol.render_activation_utterance(
            fallback_template=activation.utterance_template,
            assistant_name=activation.assistant_name,
            user_id=caller.user_id,
            display_name=caller.display_name,
            role=caller.role,
        )
        pcm = self.tts.synthesize(utterance, caller.voice.voice_id)
        item = QueuedAudio(
            audio_id="assistant-activation",
            speaker_id=caller.user_id,
            text=utterance,
            start_ms=activation_start_ms,
            pcm_s16le=pcm,
            sample_rate=self.tts.sample_rate,
            metadata={
                "phase": "assistant_activation",
                "user_id": caller.user_id,
                "assistant_name": activation.assistant_name,
            },
        )
        self.queues[caller.user_id].schedule(item)
        self.activation_trace = AssistantActivationTrace(
            assistant_name=activation.assistant_name,
            caller_user_id=caller.user_id,
            utterance=utterance,
            start_ms=item.start_ms,
            end_ms=item.end_ms,
        )
        self.task_start_ms = self._align(item.end_ms + activation.post_roll_ms)

    @property
    def enabled(self) -> bool:
        return bool(self.traces)

    def prompt_bindings(self) -> list[dict[str, object]]:
        """Return evaluator-only identity-to-audio-span bindings."""
        return [
            {
                "user_id": trace.user_id,
                "participant_id": self.case.public_participant_id(trace.user_id),
                "display_name": trace.display_name,
                "role": trace.role,
                "session_audio_start_ms": trace.start_ms,
                "session_audio_end_ms": trace.end_ms,
                "voice_sample_text": trace.utterance,
            }
            for trace in self.traces
        ]

    def control_events(self) -> list[dict[str, object]]:
        """Return no events: identity and phase transitions are audio-only."""
        return []

    def mix_window(self, start_ms: int, end_ms: int) -> MixedAudioWindow:
        """Render one enrollment window on the shared session clock."""
        return self.mixer.mix_window(
            self.queues,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    def combined_session_queues(
        self,
        task_queues: dict[str, SpeakerAudioQueue],
    ) -> dict[str, SpeakerAudioQueue]:
        """Combine enrollment and task stems on the shared session clock."""
        combined = {
            user_id: SpeakerAudioQueue(user_id, self.tts.sample_rate)
            for user_id in self.queues
        }
        for user_id, queue in self.queues.items():
            for item in queue.items:
                combined[user_id].schedule(item.model_copy(deep=True))
        for user_id, queue in task_queues.items():
            for item in queue.items:
                combined[user_id].schedule(
                    item.model_copy(
                        deep=True,
                        update={"start_ms": item.start_ms + self.task_start_ms},
                    )
                )
        return combined

    def write_session_audio(
        self,
        task_queues: dict[str, SpeakerAudioQueue],
        output_dir: Path,
        task_end_ms: int,
    ) -> dict[str, Path]:
        """Persist full-session oracle stems with task audio shifted after enrollment."""
        queues = self.combined_session_queues(task_queues)
        return self.mixer.write_session(
            queues,
            output_dir,
            self.task_start_ms + task_end_ms,
        )
