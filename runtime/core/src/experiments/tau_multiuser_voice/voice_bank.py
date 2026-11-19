"""Versioned, quality-gated speech assets for reproducible voice evaluation."""

from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from experiments.tau_multiuser_voice.models import (
    ScriptedEvent,
    TriggerKind,
    TriggerSpec,
)


class VoiceQualityMetrics(BaseModel):
    """Measured properties of one rendered clip; automatic scores are proxies."""

    asr_word_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    asr_character_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    speaker_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    mos_proxy: float | None = Field(default=None, ge=1.0, le=5.0)
    clipping_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    snr_db: float | None = None
    alignment_error_ms: float | None = Field(default=None, ge=0.0)
    human_naturalness_mos: float | None = Field(default=None, ge=1.0, le=5.0)
    human_accent_authenticity: float | None = Field(default=None, ge=1.0, le=5.0)


class VoiceSetQualityMetrics(BaseModel):
    """Identity separation measured jointly over every voice in one case."""

    max_inter_speaker_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    speaker_verification_eer: float | None = Field(default=None, ge=0.0, le=1.0)
    closed_set_confusion_rate: float | None = Field(default=None, ge=0.0, le=1.0)


class VoiceQualityThresholds(BaseModel):
    """Release-specific gates calibrated on the selected TTS and human anchors."""

    max_asr_word_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    max_asr_character_error_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_speaker_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    min_mos_proxy: float | None = Field(default=None, ge=1.0, le=5.0)
    max_clipping_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    max_alignment_error_ms: float | None = Field(default=None, ge=0.0)
    min_human_naturalness_mos: float | None = Field(default=None, ge=1.0, le=5.0)
    min_human_accent_authenticity: float | None = Field(
        default=None, ge=1.0, le=5.0
    )
    max_inter_speaker_similarity: float | None = Field(
        default=None, ge=-1.0, le=1.0
    )
    max_speaker_verification_eer: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    max_closed_set_confusion_rate: float | None = Field(
        default=None, ge=0.0, le=1.0
    )


class VoiceClipAsset(BaseModel):
    """One immutable transcript-to-WAV render in a released voice bank."""

    clip_id: str
    text: str
    wav_path: str
    sha256: str
    phase: Literal["enrollment", "activation", "task"]
    dry_source_sha256: str | None = None
    acoustic_scene_id: str = "dry_close_talk"
    metrics: VoiceQualityMetrics = Field(default_factory=VoiceQualityMetrics)


class AcousticSceneSpec(BaseModel):
    """Physical rendering controls applied after synthesis to a dry voice clip."""

    scene_id: str
    room_id: str
    distance_m: float = Field(gt=0.0)
    azimuth_deg: float = Field(ge=-180.0, le=180.0)
    off_axis_deg: float = Field(default=0.0, ge=0.0, le=180.0)
    gain_db: float = 0.0
    rir_id: str | None = None
    background_bed_id: str | None = None
    channel_preset: Literal[
        "fullband",
        "far_field",
        "telephony_g711_ulaw",
        "packet_loss",
    ] = "fullband"


class SpeakerVoiceAsset(BaseModel):
    """Stable voice provenance and all clips for one benchmark participant."""

    user_id: str
    voice_id: str
    voice_catalog_id: str | None = None
    backend: str
    model_id: str
    model_version: str
    synthesis_seed: int | None = None
    reference_audio_id: str | None = None
    reference_audio_sha256: str | None = None
    locale: str = "en-US"
    intended_accent: str = "General American"
    speaking_style: str = "natural conversational speech"
    # Copied from the frozen global binding so a released case voice bank is
    # self-describing and cannot silently fall back to a different persona
    # rendering policy.
    persona_voice_profile: dict[str, object] = Field(default_factory=dict)
    consent_record_id: str | None = None
    license_id: str = Field(min_length=1)
    clips: list[VoiceClipAsset]

    @model_validator(mode="after")
    def validate_clips(self) -> SpeakerVoiceAsset:
        keys = [(clip.phase, clip.text) for clip in self.clips]
        if len(keys) != len(set(keys)):
            raise ValueError("one voice cannot contain duplicate phase/text clips")
        return self


class VoiceBankManifest(BaseModel):
    """Versioned audio release consumed identically by every tested model."""

    # 2.0 remains readable so historical frozen voice banks can be replayed;
    # new templates and CosyVoice renders publish 2.1.
    schema_version: Literal["2.0", "2.1"] = "2.1"
    voice_bank_version: str
    case_id: str
    identity_delivery: Literal["spoken_tts_self_introduction"] = (
        "spoken_tts_self_introduction"
    )
    tts_backend: Literal[
        "cosyvoice",
        "gpt_audio_smoke",
        "human_anchor",
        "external_renderer",
    ] = "external_renderer"
    voice_catalog_version: str | None = None
    enrollment_transport: Literal["audio_self_identification"] = (
        "audio_self_identification"
    )
    track: Literal[
        "clean_synthetic",
        "realistic_synthetic",
        "human_anchor",
        "counterfactual_voice_swap",
    ]
    release_status: Literal["development", "leaderboard_candidate"] = "development"
    sample_rate: int = Field(default=16000, gt=0)
    channels: Literal[1] = 1
    sample_width_bytes: Literal[2] = 2
    assignment_seed: int
    paired_render_group: str | None = None
    quality_thresholds: VoiceQualityThresholds = Field(
        default_factory=VoiceQualityThresholds
    )
    voice_set_metrics: VoiceSetQualityMetrics = Field(
        default_factory=VoiceSetQualityMetrics
    )
    effects_pipeline: list[dict[str, Any]] = Field(default_factory=list)
    acoustic_scenes: list[AcousticSceneSpec] = Field(
        default_factory=lambda: [
            AcousticSceneSpec(
                scene_id="dry_close_talk",
                room_id="dry",
                distance_m=0.3,
                azimuth_deg=0.0,
            )
        ]
    )
    speakers: list[SpeakerVoiceAsset]

    @model_validator(mode="after")
    def validate_speakers(self) -> VoiceBankManifest:
        user_ids = [speaker.user_id for speaker in self.speakers]
        voice_ids = [speaker.voice_id for speaker in self.speakers]
        if len(user_ids) != len(set(user_ids)):
            raise ValueError("voice bank user ids must be unique")
        if len(voice_ids) != len(set(voice_ids)):
            raise ValueError("voice bank voice ids must be unique within a case")
        scene_ids = [scene.scene_id for scene in self.acoustic_scenes]
        if len(scene_ids) != len(set(scene_ids)):
            raise ValueError("acoustic scene ids must be unique")
        unknown_scenes = {
            clip.acoustic_scene_id
            for speaker in self.speakers
            for clip in speaker.clips
        } - set(scene_ids)
        if unknown_scenes:
            raise ValueError(
                f"voice clips reference unknown acoustic scenes: {unknown_scenes}"
            )
        if self.tts_backend == "cosyvoice":
            if not self.voice_catalog_version:
                raise ValueError(
                    "CosyVoice voice banks must declare voice_catalog_version"
                )
            missing_catalog_ids = [
                speaker.user_id
                for speaker in self.speakers
                if not speaker.voice_catalog_id
            ]
            if missing_catalog_ids:
                raise ValueError(
                    "CosyVoice voice banks must bind every speaker to a catalog ID: "
                    + ", ".join(missing_catalog_ids)
                )
        if self.release_status == "leaderboard_candidate":
            required_thresholds = (
                self.quality_thresholds.max_asr_word_error_rate,
                self.quality_thresholds.min_speaker_similarity,
                self.quality_thresholds.min_mos_proxy,
                self.quality_thresholds.max_clipping_ratio,
                self.quality_thresholds.max_inter_speaker_similarity,
                self.quality_thresholds.max_closed_set_confusion_rate,
            )
            if any(value is None for value in required_thresholds):
                raise ValueError(
                    "leaderboard voice banks must declare content, identity, "
                    "naturalness, clipping, and inter-speaker quality gates"
                )
            missing_consent = [
                speaker.user_id
                for speaker in self.speakers
                if (
                    self.track == "human_anchor"
                    or speaker.reference_audio_id is not None
                )
                and speaker.consent_record_id is None
            ]
            if missing_consent:
                raise ValueError(
                    "human or cloned leaderboard voices require consent records: "
                    + ", ".join(missing_consent)
                )
        return self

    @classmethod
    def load(cls, path: Path) -> VoiceBankManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_quality(
    clip: VoiceClipAsset,
    thresholds: VoiceQualityThresholds,
    *,
    require_complete: bool,
) -> list[str]:
    checks = (
        ("asr_word_error_rate", "max_asr_word_error_rate", lambda a, b: a <= b),
        (
            "asr_character_error_rate",
            "max_asr_character_error_rate",
            lambda a, b: a <= b,
        ),
        ("speaker_similarity", "min_speaker_similarity", lambda a, b: a >= b),
        ("mos_proxy", "min_mos_proxy", lambda a, b: a >= b),
        ("clipping_ratio", "max_clipping_ratio", lambda a, b: a <= b),
        ("alignment_error_ms", "max_alignment_error_ms", lambda a, b: a <= b),
        (
            "human_naturalness_mos",
            "min_human_naturalness_mos",
            lambda a, b: a >= b,
        ),
        (
            "human_accent_authenticity",
            "min_human_accent_authenticity",
            lambda a, b: a >= b,
        ),
    )
    failures: list[str] = []
    for metric_name, threshold_name, predicate in checks:
        threshold = getattr(thresholds, threshold_name)
        if threshold is None:
            continue
        value = getattr(clip.metrics, metric_name)
        if value is None:
            if require_complete:
                failures.append(f"{clip.clip_id}:missing:{metric_name}")
        elif not predicate(value, threshold):
            failures.append(f"{clip.clip_id}:failed:{metric_name}")
    return failures


def _check_voice_set_quality(
    metrics: VoiceSetQualityMetrics,
    thresholds: VoiceQualityThresholds,
    *,
    require_complete: bool,
) -> list[str]:
    checks = (
        (
            "max_inter_speaker_similarity",
            "max_inter_speaker_similarity",
            lambda a, b: a <= b,
        ),
        (
            "speaker_verification_eer",
            "max_speaker_verification_eer",
            lambda a, b: a <= b,
        ),
        (
            "closed_set_confusion_rate",
            "max_closed_set_confusion_rate",
            lambda a, b: a <= b,
        ),
    )
    failures: list[str] = []
    for metric_name, threshold_name, predicate in checks:
        threshold = getattr(thresholds, threshold_name)
        if threshold is None:
            continue
        value = getattr(metrics, metric_name)
        if value is None:
            if require_complete:
                failures.append(f"voice_set:missing:{metric_name}")
        elif not predicate(value, threshold):
            failures.append(f"voice_set:failed:{metric_name}")
    return failures


class ManifestVoiceBankTTS:
    """Read exact pre-rendered PCM clips instead of resynthesizing per model run."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        require_complete_quality: bool = True,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.manifest = VoiceBankManifest.load(self.manifest_path)
        self.sample_rate = self.manifest.sample_rate
        self.require_complete_quality = require_complete_quality
        self._registered_voice_ids: set[str] = set()
        self._clips = {
            (speaker.voice_id, clip.text): clip
            for speaker in self.manifest.speakers
            for clip in speaker.clips
        }
        self._validate_assets()

    def _asset_path(self, relative_path: str) -> Path:
        root = self.manifest_path.parent.resolve()
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"voice clip escapes manifest root: {relative_path}"
            ) from exc
        return path

    def _validate_assets(self) -> None:
        failures = _check_voice_set_quality(
            self.manifest.voice_set_metrics,
            self.manifest.quality_thresholds,
            require_complete=self.require_complete_quality,
        )
        for speaker in self.manifest.speakers:
            for clip in speaker.clips:
                path = self._asset_path(clip.wav_path)
                if not path.is_file():
                    failures.append(f"{clip.clip_id}:missing_wav")
                    continue
                if _sha256(path) != clip.sha256.lower():
                    failures.append(f"{clip.clip_id}:sha256_mismatch")
                    continue
                with wave.open(str(path), "rb") as handle:
                    audio_format = (
                        handle.getnchannels(),
                        handle.getsampwidth(),
                        handle.getframerate(),
                        handle.getcomptype(),
                    )
                expected = (
                    self.manifest.channels,
                    self.manifest.sample_width_bytes,
                    self.manifest.sample_rate,
                    "NONE",
                )
                if audio_format != expected:
                    failures.append(f"{clip.clip_id}:wav_format:{audio_format}")
                failures.extend(
                    _check_quality(
                        clip,
                        self.manifest.quality_thresholds,
                        require_complete=self.require_complete_quality,
                    )
                )
        if failures:
            raise ValueError("voice bank validation failed: " + ", ".join(failures))

    def register_profiles(self, users: list[Any]) -> None:
        """Require exact case/user/voice bindings before any audio is returned."""
        user_voice = {
            speaker.user_id: speaker.voice_id for speaker in self.manifest.speakers
        }
        requested = {user.user_id: user.voice.voice_id for user in users}
        if requested != user_voice:
            raise ValueError(
                "voice bank participant binding mismatch: "
                f"expected {requested}, got {user_voice}"
            )
        self._registered_voice_ids = set(requested.values())

    def validate_for_case(self, case: Any) -> None:
        """Ensure registration, invocation, and every fixed event have a clip."""
        if self.manifest.case_id != case.case_id:
            raise ValueError(
                f"voice bank case mismatch: {self.manifest.case_id} != {case.case_id}"
            )
        if self.manifest.enrollment_transport != case.voice_enrollment.transport:
            raise ValueError(
                "voice bank enrollment transport mismatch: "
                f"{self.manifest.enrollment_transport} != "
                f"{case.voice_enrollment.transport}"
            )
        required: list[tuple[str, str, str]] = []
        first_enrolled_user_id = next(
            (
                user.user_id
                for user in case.users
                if user.voice.enrollment_status == "enrolled"
            ),
            None,
        )
        for user in case.users:
            if user.voice.enrollment_status == "enrolled":
                required.append(
                    (
                        user.voice.voice_id,
                        case.voice_enrollment.render_enrollment_utterance(
                            user_id=case.public_participant_id(user.user_id),
                            display_name=user.display_name,
                            role=user.role,
                            permissions=user.permissions,
                            oracle_user_id=user.user_id,
                            participant_aliases={
                                item.user_id: (
                                    f"{case.public_participant_id(item.user_id)} "
                                    f"{item.display_name}"
                                )
                                for item in case.users
                            },
                            assistant_name=case.assistant_activation.assistant_name,
                            include_session_preamble=(
                                user.user_id == first_enrolled_user_id
                            ),
                        ),
                        "enrollment",
                    )
                )
        activation = case.assistant_activation
        if activation.enabled:
            caller_id = activation.caller_user_id or case.users[0].user_id
            caller = next(user for user in case.users if user.user_id == caller_id)
            required.append(
                (
                    caller.voice.voice_id,
                    case.voice_enrollment.render_activation_utterance(
                        fallback_template=activation.utterance_template,
                        assistant_name=activation.assistant_name,
                        user_id=caller.user_id,
                        display_name=caller.display_name,
                        role=caller.role,
                    ),
                    "activation",
                )
            )
        by_user = {user.user_id: user for user in case.users}
        dynamic_events = [
            event.event_id
            for event in case.events
            if not event.fixed_utterance or not event.fixed_utterance_exact
        ]
        if dynamic_events:
            raise ValueError(
                "a frozen voice bank cannot cover dynamically worded events: "
                + ", ".join(dynamic_events)
            )
        required.extend(
            (by_user[event.actor_id].voice.voice_id, event.fixed_utterance, "task")
            for event in case.events
        )
        missing = [
            f"{phase}:{voice_id}:{text}"
            for voice_id, text, phase in required
            if (voice_id, text) not in self._clips
            or self._clips[(voice_id, text)].phase != phase
        ]
        if missing:
            raise ValueError(
                "voice bank is missing required clips: " + " | ".join(missing)
            )

    def validate_for_plan(self, case: Any, plan: Any) -> None:
        """Validate clips against a complete frozen event plan, not case drafts."""
        if plan.case_id != case.case_id or not plan.complete:
            raise ValueError("voice bank Final requires one complete matching plan")
        if self.manifest.case_id != case.case_id:
            raise ValueError(
                f"voice bank case mismatch: {self.manifest.case_id} != {case.case_id}"
            )
        if self.manifest.enrollment_transport != case.voice_enrollment.transport:
            raise ValueError(
                "voice bank enrollment transport mismatch: "
                f"{self.manifest.enrollment_transport} != "
                f"{case.voice_enrollment.transport}"
            )
        required: list[tuple[str, str, str]] = []
        first_enrolled_user_id = next(
            (
                user.user_id
                for user in case.users
                if user.voice.enrollment_status == "enrolled"
            ),
            None,
        )
        for user in case.users:
            if user.voice.enrollment_status == "enrolled":
                required.append(
                    (
                        user.voice.voice_id,
                        case.voice_enrollment.render_enrollment_utterance(
                            user_id=case.public_participant_id(user.user_id),
                            display_name=user.display_name,
                            role=user.role,
                            permissions=user.permissions,
                            oracle_user_id=user.user_id,
                            participant_aliases={
                                item.user_id: (
                                    f"{case.public_participant_id(item.user_id)} "
                                    f"{item.display_name}"
                                )
                                for item in case.users
                            },
                            assistant_name=case.assistant_activation.assistant_name,
                            include_session_preamble=(
                                user.user_id == first_enrolled_user_id
                            ),
                        ),
                        "enrollment",
                    )
                )
        activation = case.assistant_activation
        if activation.enabled:
            caller_id = activation.caller_user_id or case.users[0].user_id
            caller = next(user for user in case.users if user.user_id == caller_id)
            required.append(
                (
                    caller.voice.voice_id,
                    case.voice_enrollment.render_activation_utterance(
                        fallback_template=activation.utterance_template,
                        assistant_name=activation.assistant_name,
                        user_id=caller.user_id,
                        display_name=caller.display_name,
                        role=caller.role,
                    ),
                    "activation",
                )
            )
        users = {user.user_id: user for user in case.users}
        required.extend(
            (users[event.actor_id].voice.voice_id, event.utterance, "task")
            for event in plan.events
        )
        missing = [
            f"{phase}:{voice_id}:{text}"
            for voice_id, text, phase in required
            if (voice_id, text) not in self._clips
            or self._clips[(voice_id, text)].phase != phase
        ]
        if missing:
            raise ValueError(
                "voice bank is missing frozen-plan clips: " + " | ".join(missing)
            )

    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Return exact released PCM for a transcript/voice pair."""
        if voice_id not in self._registered_voice_ids:
            raise ValueError(f"voice profile was not registered: {voice_id}")
        clip = self._clips.get((voice_id, text))
        if clip is None:
            raise ValueError(f"voice bank has no exact clip for {voice_id}: {text}")
        path = self._asset_path(clip.wav_path)
        with wave.open(str(path), "rb") as handle:
            return handle.readframes(handle.getnframes())

    def provenance(self) -> dict[str, Any]:
        """Return reproducibility metadata without exposing biometric references."""
        return {
            "source": "versioned_voice_bank",
            "manifest": str(self.manifest_path),
            "voice_bank_version": self.manifest.voice_bank_version,
            "enrollment_transport": self.manifest.enrollment_transport,
            "tts_backend": self.manifest.tts_backend,
            "voice_catalog_version": self.manifest.voice_catalog_version,
            "track": self.manifest.track,
            "release_status": self.manifest.release_status,
            "assignment_seed": self.manifest.assignment_seed,
            "leaderboard_eligible": (
                self.manifest.release_status == "leaderboard_candidate"
            ),
            "models": sorted(
                {
                    f"{speaker.backend}:{speaker.model_id}:{speaker.model_version}"
                    for speaker in self.manifest.speakers
                }
            ),
        }


def write_manifest_template(case: Any, path: Path, *, track: str) -> None:
    """Write the complete clip inventory expected from an external renderer."""
    activation = case.assistant_activation
    activation_caller = activation.caller_user_id or case.users[0].user_id
    first_enrolled_user_id = next(
        (
            user.user_id
            for user in case.users
            if user.voice.enrollment_status == "enrolled"
        ),
        None,
    )
    events_by_user: dict[str, list[tuple[str, str, str]]] = {
        user.user_id: [] for user in case.users
    }
    for user in case.users:
        if user.voice.enrollment_status == "enrolled":
            events_by_user[user.user_id].append(
                (
                    f"enrollment:{user.user_id}",
                    case.voice_enrollment.render_enrollment_utterance(
                        user_id=case.public_participant_id(user.user_id),
                        display_name=user.display_name,
                        role=user.role,
                        permissions=user.permissions,
                        oracle_user_id=user.user_id,
                        participant_aliases={
                            item.user_id: (
                                f"{case.public_participant_id(item.user_id)} "
                                f"{item.display_name}"
                            )
                            for item in case.users
                        },
                        assistant_name=case.assistant_activation.assistant_name,
                        include_session_preamble=(
                            user.user_id == first_enrolled_user_id
                        ),
                    ),
                    "enrollment",
                )
            )
        if activation.enabled and user.user_id == activation_caller:
            events_by_user[user.user_id].append(
                (
                    "assistant-activation",
                    case.voice_enrollment.render_activation_utterance(
                        fallback_template=activation.utterance_template,
                        assistant_name=activation.assistant_name,
                        user_id=user.user_id,
                        display_name=user.display_name,
                        role=user.role,
                    ),
                    "activation",
                )
            )
    dynamic_events = [
        event.event_id
        for event in case.events
        if not event.fixed_utterance or not event.fixed_utterance_exact
    ]
    if dynamic_events:
        raise ValueError(
            "manifest templates require fixed event wording: "
            + ", ".join(dynamic_events)
        )
    for event in case.events:
        events_by_user[event.actor_id].append(
            (event.event_id, event.fixed_utterance, "task")
        )
    payload = {
        "schema_version": "2.1",
        "voice_bank_version": "FILL_ME",
        "case_id": case.case_id,
        "identity_delivery": "spoken_tts_self_introduction",
        "assistant_name": case.assistant_activation.assistant_name,
        "session_preamble": case.voice_enrollment.session_preamble_template,
        "formal_start_instruction": case.voice_enrollment.self_identification_activation_template,
        "tts_backend": "external_renderer",
        "voice_catalog_version": None,
        "enrollment_transport": case.voice_enrollment.transport,
        "track": track,
        "release_status": "development",
        "sample_rate": 16000,
        "channels": 1,
        "sample_width_bytes": 2,
        "assignment_seed": case.seed,
        "paired_render_group": None,
        "quality_thresholds": {},
        "voice_set_metrics": {},
        "effects_pipeline": [],
        "acoustic_scenes": [
            {
                "scene_id": "dry_close_talk",
                "room_id": "dry",
                "distance_m": 0.3,
                "azimuth_deg": 0.0,
                "off_axis_deg": 0.0,
                "gain_db": 0.0,
                "rir_id": None,
                "background_bed_id": None,
                "channel_preset": "fullband",
            }
        ],
        "speakers": [
            {
                "user_id": user.user_id,
                "voice_id": user.voice.voice_id,
                "voice_catalog_id": None,
                "backend": "FILL_ME",
                "model_id": "FILL_ME",
                "model_version": "FILL_ME",
                "synthesis_seed": case.seed,
                "reference_audio_id": None,
                "reference_audio_sha256": None,
                "locale": user.voice.locale,
                "intended_accent": user.voice.accent_label,
                "speaking_style": user.voice.speaking_style,
                "persona_voice_profile": {},
                "consent_record_id": None,
                "license_id": "FILL_ME",
                "clips": [
                    {
                        "clip_id": clip_id,
                        "text": text,
                        "wav_path": f"clips/{user.user_id}/{clip_id}.wav",
                        "sha256": "FILL_ME",
                        "phase": phase,
                        "dry_source_sha256": None,
                        "acoustic_scene_id": "dry_close_talk",
                        "metrics": {},
                    }
                    for clip_id, text, phase in events_by_user[user.user_id]
                ],
            }
            for user in case.users
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_plan_manifest_template(
    case: Any, plan: Any, path: Path, *, track: str
) -> None:
    """Export the exact dynamic lines after Probe planning has frozen them."""
    if plan.case_id != case.case_id or not plan.complete:
        raise ValueError("manifest export requires one complete matching plan")
    planned = {event.event_id: event for event in plan.events}
    case_event_ids = [event.event_id for event in case.events]
    plan_event_ids = [event.event_id for event in plan.events]
    if plan_event_ids[: len(case_event_ids)] != case_event_ids or any(
        not event_id.startswith("tail_")
        for event_id in plan_event_ids[len(case_event_ids) :]
    ):
        raise ValueError(
            "frozen plan must cover the case prefix; only adaptive tail_* events "
            "may follow it"
        )
    case_copy = case.model_copy(deep=True)
    case_copy.events = [
        event.model_copy(
            deep=True,
            update={
                "fixed_utterance": planned[event.event_id].utterance,
                "fixed_utterance_exact": True,
            },
        )
        for event in case.events
    ]
    previous_event_id = case_event_ids[-1] if case_event_ids else None
    for frozen in plan.events[len(case_event_ids) :]:
        if previous_event_id is None:
            raise ValueError("adaptive tail event cannot be the first event")
        case_copy.events.append(
            ScriptedEvent(
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
        )
        previous_event_id = frozen.event_id
    write_manifest_template(case_copy, path, track=track)
