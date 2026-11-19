"""Local, cache-first TTS backends for μ-Bench.

The production path is intentionally offline: CosyVoice runs once on a GPU
worker, uses a catalog reference recording for each participant, and stores
PCM plus provenance in a content-addressed cache.  Realtime/API TTS remains a
separate smoke path and is never silently substituted for a requested local
backend.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import os
import random
import sys
import threading
import types
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tau_multiuser_voice.voice_catalog import (
    GlobalVoiceBindingManifest,
    VoiceCatalogEntry,
    VoiceCatalogManifest,
)


def _load_wav_with_torchcodec_fallback(load_wav: Any, path: str, target_rate: int) -> Any:
    """Load CosyVoice prompts across torchaudio 2.x backend changes.

    Recent torchaudio releases route ``torchaudio.load`` through TorchCodec,
    even when the official CosyVoice helper requests the soundfile backend.
    The benchmark development CPU environment intentionally does not carry
    TorchCodec, so fall back to soundfile only for that specific import error.
    """
    try:
        return load_wav(path, target_rate)
    except ImportError as exc:
        if "torchcodec" not in str(exc).lower():
            raise
        import soundfile as sf
        import torch
        import torchaudio

        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        speech = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
        if sample_rate != target_rate:
            if sample_rate < 16_000:
                raise ValueError(
                    f"wav sample rate {sample_rate} must be at least 16000"
                )
            speech = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=target_rate
            )(speech)
        return speech


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _resample_pcm16(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    if source_rate == target_rate:
        return pcm
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    source = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    if source.size == 0:
        return pcm
    target_size = max(1, round(source.size * target_rate / source_rate))
    source_axis = np.linspace(0.0, 1.0, source.size, endpoint=False)
    target_axis = np.linspace(0.0, 1.0, target_size, endpoint=False)
    target = np.interp(target_axis, source_axis, source)
    return np.clip(target, -32768, 32767).astype("<i2").tobytes()


def _waveform_to_pcm16(waveform: Any, source_rate: int, target_rate: int) -> bytes:
    """Convert a CosyVoice tensor/array to mono PCM16 without importing torch."""

    if hasattr(waveform, "detach"):
        waveform = waveform.detach().float().cpu().numpy()
    array = np.asarray(waveform)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim > 1:
        array = array.reshape(-1)
    array = array.astype(np.float32, copy=False)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak <= 1.5:
        array = array * 32767.0
    pcm = np.clip(array, -32768, 32767).astype("<i2").tobytes()
    return _resample_pcm16(pcm, source_rate, target_rate)


def _install_offline_wetext_shim() -> bool:
    """Prevent CosyVoice's optional text frontend from making network calls.

    ``CosyVoiceFrontEnd`` falls back to ``wetext.Normalizer`` when the
    ``ttsfrd`` resources are unavailable.  Some ``wetext`` releases download
    a ModelScope normalizer during construction, which is not possible on a
    sealed GPU worker and is undesirable for a reproducible benchmark run.
    When explicitly enabled, provide the smallest compatible normalizer before
    importing the official CosyVoice modules.  Number expansion and paragraph
    splitting remain in CosyVoice's own frontend; this shim only makes the
    optional normalization step a deterministic pass-through.
    """

    # The benchmark's CosyVoice path is required to be cache-first and
    # network-isolated. Keep the opt-out explicit for development users who
    # intentionally want the official ModelScope-backed normalizer.
    if os.environ.get("MU_BENCH_COSYVOICE_OFFLINE_TEXT", "1") != "1":
        return False

    class _PassthroughNormalizer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def normalize(self, text: str, *args: Any, **kwargs: Any) -> str:
            del args, kwargs
            return text

    module = types.ModuleType("wetext")
    module.Normalizer = _PassthroughNormalizer  # type: ignore[attr-defined]
    sys.modules["wetext"] = module
    return True



# Fix-C7-B: CosyVoice ultra-short text frontend pad (infra).
# Pads only the string sent to inference_zero_shot; API/cache key text stays original.
# Policy id: fix_c7_short_text_pad_v1
_SHORT_BACKCHANNEL_ALLOWLIST: dict[str, str] = {
    "mm-hm": "Mm-hm, yeah.",
    "mm hm": "Mm-hm, yeah.",
    "mmhm": "Mm-hm, yeah.",
    "mhm": "Mhm, yeah.",
    "mm": "Mm, okay.",
    "uh-huh": "Uh-huh, yeah.",
    "uh huh": "Uh-huh, yeah.",
    "uhhuh": "Uh-huh, yeah.",
    "oh": "Oh, okay.",
    "ok": "Okay, sure.",
    "okay": "Okay, sure.",
    "got it": "Got it, okay.",
    "bye": "Bye for now.",
    "yes": "Yes, okay.",
    "no": "No, okay.",
    "yeah": "Yeah, okay.",
    "yep": "Yep, okay.",
    "yup": "Yup, okay.",
    "sure": "Sure, okay.",
    "right": "Right, okay.",
    "alright": "Alright, okay.",
    "all right": "Alright, okay.",
    "huh": "Huh, okay.",
    "hmm": "Hmm, okay.",
    "hm": "Hm, okay.",
    "ah": "Ah, okay.",
    "i see": "I see, okay.",
    "thanks": "Thanks, okay.",
    "thank you": "Thank you, okay.",
}


def _normalize_short_backchannel_key(text: str) -> str:
    normalized = " ".join(str(text).strip().lower().split())
    while normalized and normalized[-1] in ".!?,;:":
        normalized = normalized[:-1].rstrip()
    return normalized


def _is_ultra_short_tts_text(text: str) -> bool:
    """True when CosyVoice is at high risk of empty-mel / kernel>input.

    Threshold (fix_c7_short_text_pad_v1): alphanumeric+space strip length < 8
    OR whitespace word count <= 2 after collapse. Matches backchannels like
    ``Mm-hm.`` / ``Oh.`` / ``Ok.`` / ``Bye.``.
    """
    normalized = " ".join(str(text).split())
    if not normalized:
        return True
    words = normalized.split()
    alnum_space = "".join(
        ch for ch in normalized if ch.isalnum() or ch.isspace()
    ).strip()
    return len(alnum_space) < 8 or len(words) <= 2


def _pad_ultra_short_tts_text(text: str) -> tuple[str, dict[str, Any]]:
    """Return (synth_text, pad_meta). Non-short inputs pass through unchanged."""
    original = str(text)
    meta: dict[str, Any] = {
        "padded": False,
        "policy": "fix_c7_short_text_pad_v1",
        "original_text": original,
    }
    if not _is_ultra_short_tts_text(original):
        meta["padded_text"] = original
        return original, meta
    key = _normalize_short_backchannel_key(original)
    if key in _SHORT_BACKCHANNEL_ALLOWLIST:
        padded = _SHORT_BACKCHANNEL_ALLOWLIST[key]
        meta.update(
            {
                "padded": True,
                "pad_mode": "allowlist",
                "pad_key": key,
                "padded_text": padded,
            }
        )
        return padded, meta
    base = original.rstrip()
    if base and base[-1] not in ".!?":
        base = f"{base}."
    padded = f"{base} Okay."
    meta.update(
        {
            "padded": True,
            "pad_mode": "generic_min_len",
            "pad_key": key,
            "padded_text": padded,
        }
    )
    return padded, meta



class CosyVoiceTTS:
    """CosyVoice 3 0.5B zero-shot renderer with deterministic cache keys.

    A catalog entry supplies a short reference clip and its transcript.  The
    model is loaded lazily so importing the benchmark or running unit tests
    does not require the CosyVoice environment.  The adapter intentionally
    does not make network calls.
    """

    sample_rate = 16_000
    # Bump this whenever the immutable voice/persona rendering contract
    # changes.  It prevents pre-profile cache entries from being reused.
    renderer_version = "fun-cosyvoice3-0.5b-zero-shot-persona-v2"
    # Short utterances are common in duplex interaction (backchannels,
    # confirmations, corrections).  Keep a real non-silent check, but do not
    # reject a valid one-word utterance merely because the renderer produced
    # less than the legacy 250 ms floor.
    validation_version = "duration-nonsilent-gate-v3-relaxed-retry3"
    _model_cache: dict[tuple[str, str | None], tuple[Any, Any]] = {}
    _model_cache_lock = threading.Lock()
    _loaded_feature_files: set[tuple[str, str]] = set()
    _inference_lock = threading.Lock()

    def __init__(
        self,
        *,
        model_dir: Path,
        catalog_path: Path,
        binding_manifest_path: Path | None = None,
        speaker_feature_cache_path: Path | None = None,
        case_id: str,
        assignment_seed: int,
        cache_dir: Path | None = None,
        device: str | None = None,
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.catalog_path = catalog_path.expanduser().resolve()
        self.catalog = VoiceCatalogManifest.load(self.catalog_path)
        self.binding_manifest_path = (
            binding_manifest_path.expanduser().resolve()
            if binding_manifest_path is not None
            else None
        )
        self.binding_manifest = (
            GlobalVoiceBindingManifest.load(self.binding_manifest_path)
            if self.binding_manifest_path is not None
            else None
        )
        if self.binding_manifest is not None:
            catalog_sha256 = hashlib.sha256(self.catalog_path.read_bytes()).hexdigest()
            if catalog_sha256 != self.binding_manifest.catalog_sha256:
                raise ValueError(
                    "global voice binding catalog_sha256 does not match the supplied catalog"
                )
        self.speaker_feature_cache_path = (
            speaker_feature_cache_path.expanduser().resolve()
            if speaker_feature_cache_path is not None
            else None
        )
        self.case_id = case_id
        self.assignment_seed = assignment_seed
        self.cache_dir = cache_dir.expanduser().resolve() if cache_dir else None
        self.device = device
        self._model: Any | None = None
        self._load_wav: Any | None = None
        self._entry_by_voice_id: dict[str, VoiceCatalogEntry] = {}
        self._profile_by_voice_id: dict[str, Any] = {}
        self._binding_profile_by_voice_id: dict[str, dict[str, object]] = {}

    def register_profiles(self, users: list[Any]) -> None:
        if not users:
            raise ValueError("at least one user is required for local TTS")
        user_ids = [user.user_id for user in users]
        assignment = (
            self.binding_manifest.assignment_for_case(
                case_id=self.case_id, users=users, catalog=self.catalog
            )
            if self.binding_manifest is not None
            else self.catalog.assign(self.case_id, user_ids, self.assignment_seed)
        )
        self._entry_by_voice_id = {
            user.voice.voice_id: assignment[user.user_id] for user in users
        }
        self._profile_by_voice_id = {
            user.voice.voice_id: user for user in users
        }
        binding_by_user = {
            item.oracle_user_id: item
            for item in (self.binding_manifest.bindings if self.binding_manifest else [])
            if item.case_id == self.case_id
        }
        self._binding_profile_by_voice_id = {
            user.voice.voice_id: dict(
                getattr(binding_by_user.get(user.user_id), "persona_voice_profile", {})
            )
            for user in users
        }

    def assigned_catalog_entries(self) -> dict[str, VoiceCatalogEntry]:
        return dict(self._entry_by_voice_id)

    @staticmethod
    def _prompt_text(reference_transcript: str) -> str:
        """Format a catalog transcript for the CosyVoice3 prompt contract."""
        if "<|endofprompt|>" in reference_transcript:
            return reference_transcript
        return "You are a helpful assistant.<|endofprompt|>" + reference_transcript

    def _reference_path(self, entry: VoiceCatalogEntry) -> Path:
        reference = Path(entry.reference_audio_path).expanduser()
        if not reference.is_absolute():
            reference = self.catalog_path.parent / reference
        reference = reference.resolve()
        if not reference.is_file():
            raise FileNotFoundError(
                f"voice catalog reference audio is missing: {reference}"
            )
        return reference

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"CosyVoice model directory is missing: {self.model_dir}")
        _install_offline_wetext_shim()
        try:
            from cosyvoice.cli.cosyvoice import AutoModel
            from cosyvoice.utils.file_utils import load_wav
        except ImportError as exc:
            raise RuntimeError(
                "CosyVoiceTTS requires the CosyVoice environment; install the "
                "official repository and run this backend on its GPU worker"
            ) from exc

        def load_wav_compat(path: str, target_rate: int) -> Any:
            return _load_wav_with_torchcodec_fallback(load_wav, path, target_rate)

        # ``frontend.py`` imports ``load_wav`` into its own module namespace,
        # so replacing only the callable stored by this backend would not
        # affect ``add_zero_shot_spk``. Patch that alias after the official
        # modules are loaded; the wrapper still delegates to the official
        # implementation whenever TorchCodec is available.
        try:
            import cosyvoice.cli.frontend as cosyvoice_frontend

            cosyvoice_frontend.load_wav = load_wav_compat
        except ImportError:
            pass

        cache_key = (str(self.model_dir), self.device)
        with self._model_cache_lock:
            cached = self._model_cache.get(cache_key)
            if cached is None:
                modern_kwargs: dict[str, Any] = {
                    "model_dir": str(self.model_dir),
                    "load_trt": False,
                    "load_vllm": False,
                    "fp16": True,
                }
                if self.device is not None:
                    modern_kwargs["device"] = self.device
                legacy_kwargs: dict[str, Any] = {
                    "model_dir": str(self.model_dir),
                    "load_jit": False,
                    "load_trt": False,
                    "fp16": True,
                }
                if self.device is not None:
                    legacy_kwargs["device"] = self.device
                candidates = [modern_kwargs]
                if self.device is not None:
                    # Official CosyVoice3 selects CUDA internally and rejects
                    # an explicit ``device`` keyword. Preserve fp16 instead of
                    # falling all the way back to the minimal constructor.
                    candidates.append(
                        {key: value for key, value in modern_kwargs.items() if key != "device"}
                    )
                candidates.append(legacy_kwargs)
                if self.device is not None:
                    candidates.append(
                        {key: value for key, value in legacy_kwargs.items() if key != "device"}
                    )
                candidates.append({"model_dir": str(self.model_dir)})
                last_error: TypeError | None = None
                for candidate in candidates:
                    try:
                        model = AutoModel(**candidate)
                        break
                    except TypeError as exc:
                        last_error = exc
                else:  # pragma: no cover - minimal constructor is expected to work
                    assert last_error is not None
                    raise last_error
                cached = (model, load_wav_compat)
                self._model_cache[cache_key] = cached
            self._model, self._load_wav = cached
            if self.speaker_feature_cache_path is not None:
                feature_key = (str(self.model_dir), str(self.speaker_feature_cache_path))
                if feature_key not in self._loaded_feature_files:
                    if not self.speaker_feature_cache_path.is_file():
                        raise FileNotFoundError(
                            "CosyVoice speaker feature cache is missing: "
                            f"{self.speaker_feature_cache_path}"
                        )
                    import torch

                    features = torch.load(
                        self.speaker_feature_cache_path,
                        map_location=self._model.frontend.device,
                        weights_only=True,
                    )
                    self._model.frontend.spk2info.update(features)
                    self._loaded_feature_files.add(feature_key)

    def _feature_id(self, entry: VoiceCatalogEntry) -> str:
        return "mubench_" + hashlib.sha256(
            f"{self.catalog.catalog_version}:{entry.voice_catalog_id}:{entry.source_sha256}".encode()
        ).hexdigest()[:32]

    @staticmethod
    def _trim_edge_silence(pcm: bytes, sample_rate: int) -> bytes:
        frame_bytes = sample_rate * 2 // 50
        frames = [pcm[i : i + frame_bytes] for i in range(0, len(pcm), frame_bytes)]
        voiced = [
            i
            for i, frame in enumerate(frames)
            if len(frame) == frame_bytes
            and np.max(np.abs(np.frombuffer(frame, dtype="<i2"))) >= 120
        ]
        if not voiced:
            return pcm
        first = max(0, voiced[0] - 5)
        last = min(len(frames), voiced[-1] + 6)
        return b"".join(frames[first:last])

    @staticmethod
    def _apply_persona_voice_profile(
        pcm: bytes, sample_rate: int, profile: dict[str, object]
    ) -> bytes:
        """Apply small deterministic rate/energy cues after zero-shot TTS.

        This intentionally uses only NumPy interpolation so the profile works
        in the lean CosyVoice worker image.  The source reference remains the
        dominant speaker cue; bounded rate and gain make assertive/cooperative
        behavior audible without pretending to synthesize a new identity.
        """

        if not pcm:
            return pcm
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        speed = float(profile.get("render_speed", 1.0) or 1.0)
        speed = max(0.92, min(1.12, speed))
        if abs(speed - 1.0) > 1e-4 and samples.size > 1:
            target_size = max(1, round(samples.size / speed))
            source_axis = np.linspace(0.0, 1.0, samples.size, endpoint=False)
            target_axis = np.linspace(0.0, 1.0, target_size, endpoint=False)
            samples = np.interp(target_axis, source_axis, samples).astype(np.float32)
        gain_db = max(-2.0, min(2.0, float(profile.get("gain_db", 0.0) or 0.0)))
        if abs(gain_db) > 1e-4:
            samples *= 10.0 ** (gain_db / 20.0)
        return np.clip(samples, -32768, 32767).astype("<i2").tobytes()

    def _render(
        self,
        text: str,
        entry: VoiceCatalogEntry,
        *,
        enforce_validation: bool = True,
        synthesis_attempt: int = 1,
    ) -> tuple[bytes, dict[str, Any]]:
        self._ensure_model()
        assert self._model is not None
        reference_path = self._reference_path(entry)
        prompt_text = self._prompt_text(entry.reference_transcript)
        feature_id = self._feature_id(entry)
        if feature_id not in set(self._model.list_available_spks()):
            self._model.add_zero_shot_spk(
                prompt_text, str(reference_path), feature_id
            )
        # Seed stays bound to the API/request text so cache keys and fail-triple
        # reproduction remain stable; CosyVoice only sees synth_text (may pad).
        seed_input = f"{self.catalog.catalog_version}:{entry.voice_catalog_id}:{text}"
        if synthesis_attempt > 1:
            seed_input += f":retry:{synthesis_attempt}"
        synthesis_seed = int.from_bytes(
            hashlib.sha256(
                seed_input.encode()
            ).digest()[:4],
            "big",
        )
        synth_text, short_pad_meta = _pad_ultra_short_tts_text(text)
        if short_pad_meta.get("padded"):
            text_sha = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
            print(
                f"[tts_backends] short_text_pad voice={entry.voice_catalog_id} "
                f"text_sha256={text_sha} mode={short_pad_meta.get('pad_mode')} "
                f"policy={short_pad_meta.get('policy')} "
                f"padded_text={short_pad_meta.get('padded_text')!r}",
                flush=True,
            )
        with self._inference_lock:
            # Freeze stochastic speech-token sampling for rebuildability. Final
            # evaluation still replays the content-addressed WAV, so the exact
            # bytes do not depend on live model sampling or GPU scheduling.
            random.seed(synthesis_seed)
            np.random.seed(synthesis_seed)
            try:
                import torch

                torch.manual_seed(synthesis_seed)
                torch.cuda.manual_seed_all(synthesis_seed)
            except ImportError:
                pass
            try:
                generated = self._model.inference_zero_shot(
                    synth_text,
                    prompt_text,
                    str(reference_path),
                    zero_shot_spk_id=feature_id,
                    stream=False,
                )
            except TypeError:
                assert self._load_wav is not None
                prompt_speech = _load_wav_with_torchcodec_fallback(
                    self._load_wav, str(reference_path), self.sample_rate
                )
                generated = self._model.inference_zero_shot(
                    synth_text,
                    entry.reference_transcript,
                    prompt_speech,
                )
            first = next(iter(generated), None)
        if first is None or "tts_speech" not in first:
            raise RuntimeError("CosyVoice returned no tts_speech for the requested line")
        pcm = _waveform_to_pcm16(
            first["tts_speech"],
            int(getattr(self._model, "sample_rate", 24_000)),
            self.sample_rate,
        )
        pcm = self._trim_edge_silence(pcm, self.sample_rate)
        profile = self._binding_profile_by_voice_id.get(entry.voice_catalog_id)
        # ``_binding_profile_by_voice_id`` is keyed by benchmark voice ID.  A
        # reverse lookup keeps this compatible with catalog entries that are
        # used in the legacy smoke path without a binding manifest.
        if profile is None:
            for voice_id, candidate in self._entry_by_voice_id.items():
                if candidate.voice_catalog_id == entry.voice_catalog_id:
                    profile = self._binding_profile_by_voice_id.get(voice_id)
                    break
        if profile:
            pcm = self._apply_persona_voice_profile(pcm, self.sample_rate, profile)
        normalized_text = " ".join(str(text).split())
        words = max(1, len(normalized_text.split()))
        duration_seconds = len(pcm) / (self.sample_rate * 2)
        samples = np.frombuffer(pcm, dtype="<i2") if pcm else np.empty(0, dtype=np.int16)
        peak_amplitude = int(np.max(np.abs(samples))) if samples.size else 0
        rms_amplitude = (
            float(np.sqrt(np.mean(np.square(samples.astype(np.float32)))))
            if samples.size
            else 0.0
        )
        # Benchmark-specific heuristic, not an official CosyVoice rule.
        # Relax the lower bound by about 20 percent for fast/short speech;
        # preserve the upper bound and the non-empty/non-silent checks.
        short_utterance = words <= 2 or len(normalized_text) <= 12
        if words <= 1:
            min_duration_seconds = 0.08
        elif short_utterance:
            min_duration_seconds = 0.11
        else:
            min_duration_seconds = max(0.20, words * 0.065)
        max_duration_seconds = max(7.0, words * 1.2 + 2.0)
        non_silent = peak_amplitude >= 8 and rms_amplitude >= 2.0
        duration_accepted = min_duration_seconds <= duration_seconds <= max_duration_seconds
        hard_valid = bool(pcm) and non_silent
        accepted = hard_valid and duration_accepted
        validation = {
            "accepted": accepted,
            "raw_accepted": accepted,
            "duration_accepted": duration_accepted,
            "hard_valid": hard_valid,
            "validation_version": self.validation_version,
            "expected_words": words,
            "duration_seconds": round(duration_seconds, 3),
            "min_duration_seconds": min_duration_seconds,
            "max_duration_seconds": max_duration_seconds,
            "short_utterance": short_utterance,
            "peak_amplitude": peak_amplitude,
            "rms_amplitude": round(rms_amplitude, 2),
            "non_silent": non_silent,
            "content_validation": "duration_relaxed_retry3_nonsilent_v3",
            "synthesis_seed": synthesis_seed,
            "synthesis_attempt": synthesis_attempt,
            "short_text_pad": short_pad_meta,
        }
        if not validation["accepted"] and enforce_validation:
            raise RuntimeError(f"CosyVoice duration validation failed: {validation}")
        return pcm, validation

    def synthesize(self, text: str, voice_id: str) -> bytes:
        if voice_id not in self._entry_by_voice_id:
            raise ValueError(f"voice profile was not registered: {voice_id}")
        entry = self._entry_by_voice_id[voice_id]
        profile = self._profile_by_voice_id.get(voice_id)
        binding_profile = self._binding_profile_by_voice_id.get(voice_id, {})
        style = {
            "accent": getattr(getattr(profile, "voice", None), "accent_label", None),
            "speaking_style": getattr(
                getattr(profile, "voice", None), "speaking_style", None
            ),
            "persona_voice_profile": binding_profile,
        }
        reference_path = self._reference_path(entry)
        cache_path = metadata_path = lock_path = None
        if self.cache_dir is not None:
            key_payload = {
                "renderer_version": self.renderer_version,
                "validation_version": self.validation_version,
                "model_dir": str(self.model_dir),
                "catalog_version": self.catalog.catalog_version,
                "voice_catalog_id": entry.voice_catalog_id,
                "reference_audio": str(reference_path),
                "reference_sha256": entry.source_sha256,
                "text": text,
                "style": style,
                "sample_rate": self.sample_rate,
            }
            cache_key = hashlib.sha256(
                json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"{cache_key}.pcm"
            metadata_path = self.cache_dir / f"{cache_key}.json"
            lock_path = self.cache_dir / f".{cache_key}.lock"

        def render_and_store() -> bytes:
            validation: dict[str, Any] | None = None
            pcm: bytes = b""
            attempt_history: list[dict[str, Any]] = []
            zero_token_retry_from: int | None = None
            kernel_empty_mel = "Kernel size can't be greater than actual input size"
            for attempt in range(1, 4):
                try:
                    pcm, validation = self._render(
                        text, entry, enforce_validation=False,
                        synthesis_attempt=attempt,
                    )
                except RuntimeError as exc:
                    # Secondary to Fix-C7-B pad: keep A-style seed retry if pad
                    # still hits empty-mel on a given seed.
                    if kernel_empty_mel not in str(exc):
                        raise
                    text_sha = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
                    print(
                        f"[tts_backends] zero_token_retry voice={entry.voice_catalog_id} "
                        f"text_sha256={text_sha} attempt={attempt} "
                        f"reason=kernel_size_empty_mel",
                        flush=True,
                    )
                    attempt_history.append(
                        {
                            "synthesis_attempt": attempt,
                            "reason": "kernel_size_empty_mel",
                            "error": str(exc)[:300],
                        }
                    )
                    zero_token_retry_from = attempt
                    if attempt < 3:
                        continue
                    raise
                attempt_history.append({**validation, "pcm_sha256": hashlib.sha256(pcm).hexdigest()})
                # Duration rejection retries with :retry:N seeds. Pad is primary;
                # empty-mel seed retries are secondary (Fix-C7-A keep).
                if not validation["hard_valid"]:
                    raise RuntimeError(f"CosyVoice non-empty/non-silent validation failed: {validation}")
                validation["tts_attempt"] = attempt
                if zero_token_retry_from is not None:
                    validation["zero_token_retry"] = attempt
                if not validation["duration_accepted"] and attempt < 3:
                    continue
                if attempt == 3 and not validation["duration_accepted"]:
                    validation["forced_duration_accept"] = True
                    validation["validation_override_reason"] = (
                        "third_attempt_forced_duration_accept"
                    )
                validation["effective_accepted"] = True
                validation["attempt_history"] = attempt_history
                break
            if validation is None:
                raise RuntimeError("CosyVoice did not produce a validation record")
            if cache_path is not None and metadata_path is not None:
                temporary = cache_path.with_suffix(f".pcm.tmp.{os.getpid()}")
                temporary.write_bytes(pcm)
                os.replace(temporary, cache_path)
                _write_json(
                    metadata_path,
                    {
                        "renderer_version": self.renderer_version,
                        "model_dir": str(self.model_dir),
                        "catalog_version": self.catalog.catalog_version,
                        "voice_catalog_id": entry.voice_catalog_id,
                        "voice_id": voice_id,
                        "reference_audio_path": str(reference_path),
                        "validation": validation,
                    },
                )
            return pcm

        if lock_path is None:
            return render_and_store()
        with lock_path.open("a+") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                if cache_path is not None and metadata_path is not None:
                    if cache_path.exists() and metadata_path.exists():
                        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                        if (
                            metadata.get("renderer_version") == self.renderer_version
                            and metadata.get("validation", {}).get("validation_version") == self.validation_version
                            and metadata.get("validation", {}).get("effective_accepted") is True
                            and metadata.get("validation", {}).get("hard_valid") is True
                        ):
                            return cache_path.read_bytes()
                return render_and_store()
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def provenance(self) -> dict[str, Any]:
        return {
            "source": "local_zero_shot_tts",
            "backend": "cosyvoice",
            "renderer_version": self.renderer_version,
            "validation_version": self.validation_version,
            "model_dir": str(self.model_dir),
            "catalog_path": str(self.catalog_path),
            "catalog_version": self.catalog.catalog_version,
            "binding_manifest_path": (
                str(self.binding_manifest_path) if self.binding_manifest_path else None
            ),
            "binding_version": (
                self.binding_manifest.binding_version if self.binding_manifest else None
            ),
            "speaker_feature_cache_path": (
                str(self.speaker_feature_cache_path)
                if self.speaker_feature_cache_path
                else None
            ),
            "assigned_voice_catalog_ids": {
                voice_id: entry.voice_catalog_id
                for voice_id, entry in sorted(self._entry_by_voice_id.items())
            },
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "content_validation": "duration_relaxed_retry3_nonsilent_v3",
            "duration_retry_policy": "three_total_attempts_then_use_third_waveform",
            "text_frontend_policy": (
                "offline_passthrough"
                if os.environ.get("MU_BENCH_COSYVOICE_OFFLINE_TEXT", "1") == "1"
                else "official_default"
            ),
            "short_text_pad_policy": "fix_c7_short_text_pad_v1",
            "persona_profile_rendering": "bounded_rate_gain_v1",
            "requires_asr_gate_for_leaderboard": True,
            "leaderboard_eligible": False,
        }


def encode_reference_audio(path: Path) -> str:
    """Small helper for HTTP TTS adapters and catalog inspection tools."""

    return base64.b64encode(path.read_bytes()).decode("ascii")
