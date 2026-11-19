"""Curated speaker-reference catalog and deterministic case assignment.

The catalog deliberately stores pseudonymous source speaker IDs rather than
human names.  A benchmark voice is a licensed reference recording plus
provenance; it is not an assertion about the identity of the person in the
source corpus.  The same catalog can therefore be reused across cases while a
Latin-square style assignment prevents voice timbre from becoming a proxy for
role or permission.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class VoiceCatalogSource(BaseModel):
    """A public corpus or an explicitly consented private reference pool."""

    source_id: str
    name: str
    version: str
    homepage: str
    license: str
    # Public corpus pages do not all publish a stable speaker count.  Keep it
    # optional instead of inventing a number; a built catalog's ``voices`` list
    # is the authoritative count for the exact release used by μ-Bench.
    speaker_count: int | None = Field(default=None, gt=0)
    locales: list[str] = Field(default_factory=list)
    accent_coverage: list[str] = Field(default_factory=list)
    audio_profile: Literal["studio", "read_speech", "crowdsourced", "mixed"]
    identity_policy: Literal[
        "pseudonymous_source_ids",
        "do_not_identify_source_speakers",
        "explicit_consent"
    ]
    usable_for_voice_cloning: bool = False
    attribution: str | None = None
    notes: str | None = None


class VoiceCatalogEntry(BaseModel):
    """One reference recording usable by a local zero-shot TTS backend."""

    voice_catalog_id: str
    source_id: str
    source_speaker_id: str
    reference_audio_path: str
    reference_transcript: str
    locale: str
    accent: str
    gender_presentation: Literal["feminine", "masculine", "androgynous", "unknown"] = (
        "unknown"
    )
    age_band: Literal["young_adult", "adult", "older_adult", "unknown"] = "unknown"
    style_tags: list[str] = Field(default_factory=list)
    license_id: str
    consent_record_id: str | None = None
    source_sha256: str | None = None
    enabled: bool = True


class VoiceCatalogManifest(BaseModel):
    """Versioned index consumed by the TTS renderer and runner."""

    schema_version: Literal["1.0"] = "1.0"
    catalog_version: str
    sources: list[VoiceCatalogSource]
    voices: list[VoiceCatalogEntry]
    selection_policy: str = "deterministic_case_hash_then_role_balanced"

    @model_validator(mode="after")
    def validate_catalog(self) -> VoiceCatalogManifest:
        source_ids = {source.source_id for source in self.sources}
        if len(source_ids) != len(self.sources):
            raise ValueError("voice catalog source IDs must be unique")
        voice_ids = [voice.voice_catalog_id for voice in self.voices]
        if len(voice_ids) != len(set(voice_ids)):
            raise ValueError("voice catalog IDs must be unique")
        unknown_sources = {
            voice.source_id for voice in self.voices if voice.source_id not in source_ids
        }
        if unknown_sources:
            raise ValueError(f"voices reference unknown sources: {sorted(unknown_sources)}")
        for voice in self.voices:
            source = next(item for item in self.sources if item.source_id == voice.source_id)
            if not source.usable_for_voice_cloning:
                raise ValueError(
                    f"source {voice.source_id} is not marked usable_for_voice_cloning"
                )
            if source.identity_policy == "do_not_identify_source_speakers":
                # The source may still be used for a pseudonymous reference only
                # when the release explicitly keeps the source ID non-identifying.
                if not voice.source_speaker_id or voice.source_speaker_id.lower() in {
                    "name",
                    "real_name",
                    "full_name",
                }:
                    raise ValueError(
                        f"source {voice.source_id} requires a pseudonymous speaker ID"
                    )
        return self

    @classmethod
    def load(cls, path: Path) -> VoiceCatalogManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def enabled_entries(self) -> list[VoiceCatalogEntry]:
        return [voice for voice in self.voices if voice.enabled]

    def assign(self, case_id: str, user_ids: list[str], seed: int) -> dict[str, VoiceCatalogEntry]:
        """Assign distinct references within a case, deterministically.

        Assignment is intentionally independent of role, display name and
        permissions.  Hashing the case ID and seed rotates the starting point;
        this avoids teaching the SUT that ``primary`` or a particular voice
        pitch implies authority.
        """

        if len(user_ids) != len(set(user_ids)):
            raise ValueError("user IDs must be unique for voice assignment")
        candidates = sorted(self.enabled_entries(), key=lambda item: item.voice_catalog_id)
        if len(candidates) < len(user_ids):
            raise ValueError(
                f"voice catalog has {len(candidates)} enabled references, "
                f"but the case needs {len(user_ids)}"
            )
        digest = hashlib.sha256(f"{self.catalog_version}:{case_id}:{seed}".encode()).digest()
        offset = int.from_bytes(digest[:8], "big") % len(candidates)
        rotated = candidates[offset:] + candidates[:offset]
        # A local RNG makes the optional interleaving explicit and reproducible.
        # It is not seeded by role or user name.
        rng = random.Random(int.from_bytes(digest[8:16], "big"))
        stride = 1
        if len(candidates) > 1:
            stride = rng.randrange(1, len(candidates))
            while math.gcd(stride, len(candidates)) != 1:
                stride = (stride + 1) % len(candidates) or 1
        ordered = [rotated[(index * stride) % len(candidates)] for index in range(len(user_ids))]
        return dict(zip(user_ids, ordered, strict=True))

    def provenance(self) -> dict[str, object]:
        """Return non-biometric catalog provenance for run artifacts."""

        return {
            "source": "versioned_reference_voice_catalog",
            "schema_version": self.schema_version,
            "catalog_version": self.catalog_version,
            "selection_policy": self.selection_policy,
            "source_ids": sorted(source.source_id for source in self.sources),
            "enabled_voice_count": len(self.enabled_entries()),
        }


class GlobalVoiceBindingEntry(BaseModel):
    """Immutable one-to-one binding between one case participant and a voice."""

    case_id: str
    oracle_user_id: str
    public_participant_id: str
    display_name: str
    role: str
    voice_id: str
    voice_catalog_id: str
    source_id: str
    source_speaker_id: str
    reference_audio_sha256: str | None = None
    locale: str
    accent: str
    gender_presentation: str = "unknown"
    age_band: str = "unknown"
    # Observed catalog attributes plus deterministic persona-derived acoustic
    # controls.  Keeping this in the immutable binding (rather than deriving
    # it inside a runner) makes every replay use exactly the same voice
    # characteristics and gives evaluators an auditable source of variation.
    persona_voice_profile: dict[str, object] = Field(default_factory=dict)


class GlobalVoiceBindingManifest(BaseModel):
    """Frozen globally unique voice allocation for an entire benchmark release."""

    # 1.0 remains readable for the pre-persona binding release; 1.1 freezes
    # the persona-derived acoustic profile in every row.
    schema_version: Literal["1.0", "1.1"] = "1.1"
    binding_version: str
    benchmark_version: str
    catalog_version: str
    assignment_seed: int
    assignment_algorithm: str = "global_unique_sha256_permutation_v1"
    catalog_sha256: str
    case_count: int = Field(gt=0)
    participant_slot_count: int = Field(gt=0)
    speaker_embedding_artifact_sha256: str | None = None
    perceptual_separation_policy: str = "not_measured"
    separation_summary: dict[str, float | int | str] = Field(default_factory=dict)
    bindings: list[GlobalVoiceBindingEntry]

    @model_validator(mode="after")
    def validate_bindings(self) -> GlobalVoiceBindingManifest:
        keys = [(item.case_id, item.oracle_user_id) for item in self.bindings]
        if len(keys) != len(set(keys)):
            raise ValueError("case/user binding keys must be globally unique")
        voice_ids = [item.voice_catalog_id for item in self.bindings]
        if len(voice_ids) != len(set(voice_ids)):
            raise ValueError("voice_catalog_id cannot be reused across benchmark cases")
        source_speakers = [
            (item.source_id, item.source_speaker_id) for item in self.bindings
        ]
        if len(source_speakers) != len(set(source_speakers)):
            raise ValueError("source speakers cannot be reused across benchmark cases")
        if len(self.bindings) != self.participant_slot_count:
            raise ValueError("participant_slot_count does not match bindings")
        if len({item.case_id for item in self.bindings}) != self.case_count:
            raise ValueError("case_count does not match bindings")
        return self

    @classmethod
    def load(cls, path: Path) -> GlobalVoiceBindingManifest:
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def dump_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [item.model_dump(mode="json") for item in self.bindings]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def assignment_for_case(
        self,
        *,
        case_id: str,
        users: list[object],
        catalog: VoiceCatalogManifest,
    ) -> dict[str, VoiceCatalogEntry]:
        if catalog.catalog_version != self.catalog_version:
            raise ValueError(
                f"binding catalog version {self.catalog_version} does not match "
                f"catalog {catalog.catalog_version}"
            )
        catalog_by_id = {item.voice_catalog_id: item for item in catalog.voices}
        case_bindings = {
            item.oracle_user_id: item for item in self.bindings if item.case_id == case_id
        }
        user_ids = [str(getattr(user, "user_id")) for user in users]
        if set(case_bindings) != set(user_ids):
            raise ValueError(
                f"global voice binding does not exactly cover {case_id}: "
                f"expected {sorted(user_ids)}, found {sorted(case_bindings)}"
            )
        missing = {
            item.voice_catalog_id
            for item in case_bindings.values()
            if item.voice_catalog_id not in catalog_by_id
        }
        if missing:
            raise ValueError(f"global voice binding references missing voices: {sorted(missing)}")
        mismatched_hashes = {
            item.voice_catalog_id
            for item in case_bindings.values()
            if item.reference_audio_sha256
            and catalog_by_id[item.voice_catalog_id].source_sha256
            and item.reference_audio_sha256
            != catalog_by_id[item.voice_catalog_id].source_sha256
        }
        if mismatched_hashes:
            raise ValueError(
                "global voice binding reference hashes do not match catalog: "
                f"{sorted(mismatched_hashes)}"
            )
        return {
            user_id: catalog_by_id[case_bindings[user_id].voice_catalog_id]
            for user_id in user_ids
        }


def builtin_source_registry() -> VoiceCatalogManifest:
    """Return the recommended public source registry without audio paths.

    This is a registry, not a downloaded biometric bundle.  A release builder
    must add actual reference clips and hashes after checking the source terms.
    Common Voice is intentionally diagnostic-only because its data card asks
    users not to determine contributor identity.
    """

    return VoiceCatalogManifest(
        catalog_version="public-sources-v1",
        sources=[
            VoiceCatalogSource(
                source_id="vctk-0.92",
                name="CSTR VCTK Corpus",
                version="0.92",
                homepage="https://datashare.ed.ac.uk/handle/10283/3443",
                license="CC-BY-4.0",
                speaker_count=109,
                locales=["en-GB", "en-US"],
                accent_coverage=["British", "Irish", "Scottish", "American"],
                audio_profile="studio",
                identity_policy="pseudonymous_source_ids",
                usable_for_voice_cloning=True,
                attribution="Veaux, Yamagishi and MacDonald, CSTR VCTK Corpus",
            ),
            VoiceCatalogSource(
                source_id="libritts-r",
                name="LibriTTS-R",
                version="1.0",
                homepage="https://www.openslr.org/141/",
                license="CC-BY-4.0",
                speaker_count=2456,
                locales=["en-US"],
                accent_coverage=["English read speech"],
                audio_profile="read_speech",
                identity_policy="pseudonymous_source_ids",
                usable_for_voice_cloning=True,
                attribution="Koizumi et al., LibriTTS-R",
            ),
            VoiceCatalogSource(
                source_id="multilingual-librispeech",
                name="Multilingual LibriSpeech",
                version="1.0",
                homepage="https://www.openslr.org/94/",
                license="CC-BY-4.0",
                speaker_count=None,
                locales=["en", "de", "nl", "es", "fr", "it", "pt", "pl"],
                accent_coverage=["multilingual read speech"],
                audio_profile="read_speech",
                identity_policy="pseudonymous_source_ids",
                usable_for_voice_cloning=True,
                attribution="Pratap et al., Multilingual LibriSpeech",
            ),
            VoiceCatalogSource(
                source_id="globe-1.0",
                name="GLOBE",
                version="1.0",
                homepage="https://huggingface.co/datasets/MushanW/GLOBE",
                license="CC0-1.0",
                speaker_count=23519,
                locales=["en"],
                accent_coverage=["164 self-reported English accent labels"],
                audio_profile="crowdsourced",
                identity_policy="pseudonymous_source_ids",
                usable_for_voice_cloning=True,
                attribution="Wang et al., GLOBE: A High-quality English Corpus for Zero-shot Speaker Adaptive Text-to-Speech",
                notes="Select one quality-gated reference per pseudonymous speaker.",
            ),
            VoiceCatalogSource(
                source_id="common-voice-diagnostic",
                name="Mozilla Common Voice",
                version="release-selected",
                homepage="https://commonvoice.mozilla.org/data",
                license="CC0-1.0",
                speaker_count=None,
                locales=["multilingual"],
                accent_coverage=["community accents"],
                audio_profile="crowdsourced",
                identity_policy="do_not_identify_source_speakers",
                usable_for_voice_cloning=False,
                notes="Use only for accent/noise diagnostics unless a separate consented release is obtained.",
            ),
        ],
        voices=[],
    )
