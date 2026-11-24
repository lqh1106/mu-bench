"""Evidence boundaries shared by closed and open-weight logical sessions.

ASR is a fallible measurement, not an oracle. These checks certify provenance
and time bounds only; they do not certify the recognized words as correct.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

INTEGRITY_VERSION = "mu-bench-v14.2-integrity-rc1"


class OutputArtifactConflictError(ValueError):
    """A refused attempt must not write an error into another attempt."""


class TranscriptEvidenceInvalid(ValueError):
    reason = "transcript_evidence_invalid"

    def __init__(self, issue: str):
        self.issues = [{"code": issue}]
        super().__init__(issue)

    def to_dict(self):
        return {"reason": self.reason, "issues": self.issues}


def _time(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TranscriptEvidenceInvalid("invalid_" + name)
    if not math.isfinite(value) or value < 0:
        raise TranscriptEvidenceInvalid("invalid_" + name)
    return float(value)


def played_assistant_evidence(result: dict, *, task_start_ms: int,
                              session_end_ms: int) -> list[dict]:
    """Assign stable IDs without reconstructing words from generated deltas.

    Only the assistant's isolated, already played waveform is admissible.
    Fragment and context ranges use session time; returned evidence uses task
    time. No word timestamp is fabricated from a proportional character count.
    """
    if (not isinstance(result, dict) or result.get("asr_derived") is not True
            or result.get("asr_transcript_scope") != "task_only_actual_assistant_waveform"):
        raise TranscriptEvidenceInvalid("untrusted_transcript_source")
    if result.get("status") == "asr_failed":
        raise TranscriptEvidenceInvalid("asr_failed_not_empty_speech")
    start = _time(task_start_ms, "task_start")
    frontier = _time(session_end_ms, "played_frontier")
    if frontier < start:
        raise TranscriptEvidenceInvalid("task_outside_played_frontier")
    fragments = result.get("assistant_fragments")
    if not isinstance(fragments, list):
        raise TranscriptEvidenceInvalid("missing_assistant_fragments")
    turns, prior_end, seen = [], start, set()
    for item in fragments:
        if (not isinstance(item, dict)
                or item.get("source") != "audible_assistant_asr"
                or item.get("asr_derived") is not True
                or item.get("status") not in {"transcribed", "unintelligible"}):
            raise TranscriptEvidenceInvalid("non_assistant_or_unvalidated_fragment")
        status = item.get("status")
        a = _time(item.get("start_ms"), "fragment_start")
        b = _time(item.get("end_ms"), "fragment_end")
        ca = _time(item.get("context_start_ms"), "context_start")
        cb = _time(item.get("context_end_ms"), "context_end")
        if not (start <= ca <= a < b <= cb <= frontier and prior_end <= a):
            raise TranscriptEvidenceInvalid("fragment_outside_played_task_or_overlapping")
        text, digest = item.get("text"), item.get("audio_sha256")
        if not isinstance(text, str) or not text.strip():
            raise TranscriptEvidenceInvalid("missing_transcribed_text")
        if not isinstance(digest, str) or not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise TranscriptEvidenceInvalid("missing_audio_digest")
        if status == "unintelligible":
            # Neutral marker only; rejected hallucination must not be the visible text.
            if not text.startswith("[unintelligible audio, segment "):
                raise TranscriptEvidenceInvalid("invalid_unintelligible_placeholder")
            rejected = item.get("rejected_asr_texts") or []
            if rejected is None:
                rejected = []
            if not isinstance(rejected, list) or any(not isinstance(t, str) for t in rejected):
                raise TranscriptEvidenceInvalid("invalid_rejected_asr_texts")
            if any(t.strip() and t.strip() == text.strip() for t in rejected):
                raise TranscriptEvidenceInvalid("rejected_text_leaked_into_placeholder")
        identity = json.dumps([digest, a, b, text, status], ensure_ascii=False, separators=(",", ":"))
        sid = "assistant_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
        if sid in seen:
            raise TranscriptEvidenceInvalid("duplicate_assistant_evidence")
        seen.add(sid)
        turn = {"source_id": sid, "role": "assistant", "committed": True,
                "start_ms": a-start, "end_ms": b-start, "text": text,
                "audio_sha256": digest, "asr_derived": True,
                "accuracy_verified": False, "word_timestamps_available": False,
                "timestamp_basis": "task_logical_media_ms",
                "source": "played_assistant_waveform_asr",
                "source_context_start_ms": ca, "source_context_end_ms": cb,
                "status": status}
        if status == "unintelligible":
            turn["segment_index"] = item.get("segment_index")
            turn["voiced_seconds"] = item.get("voiced_seconds")
            # Stored on the evidence record; judge catalog only copies allowlisted
            # fields and does not render rejected_asr_texts into the prompt.
            turn["rejected_asr_texts"] = list(item.get("rejected_asr_texts") or [])
            if item.get("unintelligible_reason") is not None:
                turn["unintelligible_reason"] = item.get("unintelligible_reason")
        turns.append(turn)
        prior_end = b
    return turns


def failure_record(exc: Exception, *, case_id: str, phase: str) -> dict:
    """Non-model failures must not silently become negative model results."""
    name = type(exc).__name__
    if name in {"JudgeEvidenceInvalid", "TranscriptEvidenceInvalid"}:
        classification = "evaluation_invalid"
    elif name in {"SimulatorInvalid", "TypedSimulatorInvalid", "SimulatorContractInvalid"}:
        classification = "simulator_invalid"
    elif name == "CausalWindowUnrealized":
        classification = "challenge_unrealized"
    elif name == "AudibleASRError":
        classification = "evaluation_invalid" if phase == "evaluation" else "simulator_observation_invalid"
    else:
        classification = "infrastructure_error"
    record = {"case_id": case_id, "status": "error", "classification": classification,
              "error_type": name, "error": str(exc), "phase": phase,
              "valid_model_result": False, "integrity_version": INTEGRITY_VERSION}
    record["recovery_policy"] = (
        "reevaluate_preserved_final_without_sut_replay"
        if classification == "evaluation_invalid" else
        "inspect_mechanism_before_new_versioned_attempt"
    )
    details = getattr(exc, "to_dict", None) or getattr(exc, "as_dict", None)
    if callable(details):
        record["details"] = details()
    return record
