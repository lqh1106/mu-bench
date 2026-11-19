#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/chat/completions stub for audible_asr_v3.

vadgate-r2 = three changes on top of vadgate-r1 (r1 code preserved separately):
  1) Short-audio pad: if wav duration < SHORT_PAD_MAX_S (1.5s), pad PAD_MS
     silence on both ends before decode; for those short clips also loosen VAD
     (lower threshold, lower min_speech_duration_ms).
  2) Duration-tiered unintelligible gates: short/medium get looser logprob /
     compression_ratio / word-prob / no_speech thresholds; long keeps r1 gates.
  3) Voiced != empty: if light energy OR decode evidence says speech present,
     do NOT clear text to empty via unintelligible. Choice: keep text and set
     status=low_confidence (never map voiced+rejected to no_speech/empty).

asr_backend=faster-whisper-large-v3-vadgate-r2
Smoke-only; do not use for sealed formal runs. Does not touch sealed L0 scores.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import statistics
import struct
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_PATH = os.environ.get("FW_MODEL_PATH", "Systran/faster-whisper-large-v3")
ASR_BACKEND = "faster-whisper-large-v3-vadgate-r2"

# --- r1 baselines (long-clip tier keeps these) ---
R1_VAD_PARAMETERS = {
    "min_silence_duration_ms": 500,
    "speech_pad_ms": 400,
    "threshold": 0.5,
}
R1_GATES = {
    "no_speech_prob_thresh": 0.6,
    "avg_logprob_thresh": -0.60,
    "compression_ratio_post_thresh": 2.4,
    "mean_word_prob_thresh": 0.75,
    "frac_low_word_prob_thresh": 0.25,
    "low_word_prob_cutoff": 0.5,
}

# --- r2 short-audio pad / VAD loosen ---
SHORT_PAD_MAX_S = 1.5
PAD_MS = 400  # silence pad each end (within 300-500ms tune band)
SHORT_VAD_PARAMETERS = {
    "min_silence_duration_ms": 300,
    "speech_pad_ms": 400,
    "threshold": 0.35,
    "min_speech_duration_ms": 50,  # loosen / skip strict min-speech
}

# Duration tiers for post gates (wav duration after optional pad uses original)
TIER_SHORT_MAX_S = 4.0
TIER_MED_MAX_S = 10.0
TIER_GATES = {
    "short": {  # looser than r1
        "no_speech_prob_thresh": 0.80,
        "avg_logprob_thresh": -1.20,
        "compression_ratio_post_thresh": 3.0,
        "mean_word_prob_thresh": 0.50,
        "frac_low_word_prob_thresh": 0.50,
        "low_word_prob_cutoff": 0.5,
    },
    "medium": {
        "no_speech_prob_thresh": 0.70,
        "avg_logprob_thresh": -0.90,
        "compression_ratio_post_thresh": 2.7,
        "mean_word_prob_thresh": 0.60,
        "frac_low_word_prob_thresh": 0.40,
        "low_word_prob_cutoff": 0.5,
    },
    "long": dict(R1_GATES),  # keep long-clip gates closer to r1
}

# Light energy voiced detector
ENERGY_RMS_THRESH = 0.008  # int16-normalized abs mean approx; tuned lightly
ENERGY_PEAK_THRESH = 0.04

GATE_CONFIG = {
    "language": "en",
    "beam_size": 1,
    "condition_on_previous_text": False,
    "vad_filter": True,
    "vad_parameters": dict(R1_VAD_PARAMETERS),
    "hallucination_silence_threshold": 0.5,
    "word_timestamps": True,
    "no_speech_threshold": 0.6,
    "log_prob_threshold": -1.0,
    "compression_ratio_threshold": 2.4,
    # documented r2 knobs
    "r2_short_pad_max_s": SHORT_PAD_MAX_S,
    "r2_pad_ms": PAD_MS,
    "r2_short_vad_parameters": dict(SHORT_VAD_PARAMETERS),
    "r2_tier_short_max_s": TIER_SHORT_MAX_S,
    "r2_tier_med_max_s": TIER_MED_MAX_S,
    "r2_tier_gates": TIER_GATES,
    "r2_voiced_policy": "keep_text_status_low_confidence",
    "r2_energy_rms_thresh": ENERGY_RMS_THRESH,
    "quality_gate_status_when_voiced": "low_confidence",
    "quality_gate_status_when_unvoiced": "unintelligible",
    # expose long (r1) post gates for health visibility
    **{f"r1_{k}": v for k, v in R1_GATES.items()},
}

_lock = threading.Lock()
_model = None
_meta = {"loads": 0, "requests": 0, "last_latency_s": None, "last_text": None, "last_gate": None}


def get_model():
    global _model
    with _lock:
        if _model is None:
            from faster_whisper import WhisperModel
            t0 = time.time()
            device=os.environ.get("FW_DEVICE","cpu"); ctype=os.environ.get("FW_COMPUTE_TYPE","int8" if device=="cpu" else "float16"); _model = WhisperModel(MODEL_PATH, device=device, compute_type=ctype)
            _meta["loads"] += 1
            _meta["load_seconds"] = round(time.time() - t0, 3)
        return _model


def extract_wav_bytes(payload: dict):
    msgs = payload.get("messages") or []
    for msg in msgs:
        content = msg.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "input_audio":
                ia = item.get("input_audio") or {}
                data = ia.get("data")
                if isinstance(data, str) and data:
                    return base64.b64decode(data)
            if "audio" in item and isinstance(item["audio"], dict):
                data = item["audio"].get("data")
                if isinstance(data, str) and data:
                    return base64.b64decode(data)
    return None


def _wav_duration_and_pcm(wav_bytes: bytes):
    """Return (duration_s, sample_rate, n_channels, sampwidth, frames_bytes)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        nframes = w.getnframes()
        frames = w.readframes(nframes)
        dur = nframes / float(sr) if sr else 0.0
        return dur, sr, nch, sw, frames


def _pad_silence_both_ends(wav_bytes: bytes, pad_ms: int) -> bytes:
    dur, sr, nch, sw, frames = _wav_duration_and_pcm(wav_bytes)
    n_pad = int(sr * (pad_ms / 1000.0))
    silence = b"\x00" * (n_pad * nch * sw)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(nch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(silence + frames + silence)
    return out.getvalue()


def light_energy_voiced(wav_bytes: bytes) -> dict:
    """Cheap energy/VAD-ish check: peak + mean abs on PCM."""
    try:
        dur, sr, nch, sw, frames = _wav_duration_and_pcm(wav_bytes)
    except Exception as e:
        return {"voiced": False, "error": str(e)[:120], "rms": None, "peak": None, "duration_s": None}
    if sw != 2:
        # fallback: treat non-int16 as unknown -> rely on decode evidence later
        return {"voiced": False, "error": f"unsupported_sampwidth={sw}", "rms": None, "peak": None, "duration_s": dur}
    n_samp = len(frames) // 2
    if n_samp == 0:
        return {"voiced": False, "rms": 0.0, "peak": 0.0, "duration_s": dur}
    # subsample for speed
    step = max(1, n_samp // 8000)
    acc = 0.0
    peak = 0.0
    count = 0
    for i in range(0, n_samp, step):
        (val,) = struct.unpack_from("<h", frames, i * 2)
        a = abs(val) / 32768.0
        acc += a
        if a > peak:
            peak = a
        count += 1
    rms = acc / count if count else 0.0  # mean-abs proxy
    voiced = (rms >= ENERGY_RMS_THRESH) or (peak >= ENERGY_PEAK_THRESH)
    return {"voiced": bool(voiced), "rms": round(rms, 5), "peak": round(peak, 5), "duration_s": dur}


def duration_tier(duration_s: float) -> str:
    if duration_s < TIER_SHORT_MAX_S:
        return "short"
    if duration_s < TIER_MED_MAX_S:
        return "medium"
    return "long"


def _safe_mean(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(statistics.mean(vals))


def _safe_max(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(max(vals))


def apply_gates(seg_stats, word_probs, text: str, tier: str, energy: dict, orig_duration_s: float):
    gates = TIER_GATES[tier]
    n_segs = len(seg_stats)
    nsp = [s.get("no_speech_prob") for s in seg_stats]
    cr = [s.get("compression_ratio") for s in seg_stats]
    alp = [s.get("avg_logprob") for s in seg_stats]
    mean_wp = _safe_mean(word_probs)
    frac_low = None
    cutoff = gates["low_word_prob_cutoff"]
    if word_probs:
        frac_low = sum(1 for p in word_probs if p < cutoff) / len(word_probs)

    # Decode-side speech evidence: any segment with low no_speech_prob
    max_nsp = _safe_max(nsp)
    mean_nsp = _safe_mean(nsp)
    decode_voiced = False
    if n_segs > 0 and max_nsp is not None and max_nsp < 0.5:
        decode_voiced = True
    if (text or "").strip() and mean_nsp is not None and mean_nsp < 0.55:
        decode_voiced = True
    energy_voiced = bool(energy.get("voiced"))
    voiced = energy_voiced or decode_voiced

    debug = {
        "n_segments": n_segs,
        "max_no_speech_prob": max_nsp,
        "mean_no_speech_prob": mean_nsp,
        "max_compression_ratio": _safe_max(cr),
        "mean_avg_logprob": _safe_mean(alp),
        "min_avg_logprob": (min(alp) if alp and all(v is not None for v in alp) else None),
        "mean_word_prob": mean_wp,
        "frac_word_prob_lt_0.5": frac_low,
        "n_words": len(word_probs),
        "text_len": len(text or ""),
        "gate_fired": None,
        "duration_tier": tier,
        "orig_duration_s": orig_duration_s,
        "tier_gates": gates,
        "energy": energy,
        "energy_voiced": energy_voiced,
        "decode_voiced": decode_voiced,
        "voiced": voiced,
        "r2_voiced_policy": GATE_CONFIG["r2_voiced_policy"],
    }

    if n_segs == 0 or not (text or "").strip():
        # empty decode: true no_speech only when not voiced; if voiced-but-empty keep unintelligible empty? 
        # Product: never map voiced+rejected to same as no_speech empty.
        # Empty text with voiced energy -> unintelligible (empty is ok here? or low_confidence with empty?)
        # Prefer: voiced + empty text -> unintelligible (distinct from no_speech)
        if voiced:
            debug["gate_fired"] = "empty_but_voiced"
            return {"status": "unintelligible", "text": ""}, debug
        debug["gate_fired"] = "empty_or_zero_segments"
        return {"status": "no_speech", "text": ""}, debug

    thresh_nsp = gates["no_speech_prob_thresh"]
    if (max_nsp is not None and max_nsp >= thresh_nsp) or (
        mean_nsp is not None and mean_nsp >= thresh_nsp
    ):
        # High no_speech_prob: if energy says voiced, do NOT map to no_speech empty —
        # fall through to quality / keep path with text.
        if not voiced:
            debug["gate_fired"] = "no_speech_prob"
            return {"status": "no_speech", "text": ""}, debug
        debug["nsp_overridden_by_voiced"] = True

    reasons = []
    mean_alp = debug["mean_avg_logprob"]
    if mean_alp is not None and mean_alp < gates["avg_logprob_thresh"]:
        reasons.append("avg_logprob")
    max_cr = debug["max_compression_ratio"]
    if max_cr is not None and max_cr >= gates["compression_ratio_post_thresh"]:
        reasons.append("compression_ratio")
    if mean_wp is not None and mean_wp < gates["mean_word_prob_thresh"]:
        reasons.append("mean_word_prob")
    if frac_low is not None and frac_low >= gates["frac_low_word_prob_thresh"]:
        reasons.append("frac_low_word_prob")

    if reasons:
        debug["gate_fired"] = "quality:" + "+".join(reasons)
        if voiced and (text or "").strip():
            # Voiced != empty: keep text, status low_confidence
            debug["gate_fired"] = "quality_low_confidence:" + "+".join(reasons)
            return {"status": "low_confidence", "text": text}, debug
        # unvoiced quality fail -> classic unintelligible empty
        return {"status": "unintelligible", "text": ""}, debug

    debug["gate_fired"] = "pass"
    return {"status": "transcribed", "text": text}, debug


def transcribe(wav_bytes: bytes):
    model = get_model()
    t0 = time.time()
    try:
        orig_dur, sr, nch, sw, _frames = _wav_duration_and_pcm(wav_bytes)
    except Exception:
        orig_dur, sr = 0.0, 16000
    energy = light_energy_voiced(wav_bytes)
    tier = duration_tier(orig_dur)

    decode_bytes = wav_bytes
    padded = False
    vad_params = dict(R1_VAD_PARAMETERS)
    if orig_dur < SHORT_PAD_MAX_S:
        decode_bytes = _pad_silence_both_ends(wav_bytes, PAD_MS)
        padded = True
        vad_params = dict(SHORT_VAD_PARAMETERS)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(decode_bytes)
        path = f.name
    try:
        segments, info = model.transcribe(
            path,
            language=GATE_CONFIG["language"],
            beam_size=GATE_CONFIG["beam_size"],
            condition_on_previous_text=GATE_CONFIG["condition_on_previous_text"],
            vad_filter=GATE_CONFIG["vad_filter"],
            vad_parameters=vad_params,
            hallucination_silence_threshold=GATE_CONFIG["hallucination_silence_threshold"],
            word_timestamps=GATE_CONFIG["word_timestamps"],
            no_speech_threshold=GATE_CONFIG["no_speech_threshold"],
            log_prob_threshold=GATE_CONFIG["log_prob_threshold"],
            compression_ratio_threshold=GATE_CONFIG["compression_ratio_threshold"],
        )
        texts = []
        seg_stats = []
        word_probs = []
        for seg in segments:
            t = (seg.text or "").strip()
            if t:
                texts.append(t)
            seg_stats.append({
                "text": t,
                "no_speech_prob": getattr(seg, "no_speech_prob", None),
                "avg_logprob": getattr(seg, "avg_logprob", None),
                "compression_ratio": getattr(seg, "compression_ratio", None),
                "start": getattr(seg, "start", None),
                "end": getattr(seg, "end", None),
            })
            words = getattr(seg, "words", None) or []
            for w in words:
                p = getattr(w, "probability", None)
                if p is not None:
                    word_probs.append(float(p))
        text = " ".join(texts).strip()
        body, gate_debug = apply_gates(seg_stats, word_probs, text, tier, energy, orig_dur)
        gate_debug["info_language"] = getattr(info, "language", None)
        gate_debug["info_duration"] = getattr(info, "duration", None)
        gate_debug["short_padded"] = padded
        gate_debug["pad_ms"] = PAD_MS if padded else 0
        gate_debug["vad_parameters_used"] = vad_params
        _meta["last_gate"] = gate_debug
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    latency = time.time() - t0
    _meta["requests"] += 1
    _meta["last_latency_s"] = round(latency, 3)
    _meta["last_text"] = body.get("text", "")[:200]
    _meta["last_status"] = body.get("status")
    _meta["asr_backend"] = ASR_BACKEND
    return body, latency


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[fw-stub-r2] {self.address_string()} {fmt % args}", flush=True)

    def _json(self, code: int, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ("/health", "/v1/health", "/"):
            return self._json(200, {
                "ok": True,
                "asr_backend": ASR_BACKEND,
                "model_path": MODEL_PATH,
                "gate_config": GATE_CONFIG,
                "meta": _meta,
            })
        return self._json(404, {"error": "not_found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return self._json(400, {"error": "invalid_json"})
        path = self.path.split("?")[0]
        if path.endswith("/chat/completions"):
            wav = extract_wav_bytes(payload)
            if wav is None:
                return self._json(400, {"error": "no_input_audio_in_request", "hint": "stub is ASR-only"})
            try:
                result, latency = transcribe(wav)
            except Exception as e:
                return self._json(
                    500,
                    {"error": "asr_failed", "exception_type": type(e).__name__, "message": str(e)[:300]},
                )
            content = json.dumps(result, ensure_ascii=False)
            resp = {
                "id": "fw-asr-smoke-r2",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": payload.get("model") or "gpt-audio-1.5",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": content},
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "asr_backend": ASR_BACKEND,
                "asr_latency_seconds": round(latency, 3),
                "gate_debug": _meta.get("last_gate"),
            }
            return self._json(200, resp)
        return self._json(404, {"error": "not_found", "path": path})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.environ.get("FW_STUB_PORT", "18991")))
    ap.add_argument("--preload", action="store_true")
    args = ap.parse_args()
    if args.preload:
        print(f"[fw-stub-r2] preloading {MODEL_PATH}", flush=True)
        get_model()
        print(f"[fw-stub-r2] loaded meta={_meta}", flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(
        f"[fw-stub-r2] listening on http://{args.host}:{args.port} asr_backend={ASR_BACKEND}",
        flush=True,
    )
    httpd.serve_forever()


if __name__ == "__main__":
    main()
