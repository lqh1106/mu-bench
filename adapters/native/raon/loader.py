#!/usr/bin/env python3
"""Persistent GPU worker for μ-Bench open-weight full-duplex sessions.

The networked benchmark runner writes one complete user-side WAV to GPFS. The
WAV begins with spoken self-introductions; no identity roster or case prompt is
sent as text. This worker keeps one SUT loaded on the GPU, preserves its official
default system prompt, runs a fresh model session, and returns model-native
transcript/timing artifacts without network access or API credentials.
"""

from __future__ import annotations

import argparse
import ast
import base64
import gc
import io
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from typing import Any
import uuid

import numpy as np

from audio_utils import read_wav_mono, speech_intervals, wav_duration_ms, write_wav_mono
from gpfs_rpc import atomic_json, load_json
from lychee_session_integrity import (
    inspect_lychee_session_completion,
    lychee_drain_timeout_seconds,
    require_lychee_completion,
    require_verified_lychee_media_clock,
)


PERSONAPLEX_OFFICIAL_DEFAULT_SYSTEM_PROMPT = (
    "You are a wise and friendly teacher. Answer questions or provide advice "
    "in a clear and engaging way."
)

OFFICIAL_DEFAULT_PROMPT_SOURCES = {
    "minicpm-o-4_5": "MiniCPM-o as_duplex.prepare default",
    "personaplex-7b-v1": "PersonaPlex official offline CLI default",
    "raon-speechchat-9b": "Raon-SpeechChat duplex config default",
    "lychee-fd": "Lychee-FD realtime backend default",
}


def require_model_default_prompt(system_prompt: str | None) -> None:
    """Reject benchmark-authored prompt overrides in the unified protocol."""
    if system_prompt is not None:
        raise ValueError(
            "custom system prompts are forbidden; use the model official default"
        )


def plan_raon_cache_sequence_length(
    *,
    system_token_count: int,
    audio_sample_count: int,
    sample_rate: int,
    frame_rate: float,
    minimum: int = 8_192,
    maximum: int = 65_536,
) -> dict[str, int]:
    """Size Raon's static duplex KV cache before CUDA inference starts.

    A duplex frame appends either two or three text-stream tokens.  Raon's
    upstream helper silently uses an 8,192-token StaticCache, so sufficiently
    long Probe/Final tracks overflow that cache inside a CUDA kernel.  Reserve
    the worst case of three tokens per frame plus a small structural margin,
    rounded to 1,024 tokens to avoid reallocating for insignificant duration
    differences.
    """

    if system_token_count < 0 or audio_sample_count < 0:
        raise ValueError("Raon cache inputs must be non-negative")
    if sample_rate <= 0 or frame_rate <= 0:
        raise ValueError("Raon sample_rate and frame_rate must be positive")
    if minimum <= 0 or maximum < minimum:
        raise ValueError("invalid Raon cache bounds")

    samples_per_frame = int(sample_rate / frame_rate)
    if samples_per_frame <= 0:
        raise ValueError("Raon samples_per_frame must be positive")
    frame_count = audio_sample_count // samples_per_frame
    required = system_token_count + 2 + 3 * frame_count + 32
    planned = max(minimum, ((required + 1_023) // 1_024) * 1_024)
    if planned > maximum:
        raise ValueError(
            "Raon duplex track exceeds the configured KV-cache safety cap: "
            f"required={required}, planned={planned}, maximum={maximum}, "
            f"frames={frame_count}, system_tokens={system_token_count}"
        )
    return {
        "system_token_count": system_token_count,
        "audio_sample_count": audio_sample_count,
        "samples_per_frame": samples_per_frame,
        "frame_count": frame_count,
        "required_sequence_length": required,
        "planned_sequence_length": planned,
    }


class MiniCPMBackend:
    sample_rate = 24_000

    def __init__(self, args: argparse.Namespace) -> None:
        import librosa
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.model = AutoModel.from_pretrained(
            str(args.model_path),
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=False,
            init_audio=True,
            init_tts=True,
        )
        self.model.eval().cuda()
        self.ref_audio, _ = librosa.load(str(args.ref_audio), sr=16_000, mono=True)
        self.duplex = self.model.as_duplex(
            device="cuda", generate_audio=True, chunk_ms=1000, first_chunk_ms=1035
        )
        self.prompt_wav_path = str(args.ref_audio)

    def run_session(
        self, input_wav: Path, output_dir: Path, system_prompt: str | None, seed: int
    ) -> dict[str, Any]:
        del seed
        require_model_default_prompt(system_prompt)
        audio = read_wav_mono(input_wav, 16_000)
        chunk_samples = 16_000
        chunks = int(np.ceil(len(audio) / chunk_samples))
        padded = np.pad(audio, (0, chunks * chunk_samples - len(audio)))
        self.duplex.prepare(
            prefix_system_prompt=system_prompt,
            ref_audio=self.ref_audio,
            prompt_wav_path=self.prompt_wav_path,
        )
        output = np.zeros(chunks * self.sample_rate + self.sample_rate * 8, dtype=np.float32)
        playback_cursor = 0
        fragments: list[dict[str, Any]] = []
        native_ticks: list[dict[str, Any]] = []
        for index in range(chunks):
            chunk = padded[index * chunk_samples : (index + 1) * chunk_samples]
            prefill = self.duplex.streaming_prefill(audio_waveform=chunk)
            if not prefill.get("success", False):
                raise RuntimeError(f"MiniCPM streaming_prefill failed: {prefill}")
            generated = self.duplex.streaming_generate(
                prompt_wav_path=self.prompt_wav_path,
                max_new_speak_tokens_per_chunk=20,
                decode_mode="sampling",
            )
            text = str(generated.get("text", ""))
            waveform = generated.get("audio_waveform")
            emitted = waveform is not None and not bool(generated.get("is_listen", True))
            placed_start_ms = index * 1000
            if emitted:
                waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
                tick_start = index * self.sample_rate
                start = max(tick_start, playback_cursor)
                end = start + len(waveform)
                if end > len(output):
                    output = np.pad(output, (0, end - len(output)))
                output[start:end] += waveform
                playback_cursor = end
                placed_start_ms = round(start * 1000 / self.sample_rate)
            if text.strip():
                fragments.append({"timestamp_ms": placed_start_ms, "text": text})
            native_ticks.append(
                {
                    "tick": index,
                    "is_listen": bool(generated.get("is_listen", True)),
                    "end_of_turn": bool(generated.get("end_of_turn", False)),
                    "text": text,
                    "placed_start_ms": placed_start_ms,
                }
            )
        self.duplex.set_session_stop()
        duration_samples = max(chunks * self.sample_rate, playback_cursor)
        output = output[:duration_samples]
        assistant_wav = output_dir / "assistant.wav"
        write_wav_mono(assistant_wav, output, self.sample_rate)
        (output_dir / "native_ticks.json").write_text(
            json.dumps(native_ticks, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return session_result("minicpm-o-4_5", input_wav, assistant_wav, fragments)


class PersonaPlexBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        repo_moshi = args.repo_path / "moshi"
        sys.path.insert(0, str(repo_moshi))
        import sentencepiece
        import torch
        from moshi.models import loaders, LMGen
        from moshi.offline import warmup

        self.torch = torch
        self.loaders = loaders
        self.device = "cuda"
        self.mimi = loaders.get_mimi(str(args.mimi_weight), self.device)
        self.other_mimi = loaders.get_mimi(str(args.mimi_weight), self.device)
        self.tokenizer = sentencepiece.SentencePieceProcessor(str(args.tokenizer))
        lm = loaders.get_moshi_lm(str(args.moshi_weight), device=self.device)
        lm.eval()
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=self.device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False,
            use_sampling=True,
            temp=0.8,
            temp_text=0.7,
            top_k=250,
            top_k_text=25,
        )
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)
        warmup(self.mimi, self.other_mimi, self.lm_gen, self.device, self.frame_size)
        self.voice_prompt = str(args.voice_prompt)
        self.lm_gen.load_voice_prompt_embeddings(self.voice_prompt)

    def run_session(
        self, input_wav: Path, output_dir: Path, system_prompt: str | None, seed: int
    ) -> dict[str, Any]:
        # PersonaPlex advances thousands of streaming frames per benchmark case.
        # Without inference_mode, autograd graphs can remain reachable from the
        # streaming state and grow until a long trajectory exhausts an H200.
        with self.torch.inference_mode():
            return self._run_session_inference(
                input_wav, output_dir, system_prompt, seed
            )

    def _run_session_inference(
        self, input_wav: Path, output_dir: Path, system_prompt: str | None, seed: int
    ) -> dict[str, Any]:
        from moshi.offline import (
            decode_tokens_to_pcm,
            lm_encode_from_sphn,
            lm_iterate_audio,
            lm_load_audio,
            seed_all,
            wrap_with_system_tags,
        )

        seed_all(seed)
        require_model_default_prompt(system_prompt)
        effective_system_prompt = PERSONAPLEX_OFFICIAL_DEFAULT_SYSTEM_PROMPT
        self.lm_gen.text_prompt_tokens = self.tokenizer.encode(
            wrap_with_system_tags(effective_system_prompt)
        )
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()
        self.lm_gen.reset_streaming()
        self.lm_gen.step_system_prompts(self.mimi)
        self.mimi.reset_streaming()
        user_audio = lm_load_audio(str(input_wav), self.mimi.sample_rate)
        generated_frames: list[np.ndarray] = []
        fragments: list[dict[str, Any]] = []
        token_trace: list[dict[str, Any]] = []
        frame_index = 0
        for encoded in lm_encode_from_sphn(
            self.mimi,
            lm_iterate_audio(
                user_audio, sample_interval_size=self.lm_gen._frame_size, pad=True
            ),
            max_batch=1,
        ):
            for column in range(encoded.shape[-1]):
                tokens = self.lm_gen.step(encoded[:, :, column : column + 1])
                if tokens is None:
                    continue
                generated_frames.append(
                    decode_tokens_to_pcm(
                        self.mimi, self.other_mimi, self.lm_gen, tokens
                    )
                )
                token_id = int(tokens[0, 0, 0].item())
                timestamp_ms = round(frame_index * 1000 / self.mimi.frame_rate)
                piece = ""
                if token_id not in (0, 3):
                    piece = self.tokenizer.id_to_piece(token_id).replace("▁", " ")
                    fragments.append({"timestamp_ms": timestamp_ms, "text": piece})
                token_trace.append(
                    {"frame": frame_index, "timestamp_ms": timestamp_ms, "token_id": token_id, "text": piece}
                )
                frame_index += 1
        output = (
            np.concatenate(generated_frames).astype(np.float32)
            if generated_frames
            else np.zeros(user_audio.shape[-1], dtype=np.float32)
        )
        target = user_audio.shape[-1]
        output = np.pad(output[:target], (0, max(0, target - len(output))))
        assistant_wav = output_dir / "assistant.wav"
        write_wav_mono(assistant_wav, output, int(self.mimi.sample_rate))
        (output_dir / "native_tokens.json").write_text(
            json.dumps(token_trace, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return session_result("personaplex-7b-v1", input_wav, assistant_wav, fragments)


class RaonBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.torch = torch
        pipeline_class = get_class_from_dynamic_module(
            "modeling_raon.RaonPipeline", str(args.model_path), local_files_only=True
        )
        self.pipeline = pipeline_class(
            str(args.model_path),
            device="cuda",
            dtype="bfloat16",
            attn_implementation="sdpa",
            config=str(args.repo_path / "config/infer.yaml"),
            duplex_config=str(args.repo_path / "config/duplex_infer.yaml"),
        )
        self.minimum_sequence_length = 8_192
        self.maximum_sequence_length = int(args.raon_max_sequence_length)
        self.active_sequence_length = self.minimum_sequence_length
        original_init_duplex_state = self.pipeline.model.init_duplex_decoding_state

        def init_duplex_state_with_planned_cache(*call_args: Any, **call_kwargs: Any) -> Any:
            requested = int(
                call_kwargs.get("max_sequence_length", self.minimum_sequence_length)
            )
            call_kwargs["max_sequence_length"] = max(
                requested, self.active_sequence_length
            )
            return original_init_duplex_state(*call_args, **call_kwargs)

        self.pipeline.model.init_duplex_decoding_state = init_duplex_state_with_planned_cache
        torch.cuda.synchronize()

    def run_session(
        self, input_wav: Path, output_dir: Path, system_prompt: str | None, seed: int
    ) -> dict[str, Any]:
        del seed
        require_model_default_prompt(system_prompt)
        audio = self.pipeline.load_audio(str(input_wav))
        native_dir = output_dir / "native"
        effective_system_prompt = system_prompt or self.pipeline.duplex_params.get(
            "system_prompt", "You are engaging in real-time conversation."
        )
        system_messages = [{"role": "system", "content": effective_system_prompt}]
        system_inputs = self.pipeline.processor(
            system_messages,
            add_generation_prompt=False,
            device=self.pipeline.device,
            dtype=self.pipeline.dtype,
        )
        cache_plan = plan_raon_cache_sequence_length(
            system_token_count=int(system_inputs["input_ids"].shape[-1]),
            audio_sample_count=int(audio.shape[-1]),
            sample_rate=int(self.pipeline.processor.sampling_rate),
            frame_rate=float(self.pipeline.processor.frame_rate),
            minimum=self.minimum_sequence_length,
            maximum=self.maximum_sequence_length,
        )
        del system_inputs
        native_dir.mkdir(parents=True, exist_ok=True)
        (native_dir / "adapter_limits.json").write_text(
            json.dumps(cache_plan, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.active_sequence_length = cache_plan["planned_sequence_length"]
        try:
            self.pipeline.duplex(
                audio_input=audio,
                output_dir=str(native_dir),
                speak_first=False,
                system_prompt=system_prompt,
            )
            self.torch.cuda.synchronize()
        finally:
            self.active_sequence_length = self.minimum_sequence_length
        assistant_wav = native_dir / "assistant.wav"
        fragments: list[dict[str, Any]] = []
        line_pattern = re.compile(r"f=(\d+).*?text=(.+?) out_rms=")
        for line in (native_dir / "frame_log.txt").read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            match = line_pattern.search(line)
            if not match or match.group(2) == "-":
                continue
            try:
                text = str(ast.literal_eval(match.group(2)))
            except (ValueError, SyntaxError):
                text = match.group(2).strip("'\"")
            if text and not text.startswith("<|audio_output_"):
                fragments.append(
                    {"timestamp_ms": int(match.group(1)) * 80, "text": text}
                )
        return session_result("raon-speechchat-9b", input_wav, assistant_wav, fragments)


class LycheeBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.app_root = args.lychee_app_root
        self.python = args.lychee_python
        self.port = int(getattr(args, "lychee_port", 19686))
        if not 1 <= self.port <= 65535:
            raise ValueError("invalid Lychee internal service port")
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.backend: subprocess.Popen[str] | None = None
        self.prompt: str | None = None
        self._undrained_native_session_id: str | None = None
        self.log_root = args.worker_root / "lychee_backend_logs"
        self.log_root.mkdir(parents=True, exist_ok=True)

    def _stop_backend(self) -> None:
        if self.backend is not None and self.backend.poll() is None:
            self.backend.terminate()
            try:
                self.backend.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.backend.kill()
                self.backend.wait(timeout=10)
        self.backend = None

    def _ensure_backend(self, prompt: str | None) -> None:
        import requests

        if self.backend is not None and self.backend.poll() is None and self.prompt == prompt:
            return
        self._stop_backend()
        self.prompt = prompt
        env = dict(os.environ)
        # The HTTP control service uses base Python plus isolated --target
        # dependencies. The official backend must use only its own extracted
        # conda runtime, never the outer service's PYTHONPATH/PYTHONHOME.
        conda_root = self.python.parent.parent
        env["PYTHONHOME"] = str(conda_root)
        env["PYTHONPATH"] = str(self.app_root)
        env["PYTHONNOUSERSITE"] = "1"
        env["PATH"] = str(conda_root / "bin") + os.pathsep + env.get("PATH", "")
        env["LYCHEEFD_CONDA_ENV_PATH"] = str(conda_root)
        env["LYCHEEFD_SERVER_PORT"] = str(self.port)
        env["LYCHEEFD_VLLM_SYNC_FLASH_ATTN"] = "0"
        if prompt is not None:
            env["LYCHEEFD_SYSTEM_MESSAGE_PREFIX"] = (
                f"<|BOT|>system\n{prompt}<|EOT|>"
            )
        else:
            env.pop("LYCHEEFD_SYSTEM_MESSAGE_PREFIX", None)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = self.log_root / f"backend_{stamp}_{uuid.uuid4().hex[:8]}.log"
        log_handle = log_path.open("x", encoding="utf-8")
        self.backend = subprocess.Popen(
            [str(self.app_root / "scripts/start_backend.sh"), "stable", "vllm"],
            cwd=str(self.app_root),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if self.backend.poll() is not None:
                raise RuntimeError(f"Lychee backend exited; inspect {log_path}")
            try:
                response = requests.get(f"{self.base_url}/api/realtime/voices", timeout=5)
                if response.ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(f"Lychee backend readiness timed out; inspect {log_path}")

    def run_session(
        self, input_wav: Path, output_dir: Path, system_prompt: str | None, seed: int
    ) -> dict[str, Any]:
        del seed
        require_model_default_prompt(system_prompt)
        import requests
        import soundfile as sf

        if self._undrained_native_session_id is not None:
            raise RuntimeError(
                "Lychee infrastructure-invalid: prior native session has not been "
                f"proven drained: {self._undrained_native_session_id}; refusing a new session"
            )
        self._ensure_backend(system_prompt)
        audio, sample_rate = sf.read(str(input_wav), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16_000:
            raise ValueError(f"Lychee expected 16 kHz input, got {sample_rate}")
        drain_timeout = lychee_drain_timeout_seconds(
            len(audio), sample_rate, os.environ.get("LYCHEEFD_BENCH_DRAIN_TIMEOUT_SECONDS")
        )
        start = requests.post(
            f"{self.base_url}/api/realtime/session/start",
            json={
                "prompt_voice": "default_female",
                "infer_window_ms": 400,
                "strict_infer_window": True,
                "incremental_backend": "auto",
                "start_speak_factor": 1.2,
                "start_listen_factor": 1.2,
                "end_speak_factor": 1.0,
            },
            timeout=180,
        )
        start.raise_for_status()
        session_id = start.json()["session_id"]
        self._undrained_native_session_id = session_id
        events: list[dict[str, Any]] = []
        receive_errors: list[str] = []
        done = threading.Event()
        session_zero = time.monotonic()

        def receiver() -> None:
            try:
                with requests.get(
                    f"{self.base_url}/api/realtime/session/{session_id}/events",
                    stream=True,
                    timeout=(30, 300),
                ) as stream:
                    stream.raise_for_status()
                    for raw in stream.iter_lines(decode_unicode=True):
                        if not raw or not raw.startswith("data:"):
                            continue
                        event = json.loads(raw[5:].strip())
                        event["received_ms"] = round((time.monotonic() - session_zero) * 1000)
                        events.append(event)
            except Exception as exc:  # noqa: BLE001
                receive_errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                done.set()

        thread = threading.Thread(target=receiver, daemon=True)
        thread.start()
        chunk_samples = 6_400
        send_zero = time.monotonic()
        for offset in range(0, len(audio), chunk_samples):
            chunk = audio[offset : offset + chunk_samples]
            target = send_zero + (offset + len(chunk)) / sample_rate
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            buffer = io.BytesIO()
            sf.write(buffer, chunk, sample_rate, format="WAV", subtype="PCM_16")
            response = requests.post(
                f"{self.base_url}/api/realtime/session/{session_id}/chunk",
                data=buffer.getvalue(),
                headers={"Content-Type": "audio/wav"},
                timeout=180,
            )
            response.raise_for_status()
        time.sleep(2.0)
        requests.post(
            f"{self.base_url}/api/realtime/session/{session_id}/stop", timeout=60
        ).raise_for_status()
        # /stop only requests EOF. Official Lychee continues consuming its
        # pending native 400 ms quanta, then flushes TTS and emits native done.
        # Slow batch-semantic inference is allowed; ignored timeouts are not.
        receiver_finished = done.wait(timeout=drain_timeout)
        captured_events = list(events)
        captured_receive_errors = list(receive_errors)
        completion = inspect_lychee_session_completion(
            captured_events, input_samples=len(audio), sample_rate=sample_rate,
            receiver_finished=receiver_finished, receive_errors=captured_receive_errors,
        )
        completion.update(session_id=session_id, drain_timeout_seconds=drain_timeout)
        atomic_json(output_dir / "lychee_completion_audit.json", completion)
        (output_dir / "native_events.json").write_text(
            json.dumps({"events": captured_events, "receive_errors": captured_receive_errors}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_rate = 24_000
        output = np.zeros(int((len(audio) / sample_rate + 15) * output_rate), dtype=np.float32)
        cursor = 0
        fragments: list[dict[str, Any]] = []
        for event in captured_events:
            kind = str(event.get("type") or event.get("event_type") or "")
            if kind in {"assistant_text", "text_delta"}:
                text = str(event.get("text") or event.get("delta") or "")
                if text:
                    fragments.append({"timestamp_ms": int(event["received_ms"]), "text": text})
            payload = event.get("pcm_b64")
            if isinstance(payload, str) and payload:
                output_rate = int(event.get("sample_rate") or output_rate)
                pcm = np.frombuffer(base64.b64decode(payload), dtype="<i2").astype(np.float32) / 32768.0
                start_sample = max(cursor, round(int(event["received_ms"]) * output_rate / 1000))
                end_sample = start_sample + len(pcm)
                if end_sample > len(output):
                    output = np.pad(output, (0, end_sample - len(output)))
                output[start_sample:end_sample] += pcm
                cursor = end_sample
        duration = max(round(len(audio) * output_rate / sample_rate), cursor)
        assistant_wav = output_dir / "assistant.wav"
        write_wav_mono(assistant_wav, output[:duration], output_rate)
        require_lychee_completion(completion)
        self._undrained_native_session_id = None
        # Until source audio-token/frame timing is instrumented, preserve the
        # diagnostic WAV/events but never admit this wall-arrival track to the
        # official Probe planner or declare it a valid model result.
        require_verified_lychee_media_clock()
        return session_result("lychee-fd", input_wav, assistant_wav, fragments, session_id=session_id)

    def close(self) -> None:
        self._stop_backend()


def session_result(
    model: str,
    input_wav: Path,
    assistant_wav: Path,
    fragments: list[dict[str, Any]],
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    intervals = speech_intervals(assistant_wav)
    return {
        "model": model,
        "session_id": session_id or f"local-{model}-{uuid.uuid4().hex}",
        "input_wav": str(input_wav),
        "assistant_wav": str(assistant_wav),
        "media_duration_ms": max(wav_duration_ms(input_wav), wav_duration_ms(assistant_wav)),
        "assistant_fragments": fragments,
        "assistant_intervals": intervals,
        "assistant_text": "".join(str(item.get("text", "")) for item in fragments).strip(),
        "system_prompt_policy": "model_official_default",
        "custom_system_prompt_sent": False,
        "system_prompt_source": OFFICIAL_DEFAULT_PROMPT_SOURCES[model],
        "identity_delivery": "spoken_tts_self_introduction",
    }


def run_session_spec_compat(
    backend: Any,
    spec_path: Path,
    result_path: Path,
    *,
    started_at_unix: float,
) -> dict[str, Any]:
    """Execute a unified SessionSpec through the legacy whole-track adapter.

    The development runner submits ``SessionSubmission`` pointers with the
    ``run_session_spec`` method.  Older model workers only understood the
    private ``run_session`` method, which caused an infrastructure failure
    before inference.  This narrow bridge keeps the model-native call intact
    and publishes the versioned SessionResult expected by the coordinator.
    """
    spec = load_json(spec_path)
    artifacts = dict(spec["artifacts"])
    input_wav = Path(artifacts["input_wav"])
    output_dir = Path(artifacts["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = None
    prompt_path = artifacts.get("system_prompt_path")
    if prompt_path:
        prompt = Path(prompt_path).read_text(encoding="utf-8")
    payload = backend.run_session(
        input_wav,
        output_dir,
        prompt,
        int(spec.get("seed", 42)),
    )
    completed_at_unix = time.time()
    result = {
        "schema_version": "mu-bench-session-result-v1",
        "runtime_protocol_version": "mu-bench-runtime-v1.3-hybrid-control",
        "session_id": str(spec["session_id"]),
        "attempt_id": int(spec.get("attempt_id", 1)),
        "case_id": str(spec["case_id"]),
        "phase": str(spec["phase"]),
        "state": "succeeded",
        "model_name": str(spec["model_name"]),
        "backend_name": str(spec["backend_name"]),
        "started_at_unix": started_at_unix,
        "completed_at_unix": completed_at_unix,
        "media_duration_ms": int(payload.get("media_duration_ms", 0)),
        "inference_wall_ms": round(
            max(0.0, completed_at_unix - started_at_unix) * 1000.0, 3
        ),
        "input_wav": str(input_wav),
        "assistant_wav": str(payload["assistant_wav"]),
        "assistant_text": str(payload.get("assistant_text", "")),
        "assistant_fragments": list(payload.get("assistant_fragments", [])),
        "assistant_intervals": list(payload.get("assistant_intervals", [])),
        "tool_observations": list(payload.get("tool_observations", [])),
        "runtime_metrics": {
            "frame_transport": "gpu_process_local_legacy_track",
            "tick_ms": int(spec.get("media_clock", {}).get("tick_ms", 200)),
            "wall_timing_eligible": False,
        },
        "model_metadata": {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "assistant_wav",
                "assistant_text",
                "assistant_fragments",
                "assistant_intervals",
                "tool_observations",
            }
        },
    }
    atomic_json(result_path, result)
    return {
        "session_id": result["session_id"],
        "state": result["state"],
        "result_path": str(result_path),
        "inference_wall_ms": result["inference_wall_ms"],
    }


def build_backend(args: argparse.Namespace) -> Any:
    if args.model == "minicpm":
        return MiniCPMBackend(args)
    if args.model == "personaplex":
        return PersonaPlexBackend(args)
    if args.model == "raon":
        return RaonBackend(args)
    if args.model == "lychee":
        return LycheeBackend(args)
    raise ValueError(args.model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("minicpm", "personaplex", "raon", "lychee"), required=True)
    parser.add_argument("--queue-dir", type=Path, required=True)
    parser.add_argument("--worker-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--repo-path", type=Path)
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--moshi-weight", type=Path)
    parser.add_argument("--mimi-weight", type=Path)
    parser.add_argument("--voice-prompt", type=Path)
    parser.add_argument("--lychee-app-root", type=Path)
    parser.add_argument("--lychee-python", type=Path)
    parser.add_argument("--raon-max-sequence-length", type=int, default=65_536)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    args.worker_root.mkdir(parents=True, exist_ok=True)
    queue_dir = args.queue_dir.resolve()
    for directory in (queue_dir / "requests", queue_dir / "processing", queue_dir / "responses"):
        directory.mkdir(parents=True, exist_ok=True)
        directory.chmod(0o777)
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    load_started = time.perf_counter()
    backend = build_backend(args)
    atomic_json(
        queue_dir / "worker-ready.json",
        {
            "model": args.model,
            "ready_at": time.time(),
            "load_seconds": time.perf_counter() - load_started,
        },
    )
    print(json.dumps({"worker_ready": True, "model": args.model}), flush=True)
    try:
        while not stop and not (queue_dir / "STOP").exists():
            found = False
            for request_path in sorted((queue_dir / "requests").glob("*.json")):
                found = True
                claimed = queue_dir / "processing" / request_path.name
                try:
                    os.replace(request_path, claimed)
                except FileNotFoundError:
                    continue
                request_id = claimed.stem
                response_path = queue_dir / "responses" / f"{request_id}.json"
                try:
                    request = load_json(claimed)
                    method = str(request["method"])
                    payload = dict(request.get("payload", {}))
                    if method == "health":
                        response = {"ready": True, "model": args.model}
                    elif method == "run_session_spec":
                        started = time.time()
                        response = run_session_spec_compat(
                            backend,
                            Path(payload["spec_path"]),
                            Path(payload["result_path"]),
                            started_at_unix=started,
                        )
                    elif method == "run_session":
                        output_dir = Path(payload["output_dir"])
                        output_dir.mkdir(parents=True, exist_ok=True)
                        started = time.perf_counter()
                        response = backend.run_session(
                            Path(payload["input_wav"]),
                            output_dir,
                            payload.get("system_prompt"),
                            int(payload.get("seed", 42)),
                        )
                        response["inference_wall_ms"] = round(
                            (time.perf_counter() - started) * 1000, 3
                        )
                        atomic_json(output_dir / "session_result.json", response)
                    elif method == "stop":
                        response = {"stopping": True}
                        stop = True
                    else:
                        raise ValueError(f"unknown method: {method}")
                    atomic_json(response_path, {"id": request_id, "ok": True, "payload": response})
                except Exception as exc:  # noqa: BLE001
                    atomic_json(
                        response_path,
                        {
                            "id": request_id,
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(),
                        },
                    )
                finally:
                    claimed.unlink(missing_ok=True)
                    # Long audio sessions can leave cyclic references to large
                    # generation tensors alive until Python's periodic GC.  A
                    # persistent worker must reclaim them between independent
                    # benchmark cases or the CUDA allocator grows case by case.
                    gc.collect()
                    torch_module = sys.modules.get("torch")
                    if torch_module is not None and torch_module.cuda.is_available():
                        torch_module.cuda.empty_cache()
                        torch_module.cuda.ipc_collect()
            if not found:
                time.sleep(args.poll_seconds)
    finally:
        if hasattr(backend, "close"):
            backend.close()
        atomic_json(queue_dir / "worker-stopped.json", {"stopped_at": time.time()})


if __name__ == "__main__":
    main()
