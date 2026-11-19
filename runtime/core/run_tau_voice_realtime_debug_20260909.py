#!/usr/bin/env python3
"""Run one real τ-Voice case against a proxied GPT Realtime endpoint."""

from __future__ import annotations

import argparse
import audioop
import base64
import json
import os
import socket
import time
import urllib.request
import wave
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets
from loguru import logger


def load_env(path: Path) -> dict[str, str]:
    """Load simple dotenv entries without exposing values."""
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'").strip('"')
    return values


def write_json(path: Path, payload: object) -> None:
    """Atomically write one JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def redact_sensitive(value: Any, secret_values: set[str]) -> Any:
    """Return a JSON-safe copy with credentials removed."""
    sensitive_keys = {
        "api_key",
        "authorization",
        "gpt_key",
        "realtime_gpt_key",
    }
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if key.lower() in sensitive_keys
                else redact_sensitive(item, secret_values)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item, secret_values) for item in value]
    if isinstance(value, str) and value in secret_values:
        return "[REDACTED]"
    return value


def sanitize_json_artifacts(root: Path, secret_values: set[str]) -> int:
    """Remove credentials from JSON artifacts written by tau2-bench."""
    sanitized = 0
    for path in root.rglob("*.json"):
        raw = path.read_text(encoding="utf-8")
        if not any(secret and secret in raw for secret in secret_values):
            continue
        payload = json.loads(raw)
        write_json(path, redact_sensitive(payload, secret_values))
        sanitized += 1
    return sanitized


def upstream_url(base_url: str, path: str) -> str:
    """Join an OpenAI-compatible base URL and endpoint path."""
    base = base_url.rstrip("/")
    normalized = "/" + path.lstrip("/")
    if base.endswith("/v1") and normalized.startswith("/v1/"):
        normalized = normalized[3:]
    return base + normalized


def post_json(
    base_url: str,
    api_key: str,
    path: str,
    payload: dict[str, Any],
    timeout_seconds: float = 720.0,
) -> dict[str, Any]:
    """POST JSON to the configured GPT broker."""
    if payload.get("model") == "gpt-6-luna":
        payload = dict(payload, model=os.environ.get("MUBENCH_TEXT_MODEL", "gpt-6-luna"))
    request = urllib.request.Request(
        upstream_url(base_url, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        # z4vfq diagnostic copy: keep only the upstream status/body summary;
        # the ASR audit separately redacts configured credentials.
        raise RuntimeError(f"upstream_http_error status={exc.code} body={body.replace(api_key, chr(91) + chr(82) + chr(69) + chr(68) + chr(65) + chr(67) + chr(84) + chr(69) + chr(68) + chr(93))[:1800]}") from exc


def wav_to_pcm16_mono(wav_bytes: bytes, target_rate: int = 16_000) -> bytes:
    """Convert broker TTS WAV bytes to τ-Voice's user PCM format."""
    with wave.open(BytesIO(wav_bytes), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frames = source.readframes(source.getnframes())
    if sample_width != 2:
        frames = audioop.lin2lin(frames, sample_width, 2)
        sample_width = 2
    if channels == 2:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1
    if channels != 1:
        raise ValueError(f"unsupported TTS channel count: {channels}")
    if sample_rate != target_rate:
        frames, _ = audioop.ratecv(
            frames, sample_width, channels, sample_rate, target_rate, None
        )
    return frames


def install_user_service_patches(
    gpt_url: str,
    gpt_key: str,
    tts_model: str,
) -> None:
    """Route τ user-model decisions and TTS through the configured GPT broker."""
    import tau2.agent.base.voice as voice_mixin_module
    import tau2.user.user_simulator_streaming as streaming_user_module
    import tau2.voice.synthesis.audio_effects.speech_generator as speech_generator_module
    import tau2.voice.synthesis.synthesize as synthesis_module
    from tau2.data_model.audio import AudioData, AudioEncoding, AudioFormat

    def relay_synthesize_voice(text: str, provider: str, provider_config: object):
        del provider, provider_config
        response = post_json(
            gpt_url,
            gpt_key,
            "/v1/chat/completions",
            {
                "model": tts_model,
                "modalities": ["text", "audio"],
                "audio": {"voice": "alloy", "format": "wav"},
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Read the supplied utterance verbatim in natural "
                            "conversational English. Do not add, remove, explain, "
                            "or paraphrase any words."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
            },
        )
        audio_payload = response["choices"][0]["message"]["audio"]["data"]
        pcm16 = wav_to_pcm16_mono(base64.b64decode(audio_payload), target_rate=16_000)
        return AudioData(
            data=pcm16,
            format=AudioFormat(
                encoding=AudioEncoding.PCM_S16LE,
                sample_rate=16_000,
                channels=1,
            ),
        )

    voice_mixin_module.synthesize_voice = relay_synthesize_voice
    speech_generator_module.synthesize_voice = relay_synthesize_voice
    synthesis_module.synthesize_voice = relay_synthesize_voice
    streaming_user_module.VOICE_USER_SIMULATOR_DECISION_MODEL = "openai/gpt-4o"


def with_model(base_url: str, model: str) -> str:
    """Set the model query parameter while preserving other query parameters."""
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["model"] = model
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


def open_http_tunnel(target_url: str, proxy_url: str) -> socket.socket:
    """Open CONNECT while accepting PJLab proxy's HTTP/1.0 success response."""
    target = urlsplit(target_url)
    proxy = urlsplit(proxy_url)
    if not target.hostname or not proxy.hostname or not proxy.port:
        raise RuntimeError("invalid Realtime target or HTTP proxy URL")
    target_port = target.port or 443
    tunnel = socket.create_connection((proxy.hostname, proxy.port), timeout=10)
    authority = f"{target.hostname}:{target_port}"
    request = (
        f"CONNECT {authority} HTTP/1.1\r\n"
        f"Host: {authority}\r\n"
        "Proxy-Connection: Keep-Alive\r\n\r\n"
    )
    tunnel.sendall(request.encode("ascii"))
    response = bytearray()
    while b"\r\n\r\n" not in response:
        chunk = tunnel.recv(4096)
        if not chunk:
            break
        response.extend(chunk)
        if len(response) > 65536:
            raise RuntimeError("HTTP proxy CONNECT response was too large")
    status_line = bytes(response).split(b"\r\n", 1)[0]
    if b" 200 " not in status_line:
        raise RuntimeError("HTTP proxy CONNECT did not return success")
    tunnel.setblocking(False)
    return tunnel


def install_realtime_adapter(
    realtime_url: str,
    realtime_key: str,
    model: str,
) -> list[Any]:
    """Patch τ's adapter factory with a provider using the configured gateway."""
    import tau2.agent.discrete_time_audio_native_agent as agent_module
    from tau2.voice.audio_native.openai.discrete_time_adapter import (
        DiscreteTimeOpenAIAdapter,
    )
    from tau2.voice.audio_native.openai.provider import OpenAIRealtimeProvider

    class GatewayRealtimeProvider(OpenAIRealtimeProvider):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(api_key=realtime_key, model=model, **kwargs)
            self.base_url = realtime_url

        async def connect(self) -> None:
            if self.is_connected:
                return
            url = with_model(self.base_url, self.model)
            proxy_url = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
            tunnel = open_http_tunnel(url, proxy_url) if proxy_url else None
            import inspect
            connect_options = dict(sock=tunnel, max_size=16 * 1024 * 1024)
            if "additional_headers" in inspect.signature(websockets.connect).parameters:
                connect_options.update(additional_headers={"Authorization": f"Bearer {self.api_key}"}, proxy=None)
            else:
                connect_options.update(extra_headers={"Authorization": f"Bearer {self.api_key}"})
            self.ws = await websockets.connect(url, **connect_options)
            payload = json.loads(await self.ws.recv())
            if payload.get("type") != "session.created":
                raise RuntimeError(
                    f"expected session.created, got {payload.get('type')}"
                )
            self.session_id = (payload.get("session") or {}).get("id")
            logger.info(
                "Realtime gateway session created (session_id={})",
                self.session_id,
            )

    created_adapters: list[Any] = []

    def create_gateway_adapter(**kwargs: Any):
        provider = GatewayRealtimeProvider(
            reasoning_effort=kwargs.get("reasoning_effort")
        )
        adapter = DiscreteTimeOpenAIAdapter(
            tick_duration_ms=int(kwargs["tick_duration_ms"]),
            send_audio_instant=bool(kwargs.get("send_audio_instant", False)),
            provider=provider,
            audio_format=kwargs.get("audio_format"),
        )
        # The wrapper creates a dedicated provider for this adapter.  Tau's
        # constructor assumes injected providers are externally owned, so make
        # the actual ownership explicit and let disconnect() close the socket.
        adapter._owns_provider = True
        created_adapters.append(adapter)
        return adapter, model

    agent_module.create_adapter = create_gateway_adapter
    return created_adapters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--task-split", default="base")
    parser.add_argument("--task-id", default="0")
    parser.add_argument("--realtime-model", default="gpt-realtime-1.5")
    parser.add_argument("--user-model", default="openai/gpt-4o")
    parser.add_argument("--tts-model", default="gpt-audio-1.5")
    parser.add_argument("--speech-complexity", default="control")
    parser.add_argument("--max-steps-seconds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = load_env(args.env)
    required = [
        "REALTIME_GPT_URL",
        "REALTIME_GPT_KEY",
        "GPT_URL",
        "GPT_KEY",
    ]
    missing = [name for name in required if not values.get(name)]
    if missing:
        raise RuntimeError(f"missing required environment entries: {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpt_url = values["GPT_URL"]
    gpt_key = values["GPT_KEY"]
    os.environ["OPENAI_API_KEY"] = gpt_key
    os.environ["OPENAI_API_BASE"] = gpt_url
    os.environ["OPENAI_BASE_URL"] = gpt_url
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"

    install_user_service_patches(gpt_url, gpt_key, args.tts_model)
    created_adapters = install_realtime_adapter(
        values["REALTIME_GPT_URL"],
        values["REALTIME_GPT_KEY"],
        args.realtime_model,
    )

    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
    from tau2.runner.batch import run_domain

    config = VoiceRunConfig(
        domain=args.domain,
        task_split_name=args.task_split,
        task_ids=[args.task_id],
        llm_user=args.user_model,
        llm_args_user={
            "api_base": gpt_url,
            "api_key": gpt_key,
            "temperature": 0.0,
        },
        num_trials=1,
        max_errors=10,
        timeout=max(300.0, args.max_steps_seconds * 5.0),
        save_to=str(args.output_dir.resolve()),
        max_concurrency=1,
        workers=0,
        seed=args.seed,
        log_level="INFO",
        verbose_logs=True,
        max_retries=0,
        auto_resume=False,
        auto_review=False,
        hallucination_retries=0,
        speech_complexity=args.speech_complexity,
        audio_debug=True,
        audio_taps=True,
        audio_native_config=AudioNativeConfig(
            provider="openai",
            model=args.realtime_model,
            tick_duration_seconds=0.2,
            max_steps_seconds=args.max_steps_seconds,
            send_audio_instant=False,
        ),
    )
    secret_values = {
        values["REALTIME_GPT_KEY"],
        values["GPT_KEY"],
    }
    write_json(
        args.output_dir / "integration-manifest.json",
        redact_sensitive(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "sut": {
                    "provider": "openai-compatible-realtime-gateway",
                    "model": args.realtime_model,
                    "tick_ms": 200,
                },
                "user_services": {
                    "llm": args.user_model,
                    "tts": args.tts_model,
                },
                "case": {
                    "domain": args.domain,
                    "split": args.task_split,
                    "task_id": args.task_id,
                },
                "config": config.model_dump(mode="json"),
            },
            secret_values,
        ),
    )

    started = time.perf_counter()
    status = "failed"
    error: str | None = None
    try:
        print(
            json.dumps(
                {
                    "event": "tau_realtime_smoke_started",
                    "task_id": args.task_id,
                    "model": args.realtime_model,
                    "output_dir": str(args.output_dir),
                }
            ),
            flush=True,
        )
        run_domain(config)
        status = "completed"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for adapter in created_adapters:
            if adapter.is_connected:
                adapter.disconnect()
        sanitized_files = sanitize_json_artifacts(args.output_dir, secret_values)
        write_json(
            args.output_dir / "run-status.json",
            {
                "status": status,
                "error": error,
                "elapsed_seconds": time.perf_counter() - started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "adapter_instances": len(created_adapters),
                "sanitized_json_files": sanitized_files,
            },
        )


if __name__ == "__main__":
    main()
