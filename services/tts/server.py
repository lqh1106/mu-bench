#!/usr/bin/env python3
"""Serve the existing deterministic CosyVoice backend as synchronous PCM HTTP."""

from __future__ import annotations

import argparse
import hashlib
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlparse
import uuid


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO_ROOT))
from mubench.config import activate,profile
activate(profile('native','tool_free'))

from experiments.tau_multiuser_voice.tts_backends import CosyVoiceTTS  # noqa: E402
from experiments.tau_multiuser_voice.tts_http import (  # noqa: E402
    TTSCaseRegistration,
    TTSCaseRelease,
    TTSProfile,
    TTSRegistrationResult,
    TTSServiceEndpoint,
    TTSSynthesisRequest,
)
from experiments.tau_multiuser_voice.voice_catalog import (  # noqa: E402
    GlobalVoiceBindingManifest,
    VoiceCatalogManifest,
)


class RegistrationConflict(ValueError):
    """A case was registered with incompatible participant metadata."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}

    def as_dict(self) -> dict[str, Any]:
        return {"error_type": type(self).__name__, "error": str(self), **self.diagnostics}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_ipv4(advertise_host: str | None = None) -> str:
    if advertise_host:
        return advertise_host
    for variable in ("POD_IP", "RJOB_POD_IP"):
        if os.environ.get(variable):
            return str(os.environ[variable])
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as handle:
            handle.connect(("10.255.255.255", 1))
            address = str(handle.getsockname()[0])
            if address and not address.startswith("127."):
                return address
    except OSError:
        pass
    for address in socket.gethostbyname_ex(socket.gethostname())[2]:
        if address and not address.startswith("127."):
            return address
    raise RuntimeError("could not discover a non-loopback service address")


def atomic_endpoint(path: Path, endpoint: TTSServiceEndpoint) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(endpoint.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class CosyVoiceServiceRuntime:
    """Own case-scoped adapters sharing one loaded model and GPFS cache."""

    sample_rate = CosyVoiceTTS.sample_rate

    def __init__(
        self,
        *,
        model_dir: Path,
        catalog_path: Path,
        binding_manifest_path: Path,
        cache_dir: Path,
        speaker_feature_cache_path: Path | None = None,
        device: str | None = "cpu",
        replica_id: str = "replica-0",
        backend_factory: Callable[..., Any] = CosyVoiceTTS,
    ) -> None:
        self.model_dir = model_dir.expanduser().resolve()
        self.catalog_path = catalog_path.expanduser().resolve()
        self.binding_manifest_path = binding_manifest_path.expanduser().resolve()
        self.cache_dir = cache_dir.expanduser().resolve()
        self.speaker_feature_cache_path = (
            speaker_feature_cache_path.expanduser().resolve()
            if speaker_feature_cache_path is not None
            else None
        )
        self.device = device
        self.replica_id = replica_id
        self.backend_factory = backend_factory
        self.catalog = VoiceCatalogManifest.load(self.catalog_path)
        self.binding_manifest = GlobalVoiceBindingManifest.load(
            self.binding_manifest_path
        )
        if self.binding_manifest.catalog_version != self.catalog.catalog_version:
            raise ValueError("binding and voice catalog versions do not match")
        self.catalog_sha256 = file_sha256(self.catalog_path)
        self.binding_sha256 = file_sha256(self.binding_manifest_path)
        if self.binding_manifest.catalog_sha256 != self.catalog_sha256:
            raise ValueError("binding manifest catalog_sha256 does not match catalog")
        self._bindings_by_case: dict[str, dict[str, Any]] = {}
        for item in self.binding_manifest.bindings:
            self._bindings_by_case.setdefault(item.case_id, {})[
                item.oracle_user_id
            ] = item
        self._backends: dict[str, Any] = {}
        self._registration_by_key: dict[tuple[str, str], TTSRegistrationResult] = {}
        self._registration_meta: dict[tuple[str, str], dict[str, Any]] = {}
        self._inflight: dict[str, dict[str, Any]] = {}
        self._released_count = 0
        self._broken_pipe_count = 0
        self._lock = threading.RLock()

    def _new_backend(self, registration: TTSCaseRegistration) -> Any:
        return self.backend_factory(
            model_dir=self.model_dir,
            catalog_path=self.catalog_path,
            binding_manifest_path=self.binding_manifest_path,
            speaker_feature_cache_path=self.speaker_feature_cache_path,
            case_id=registration.case_id,
            assignment_seed=registration.assignment_seed,
            cache_dir=self.cache_dir,
            device=self.device,
        )

    @staticmethod
    def _user(profile: TTSProfile) -> Any:
        return SimpleNamespace(
            user_id=profile.user_id,
            voice=SimpleNamespace(
                voice_id=profile.voice_id,
                accent_label=profile.accent_label,
                speaking_style=profile.speaking_style,
            ),
        )

    def register(self, registration: TTSCaseRegistration) -> TTSRegistrationResult:
        expected = self._bindings_by_case.get(registration.case_id)
        if expected is None:
            raise ValueError(f"case is absent from the binding manifest: {registration.case_id}")
        supplied = {item.user_id: item for item in registration.profiles}
        if set(supplied) != set(expected):
            raise ValueError(
                "registration users do not exactly match the frozen case binding"
            )
        wrong_voices = {
            user_id: (profile.voice_id, expected[user_id].voice_id)
            for user_id, profile in supplied.items()
            if profile.voice_id != expected[user_id].voice_id
        }
        if wrong_voices:
            raise ValueError(f"registration voice IDs disagree with binding: {wrong_voices}")
        profile_digest = registration.profile_digest()
        key = (registration.run_namespace, registration.case_id)
        with self._lock:
            prior = self._registration_by_key.get(key)
            if prior is not None:
                if prior.profile_digest != profile_digest:
                    raise RegistrationConflict(
                        "case was already registered with a different profile digest",
                        diagnostics={
                            "run_namespace": registration.run_namespace,
                            "case_id": registration.case_id,
                            "existing_profile_digest": prior.profile_digest,
                            "incoming_profile_digest": profile_digest,
                            "existing_registration_id": prior.registration_id,
                            "existing_registered_at_unix": self._registration_meta[key]["registered_at_unix"],
                            "replica_id": self.replica_id,
                        },
                    )
                return prior
            backend = self._new_backend(registration)
            backend.register_profiles(
                [self._user(item) for item in registration.profiles]
            )
            registration_id = hashlib.sha256(
                f"{registration.run_namespace}:{registration.case_id}:{profile_digest}".encode("utf-8")
            ).hexdigest()
            assigned = {
                voice_id: entry.voice_catalog_id
                for voice_id, entry in backend.assigned_catalog_entries().items()
            }
            result = TTSRegistrationResult(
                run_namespace=registration.run_namespace,
                case_id=registration.case_id,
                registration_id=registration_id,
                profile_digest=profile_digest,
                sample_rate=self.sample_rate,
                assigned_voice_catalog_ids=assigned,
                provenance=backend.provenance(),
            )
            self._backends[registration_id] = backend
            self._registration_by_key[key] = result
            self._registration_meta[key] = {
                "registered_at_unix": time.time(),
                "profile_digest": profile_digest,
                "registration_id": registration_id,
            }
            return result

    def synthesize(self, request: TTSSynthesisRequest) -> bytes:
        with self._lock:
            key = (request.run_namespace, request.case_id)
            registration = self._registration_by_key.get(key)
            backend = self._backends.get(request.registration_id)
        if registration is None or backend is None:
            raise RegistrationConflict(
                "unknown or expired TTS registration",
                diagnostics={
                    "run_namespace": request.run_namespace,
                    "case_id": request.case_id,
                    "registration_id": request.registration_id,
                    "replica_id": self.replica_id,
                },
            )
        if registration.registration_id != request.registration_id:
            raise RegistrationConflict(
                "registration does not belong to case",
                diagnostics={
                    "run_namespace": request.run_namespace,
                    "case_id": request.case_id,
                    "registration_id": request.registration_id,
                    "expected_registration_id": registration.registration_id,
                    "replica_id": self.replica_id,
                },
            )
        if request.voice_id not in registration.assigned_voice_catalog_ids:
            raise ValueError(f"voice was not registered for case: {request.voice_id}")
        pcm = backend.synthesize(request.text, request.voice_id)
        if not pcm or len(pcm) % 2:
            raise RuntimeError("CosyVoice returned empty or non-PCM16-aligned audio")
        return pcm

    def begin_synthesis(self, request: TTSSynthesisRequest) -> str:
        token = uuid.uuid4().hex
        with self._lock:
            self._inflight[token] = {
                "run_namespace": request.run_namespace,
                "case_id": request.case_id,
                "registration_id": request.registration_id,
                "voice_id": request.voice_id,
                "started_at_unix": time.time(),
                "text_sha256": hashlib.sha256(request.text.encode("utf-8")).hexdigest(),
            }
        return token

    def end_synthesis(self, token: str, *, status: str, error: str | None = None) -> None:
        with self._lock:
            item = self._inflight.pop(token, None)
        if item is not None:
            item.update(status=status, error=error, ended_at_unix=time.time())
            print(json.dumps({"component": "mu-tts-http", "event": "synthesis_finished", **item},
                             ensure_ascii=False), flush=True)

    def mark_client_disconnect(self, token: str, exc: BaseException) -> None:
        with self._lock:
            self._broken_pipe_count += 1
            item = dict(self._inflight.get(token, {}))
        print(json.dumps({
            "component": "mu-tts-http",
            "event": "client_disconnected_during_synthesis_response",
            "error_type": type(exc).__name__,
            "error": str(exc),
            **item,
        }, ensure_ascii=False), flush=True)

    def release(self, release: TTSCaseRelease) -> dict[str, Any]:
        key = (release.run_namespace, release.case_id)
        with self._lock:
            registration = self._registration_by_key.get(key)
            if registration is None:
                return {"released": False, "reason": "not_registered_or_already_released",
                        "run_namespace": release.run_namespace, "case_id": release.case_id,
                        "replica_id": self.replica_id, "active_registrations": len(self._registration_by_key)}
            if release.registration_id is not None and registration.registration_id != release.registration_id:
                raise RegistrationConflict(
                    "release registration_id does not match active case registration",
                    diagnostics={
                        "run_namespace": release.run_namespace,
                        "case_id": release.case_id,
                        "incoming_registration_id": release.registration_id,
                        "expected_registration_id": registration.registration_id,
                        "replica_id": self.replica_id,
                    },
                )
            backend = self._backends.pop(registration.registration_id, None)
            self._registration_by_key.pop(key, None)
            self._registration_meta.pop(key, None)
            self._released_count += 1
        for method_name in ("close", "release", "finish"):
            method = getattr(backend, method_name, None)
            if callable(method):
                try:
                    method()
                except TypeError:
                    continue
                except Exception as exc:
                    print(json.dumps({
                        "component": "mu-tts-http",
                        "event": "backend_release_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "registration_id": registration.registration_id,
                    }, ensure_ascii=False), flush=True)
                break
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return {"released": True, "run_namespace": release.run_namespace,
                "case_id": release.case_id, "registration_id": registration.registration_id,
                "replica_id": self.replica_id}

    def runtime_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_registrations": len(self._registration_by_key),
                "active_backends": len(self._backends),
                "inflight_synthesis": len(self._inflight),
                "released_count": self._released_count,
                "broken_pipe_count": self._broken_pipe_count,
                "inflight": dict(self._inflight),
            }

    def capabilities(self) -> dict[str, Any]:
        return {
            "schema_version": "mu-bench-tts-capabilities-v1",
            "backend": "cosyvoice_http",
            "transport": "synchronous_raw_pcm_s16le_http",
            "sample_rate": self.sample_rate,
            "renderer_version": CosyVoiceTTS.renderer_version,
            "validation_version": CosyVoiceTTS.validation_version,
            "catalog_sha256": self.catalog_sha256,
            "binding_sha256": self.binding_sha256,
            "cache_dir": str(self.cache_dir),
            "replica_id": self.replica_id,
            "run_namespace_supported": True,
            "registration_schema": "mu-bench-tts-register-v2",
            "future_case_state_received": False,
            **self.runtime_stats(),
        }

    def warm_model(self) -> None:
        """Load model weights before publishing a ready endpoint."""
        first_case_id = next(iter(self._bindings_by_case), None)
        if first_case_id is None:
            raise RuntimeError("binding manifest has no cases")
        backend = self.backend_factory(
            model_dir=self.model_dir,
            catalog_path=self.catalog_path,
            binding_manifest_path=self.binding_manifest_path,
            speaker_feature_cache_path=self.speaker_feature_cache_path,
            case_id=first_case_id,
            assignment_seed=self.binding_manifest.assignment_seed,
            cache_dir=self.cache_dir,
            device=self.device,
        )
        ensure_model = getattr(backend, "_ensure_model", None)
        if ensure_model is None:
            raise RuntimeError("CosyVoice backend cannot be warmed")
        ensure_model()


class CosyVoiceHTTPServer:
    """Authenticated HTTP wrapper around one persistent renderer replica."""

    def __init__(
        self,
        *,
        runtime: CosyVoiceServiceRuntime,
        host: str = "0.0.0.0",
        port: int = 0,
        api_key: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.api_key = api_key
        self.server = ThreadingHTTPServer((host, port), self._handler_type())

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        runtime = self.runtime
        expected_key = self.api_key

        class Handler(BaseHTTPRequestHandler):
            server_version = "mu-bench-cosyvoice/1.0"

            def log_message(self, format: str, *args: object) -> None:
                print(
                    json.dumps(
                        {
                            "component": "mu-tts-http",
                            "client": self.client_address[0],
                            "message": format % args,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            def _authorized(self) -> bool:
                if expected_key is None:
                    return True
                supplied = self.headers.get("Authorization", "")
                return hmac.compare_digest(supplied, f"Bearer {expected_key}")

            def _json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _pcm(self, pcm: bytes) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(pcm)))
                self.send_header("X-Sample-Rate", str(runtime.sample_rate))
                self.send_header("X-PCM-SHA256", hashlib.sha256(pcm).hexdigest())
                self.end_headers()
                self.wfile.write(pcm)

            def _body(self) -> dict[str, Any]:
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError as exc:
                    raise ValueError("invalid Content-Length") from exc
                if length <= 0 or length > 1024 * 1024:
                    raise ValueError("request body must be between 1 byte and 1 MiB")
                value = json.loads(self.rfile.read(length))
                if not isinstance(value, dict):
                    raise ValueError("request body must be a JSON object")
                return value

            def _guard(self) -> bool:
                if self._authorized():
                    return True
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return False

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                if path == "/health/live":
                    self._json(HTTPStatus.OK, {"live": True})
                    return
                if not self._guard():
                    return
                if path == "/health/ready":
                    self._json(
                        HTTPStatus.OK,
                        {
                            "ready": True,
                            "backend": "cosyvoice_http",
                            "replica_id": runtime.replica_id,
                            **runtime.runtime_stats(),
                        },
                    )
                elif path == "/mu/v1/tts/capabilities":
                    self._json(HTTPStatus.OK, runtime.capabilities())
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

            def do_POST(self) -> None:  # noqa: N802
                if not self._guard():
                    return
                path = urlparse(self.path).path
                try:
                    if path == "/mu/v1/tts/cases/register":
                        request = TTSCaseRegistration.model_validate(self._body())
                        result = runtime.register(request)
                        self._json(HTTPStatus.OK, result.model_dump(mode="json"))
                    elif path == "/mu/v1/tts/cases/release":
                        request = TTSCaseRelease.model_validate(self._body())
                        self._json(HTTPStatus.OK, runtime.release(request))
                    elif path == "/mu/v1/tts/synthesize":
                        request = TTSSynthesisRequest.model_validate(self._body())
                        token = runtime.begin_synthesis(request)
                        try:
                            pcm = runtime.synthesize(request)
                            self._pcm(pcm)
                            runtime.end_synthesis(token, status="ok")
                        except (BrokenPipeError, ConnectionResetError) as exc:
                            runtime.mark_client_disconnect(token, exc)
                            runtime.end_synthesis(token, status="client_disconnected", error=str(exc))
                        except Exception as exc:
                            runtime.end_synthesis(token, status="error", error=f"{type(exc).__name__}: {exc}")
                            raise
                    else:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                except RegistrationConflict as exc:
                    payload = {
                        "ok": False,
                        **exc.as_dict(),
                        "path": path,
                        "replica_id": runtime.replica_id,
                        "timestamp_unix": time.time(),
                    }
                    print(json.dumps(payload, ensure_ascii=False), flush=True)
                    self._json(HTTPStatus.CONFLICT, payload)
                except (ValueError, json.JSONDecodeError) as exc:
                    payload = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "path": path,
                        "replica_id": runtime.replica_id,
                        "timestamp_unix": time.time(),
                    }
                    print(json.dumps(payload, ensure_ascii=False), flush=True)
                    self._json(HTTPStatus.BAD_REQUEST, payload)
                except Exception as exc:  # noqa: BLE001
                    payload = {
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "path": path,
                        "replica_id": runtime.replica_id,
                        "timestamp_unix": time.time(),
                        "traceback_tail": traceback.format_exc().splitlines()[-12:],
                    }
                    print(json.dumps(payload, ensure_ascii=False), flush=True)
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, payload)

        return Handler

    def serve_forever(self) -> None:
        self.server.serve_forever(poll_interval=0.25)

    def shutdown(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def load_api_key(path: Path | None) -> str | None:
    if path is None:
        return None
    value = path.expanduser().resolve().read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"empty API key file: {path}")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--cosyvoice-repo", type=Path, required=True)
    parser.add_argument("--voice-catalog", type=Path, required=True)
    parser.add_argument("--voice-binding-manifest", type=Path, required=True)
    parser.add_argument("--speaker-feature-cache", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-num-threads", type=int)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--advertise-host")
    parser.add_argument("--api-key-file", type=Path, required=True)
    parser.add_argument("--endpoint-file", type=Path, required=True)
    parser.add_argument("--replica-id", required=True)
    return parser.parse_args()


def endpoint_for(
    *,
    runtime: CosyVoiceServiceRuntime,
    host: str,
    port: int,
    api_key_file: Path,
    started_at_unix: float,
    ready: bool,
) -> TTSServiceEndpoint:
    return TTSServiceEndpoint(
        ready=ready,
        host=host,
        port=port,
        base_url=f"http://{host}:{port}",
        hostname=socket.gethostname(),
        pid=os.getpid(),
        api_key_file=str(api_key_file.expanduser().resolve()),
        started_at_unix=started_at_unix,
        updated_at_unix=time.time(),
        renderer_version=CosyVoiceTTS.renderer_version,
        validation_version=CosyVoiceTTS.validation_version,
        sample_rate=runtime.sample_rate,
        catalog_sha256=runtime.catalog_sha256,
        binding_sha256=runtime.binding_sha256,
        cache_dir=str(runtime.cache_dir),
        replica_id=runtime.replica_id,
    )


def main() -> None:
    args = parse_args()
    if args.torch_num_threads is not None:
        if args.torch_num_threads < 1:
            raise ValueError("--torch-num-threads must be positive")
        import torch

        torch.set_num_threads(args.torch_num_threads)
        torch.set_num_interop_threads(max(1, min(4, args.torch_num_threads)))
    for path in (
        args.model_dir,
        args.cosyvoice_repo,
        args.voice_catalog,
        args.voice_binding_manifest,
        args.api_key_file,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.speaker_feature_cache is not None and not args.speaker_feature_cache.is_file():
        raise FileNotFoundError(args.speaker_feature_cache)
    for source in (
        args.cosyvoice_repo.expanduser().resolve(),
        args.cosyvoice_repo.expanduser().resolve() / "third_party" / "Matcha-TTS",
    ):
        if str(source) not in sys.path:
            sys.path.append(str(source))
    os.environ.setdefault("MU_BENCH_COSYVOICE_OFFLINE_TEXT", "1")
    runtime = CosyVoiceServiceRuntime(
        model_dir=args.model_dir,
        catalog_path=args.voice_catalog,
        binding_manifest_path=args.voice_binding_manifest,
        speaker_feature_cache_path=args.speaker_feature_cache,
        cache_dir=args.cache_dir,
        device=args.device,
        replica_id=args.replica_id,
    )
    runtime.warm_model()
    api_key = load_api_key(args.api_key_file)
    server = CosyVoiceHTTPServer(
        runtime=runtime,
        host=args.host,
        port=args.port,
        api_key=api_key,
    )
    advertise_host = discover_ipv4(args.advertise_host)
    started_at_unix = time.time()
    atomic_endpoint(
        args.endpoint_file,
        endpoint_for(
            runtime=runtime,
            host=advertise_host,
            port=args.port,
            api_key_file=args.api_key_file,
            started_at_unix=started_at_unix,
            ready=True,
        ),
    )
    print(
        json.dumps(
            {
                "ready": True,
                "transport": "synchronous_raw_pcm_s16le_http",
                "endpoint_file": str(args.endpoint_file.expanduser().resolve()),
                "replica_id": args.replica_id,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        atomic_endpoint(
            args.endpoint_file,
            endpoint_for(
                runtime=runtime,
                host=advertise_host,
                port=args.port,
                api_key_file=args.api_key_file,
                started_at_unix=started_at_unix,
                ready=False,
            ),
        )
        server.server.server_close()


if __name__ == "__main__":
    main()
