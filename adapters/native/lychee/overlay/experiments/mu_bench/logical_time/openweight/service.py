"""Opt-in HTTP control plane with one dedicated native process per session.

Not launched by importing this module. Do not point it at existing legacy worker
ports. The worker loads no API keys and makes no planner/TTS/Judge network calls.
"""
from __future__ import annotations

import base64
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import multiprocessing
import os
from pathlib import Path
import secrets
import subprocess
import threading
import time
import traceback
from typing import Any
import uuid

from .session import StatefulSession, SessionProtocolError


class NativeWorkerError(RuntimeError):
    """A native child failed; retain its structured error for the HTTP client."""

    def __init__(self, payload):
        self.payload = dict(payload) if isinstance(payload, dict) else {}
        message = (
            f"{self.payload.get('error_type', 'NativeError')}: "
            f"{self.payload.get('error', 'native worker failed')}"
        )
        super().__init__(message)

    def to_dict(self):
        return {"native": self.payload}


def _query_gpu_free_bytes():
    """Read free VRAM without initializing torch in the control process."""
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
        values = [int(line.strip()) for line in raw.splitlines() if line.strip()]
        return values[0] * 1024 * 1024 if values else None
    except Exception:
        return None


def _child(connection, native_config, session_config):
    session = None
    try:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        if native_config.get("model") == "test_fake":
            from .testing import DeterministicNative
            backend = DeterministicNative(quantum_ms=native_config.get("native_quantum_ms", 80))
        else:
            from .native import build_native_backend
            backend = build_native_backend(native_config)
        session = StatefulSession(backend, **session_config)
        connection.send({"ok": True, "result": session.status()})
        while True:
            command = connection.recv()
            try:
                operation = command.pop("operation")
                if operation == "advance":
                    result = session.advance(**command)
                elif operation in ("pause", "resume", "status", "finish"):
                    result = getattr(session, operation)()
                else:
                    raise SessionProtocolError("unknown native operation")
                connection.send({"ok": True, "result": result})
                if operation == "finish":
                    return
            except Exception as exc:
                connection.send({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
    except Exception as exc:
        try:
            connection.send({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if session is not None and not session.closed:
            session.finish()
        connection.close()


class SessionActor:
    def __init__(self, native_config, session_config, *, timeout_seconds=1800):
        context = multiprocessing.get_context("spawn")
        self.connection, child_connection = context.Pipe()
        self.process = context.Process(target=_child, args=(child_connection, native_config, session_config))
        self.process.start()
        child_connection.close()
        self.lock = threading.Lock()
        self.timeout_seconds = timeout_seconds
        self.unresolved = False
        self.closed = False
        self.initial_status = None

    def wait_ready(self):
        self.initial_status = self._receive()
        return self.initial_status

    def _receive(self):
        if not self.connection.poll(self.timeout_seconds):
            self.unresolved = True
            raise SessionProtocolError("native request timed out; state uncertain, never replay automatically")
        try:
            payload = self.connection.recv()
        except EOFError as exc:
            raise SessionProtocolError("native process exited; session invalid, no transcript-based recovery") from exc
        if not payload["ok"]:
            raise NativeWorkerError(payload)
        return payload["result"]

    def call(self, operation, **kwargs):
        with self.lock:
            if self.unresolved:
                raise SessionProtocolError("prior native operation unresolved; do not retry in a new context")
            if self.closed or not self.process.is_alive():
                raise SessionProtocolError("native actor is terminal")
            self.connection.send({"operation": operation, **kwargs})
            result = self._receive()
            if operation == "finish":
                self.closed = True
                self.process.join(timeout=10)
            return result

    def abort(self):
        """Best-effort terminal cleanup after an unresolved client failure."""
        with self.lock:
            if self.closed:
                return {"aborted": False, "already_closed": True}
            self.closed = True
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=10)
            try:
                self.connection.close()
            except OSError:
                pass
            return {"aborted": True, "session_id": self.initial_status.get("session_id") if isinstance(self.initial_status, dict) else None}


class SessionManager:
    def __init__(self, *, native_config: dict[str, Any], allowed_roots: list[str],
                 output_root: str, max_sessions: int = 1, timeout_seconds=1800,
                 allow_test_backend=False):
        if native_config.get("model") == "test_fake" and not allow_test_backend:
            raise SessionProtocolError("test backend must be explicitly enabled; never a model score")
        self.native_config = dict(native_config)
        self.roots = [Path(p).resolve() for p in allowed_roots]
        self.output_root = self.resolve(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.max_sessions = max_sessions
        self.timeout_seconds = timeout_seconds
        self.actors: dict[str, SessionActor] = {}
        self.starts: dict[str, tuple[str, str]] = {}
        self.lock = threading.Lock()
        self.active_reservations = 0
        # The previous global policy rejected Lychee below 57,344 MiB free,
        # although the observed native peak is about 29 GiB and the allocator
        # cap is 32 GiB.  Keep a versioned, per-session guard instead of
        # silently bypassing admission.  The value is intentionally
        # configurable so a future image can tighten it without code edits.
        self.session_min_free_mib = int(
            os.environ.get("MU_LYCHEE_MIN_FREE_GPU_MIB", "40960")
        ) if self.native_config.get("model") == "lychee" else 0
        self.session_admission = {
            "model": self.native_config.get("model"),
            "min_free_mib": self.session_min_free_mib,
            "old_policy_min_free_mib": 57344 if self.native_config.get("model") == "lychee" else None,
            "last_check": None,
        }

    def resolve(self, value: str) -> Path:
        path = Path(value).resolve()
        if not any(path == root or root in path.parents for root in self.roots):
            raise SessionProtocolError("artifact path outside configured shared roots")
        return path

    def start(self, payload):
        allowed = {"request_id", "seed", "max_media_ms"}
        if set(payload) - allowed:
            raise SessionProtocolError("start accepts no prompt, roster, future track, or model override")
        request_id = str(payload.get("request_id", ""))
        if not request_id:
            raise SessionProtocolError("start request_id required")
        fp = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        # Hold start lock through load: an ambiguous timeout cannot create a
        # second GPU model context. Other per-session actions use actor locks.
        with self.lock:
            if request_id in self.starts:
                previous_fp, sid = self.starts[request_id]
                if previous_fp != fp:
                    raise SessionProtocolError("start request_id collision")
                actor = self.actors.get(sid)
                if actor is None:
                    raise SessionProtocolError("prior start failed; requires a new explicit attempt")
                if actor.initial_status is None:
                    raise SessionProtocolError("prior start unresolved or failed; no duplicate actor")
                return actor.initial_status
            if sum(not a.closed and a.process.is_alive() for a in self.actors.values()) >= self.max_sessions:
                raise SessionProtocolError("maximum live sessions reached")
            if self.session_min_free_mib:
                free_bytes = _query_gpu_free_bytes()
                check = {
                    "free_bytes": free_bytes,
                    "free_mib": (free_bytes / (1024 * 1024)) if free_bytes is not None else None,
                    "min_free_mib": self.session_min_free_mib,
                    "old_policy_min_free_mib": 57344,
                    "model": self.native_config.get("model"),
                    "decision": "allow" if free_bytes is None or free_bytes >= self.session_min_free_mib * 1024 * 1024 else "reject",
                    "timestamp_unix": time.time(),
                }
                self.session_admission["last_check"] = check
                if free_bytes is not None and free_bytes < self.session_min_free_mib * 1024 * 1024:
                    raise SessionProtocolError(
                        "Insufficient free GPU memory for Lychee session admission: "
                        f"{free_bytes} < {self.session_min_free_mib * 1024 * 1024} bytes "
                        f"(free_mib={check['free_mib']:.0f}, min_free_mib={self.session_min_free_mib})"
                    )
            sid = uuid.uuid4().hex
            self.starts[request_id] = (fp, sid)
            actor = SessionActor(self.native_config, dict(
                session_id=sid, seed=int(payload.get("seed", 0)),
                max_media_ms=int(payload["max_media_ms"]), output_dir=str(self.output_root / sid)),
                timeout_seconds=self.timeout_seconds)
            self.actors[sid] = actor
            return actor.wait_ready()

    def call(self, sid, operation, payload):
        if sid not in self.actors:
            raise SessionProtocolError("unknown session")
        if operation == "advance":
            allowed = {"pcm16_b64", "pcm_path", "pcm_sha256", "sim_start_ms", "sequence", "request_id"}
            if set(payload) - allowed:
                raise SessionProtocolError("unexpected advance fields")
            if bool(payload.get("pcm_path")) == bool(payload.get("pcm16_b64")):
                raise SessionProtocolError("provide exactly one current audio segment transport")
            if payload.get("pcm_path"):
                path = self.resolve(payload["pcm_path"])
                if path.stat().st_size > 16 * 1024 * 1024:
                    raise SessionProtocolError("current segment exceeds 16 MiB")
                pcm = path.read_bytes()
                if not payload.get("pcm_sha256") or hashlib.sha256(pcm).hexdigest() != payload["pcm_sha256"]:
                    raise SessionProtocolError("segment hash mismatch")
            else:
                pcm = base64.b64decode(payload["pcm16_b64"], validate=True)
            return self.actors[sid].call("advance", pcm16=pcm, sim_start_ms=int(payload["sim_start_ms"]),
                                         sequence=int(payload["sequence"]), request_id=payload["request_id"])
        if payload:
            raise SessionProtocolError("control operation takes no payload")
        if operation == "abort":
            if payload:
                raise SessionProtocolError("abort operation takes no payload")
            return self.actors[sid].abort()
        if operation not in ("pause", "resume", "status", "finish"):
            raise SessionProtocolError("unknown operation")
        return self.actors[sid].call(operation)


def make_server(manager: SessionManager, *, host: str, port: int, token: str):
    if len(token) < 24:
        raise ValueError("service token must contain at least 24 characters")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Avoid logging credentials or speech payloads.

        def do_GET(self):
            if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_error(401); return
            if self.path != "/health":
                self.send_error(404); return
            data = {"protocol": "mu-v14-stateful-logical-v1", "model": manager.native_config["model"],
                    "active_sessions": sum(not a.closed and a.process.is_alive() for a in manager.actors.values()),
                    "session_ids": [sid for sid, actor in manager.actors.items()
                                    if not actor.closed and actor.process.is_alive()],
                    "session_admission": manager.session_admission,
                    "native_gpu_validation": "not_performed_by_this_implementation", "model_step_performed": False}
            raw = json.dumps(data).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        def do_POST(self):
            if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_error(401)
                return
            try:
                count = int(self.headers.get("Content-Length", "0"))
                if not 0 < count <= 24 * 1024 * 1024:
                    raise SessionProtocolError("invalid request size")
                payload = json.loads(self.rfile.read(count))
                parts = self.path.strip("/").split("/")
                if parts == ["sessions"]:
                    result = manager.start(payload)
                elif len(parts) == 3 and parts[0] == "sessions":
                    result = manager.call(parts[1], parts[2], payload)
                else:
                    raise SessionProtocolError("unknown endpoint")
                status, data = 200, {"ok": True, "result": result}
            except Exception as exc:
                # 409 remains reserved for a deterministic protocol/session
                # conflict. Native model failures are 500 so a client cannot
                # mistake them for a recoverable admission conflict.
                # A child-process failure is an execution failure even when the
                # child happened to serialize a SessionProtocolError.  Reserve
                # 409 for validation/admission errors raised directly by the
                # HTTP control plane; otherwise callers cannot distinguish a
                # missing native dependency (for example s3tokenizer) from a
                # recoverable session conflict.
                status = 409 if isinstance(exc, SessionProtocolError) else 500
                data = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "path": self.path,
                    "timestamp_unix": time.time(),
                }
                details = getattr(exc, "to_dict", None)
                if callable(details):
                    data["details"] = details()
                if status >= 500:
                    data["traceback_tail"] = traceback.format_exc().splitlines()[-16:]
                print(json.dumps(data, ensure_ascii=False), flush=True)
            raw = json.dumps(data).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw))); self.end_headers()
            self.wfile.write(raw)

    return ThreadingHTTPServer((host, port), Handler)
