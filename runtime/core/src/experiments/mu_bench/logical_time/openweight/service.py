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
import signal
import socket
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


def _child(connection, native_config, session_config):
    session = None
    try:
        try:
            os.setsid()
        except OSError:
            pass
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
                try:
                    connection.send({
                        "ok": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "terminal_after_error": True,
                    })
                finally:
                    return
    except Exception as exc:
        try:
            connection.send({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if session is not None and not session.closed:
            try:
                session.finish()
            except Exception as exc:  # pragma: no cover - last-resort cleanup telemetry
                print(json.dumps({
                    "component": "mu-openweight-native-child",
                    "event": "finish_after_error_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "session_id": getattr(session, "session_id", None),
                    "timestamp_unix": time.time(),
                }, ensure_ascii=False), flush=True)
        connection.close()


def _terminate_process_group(process: multiprocessing.Process, *, grace_seconds: float = 8.0) -> dict[str, Any]:
    pid = process.pid
    result: dict[str, Any] = {"pid": pid, "terminated": False, "killed": False}
    if pid is None:
        return result
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        result["already_exited"] = True
        return result
    except OSError as exc:
        pgid = None
        result["pgid_error"] = f"{type(exc).__name__}: {exc}"
    result["pgid"] = pgid
    try:
        if pgid is not None:
            os.killpg(pgid, signal.SIGTERM)
        else:
            process.terminate()
        result["terminated"] = True
    except ProcessLookupError:
        result["already_exited"] = True
        return result
    process.join(timeout=grace_seconds)
    if process.is_alive():
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGKILL)
            else:
                process.kill()
            result["killed"] = True
        except ProcessLookupError:
            result["already_exited"] = True
        process.join(timeout=3)
    result["alive_after"] = process.is_alive()
    return result


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
        self.last_status = None
        self.last_error = None

    def wait_ready(self):
        self.initial_status = self._decorate_status(self._receive())
        self.last_status = self.initial_status
        return self.initial_status

    def _decorate_status(self, result):
        if isinstance(result, dict):
            value = dict(result)
            value["native_actor_pid"] = self.process.pid
            try:
                value["native_actor_pgid"] = os.getpgid(self.process.pid) if self.process.pid else None
            except OSError:
                value["native_actor_pgid"] = None
            value["native_actor_alive"] = self.process.is_alive()
            return value
        return result

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
            try:
                self.connection.send({"operation": operation, **kwargs})
                result = self._decorate_status(self._receive())
                if isinstance(result, dict):
                    self.last_status = result
            except Exception as exc:
                self.unresolved = True
                self.last_error = {"error_type": type(exc).__name__, "error": str(exc), "operation": operation}
                cleanup = _terminate_process_group(self.process)
                self.closed = True
                try:
                    self.connection.close()
                except OSError:
                    pass
                self.last_error["cleanup"] = cleanup
                raise
            if operation == "finish":
                self.closed = True
                self.process.join(timeout=10)
                if self.process.is_alive():
                    _terminate_process_group(self.process)
            return result

    def abort(self, *, reason: str = "explicit_abort"):
        """Best-effort terminal cleanup after an unresolved client failure."""
        with self.lock:
            if self.closed:
                return {"aborted": False, "already_closed": True}
            self.closed = True
            cleanup = {"alive_before": self.process.is_alive(), "reason": reason}
            if self.process.is_alive():
                cleanup.update(_terminate_process_group(self.process))
            try:
                self.connection.close()
            except OSError:
                pass
            return {"aborted": True,
                    "session_id": self.initial_status.get("session_id") if isinstance(self.initial_status, dict) else None,
                    "cleanup": cleanup}

    def summary(self) -> dict[str, Any]:
        alive = self.process.is_alive()
        try:
            pgid = os.getpgid(self.process.pid) if self.process.pid else None
        except OSError:
            pgid = None
        status = self.last_status if isinstance(self.last_status, dict) else {}
        return {
            "pid": self.process.pid,
            "pgid": pgid,
            "alive": alive,
            "closed": self.closed,
            "unresolved": self.unresolved,
            "exitcode": self.process.exitcode,
            "session_id": status.get("session_id"),
            "logical_ms": status.get("logical_ms"),
            "processed_end_ms": status.get("processed_end_ms"),
            "buffered_input_bytes": status.get("buffered_input_bytes"),
            "pending_output_samples": status.get("pending_output_samples"),
            "unplayed_bytes": status.get("unplayed_bytes"),
            "gpu_memory_allocated_bytes": status.get("gpu_memory_allocated_bytes"),
            "gpu_memory_reserved_bytes": status.get("gpu_memory_reserved_bytes"),
            "gpu_memory_active_bytes": status.get("gpu_memory_active_bytes"),
            "last_error": self.last_error,
        }


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

    def _prune_locked(self) -> list[str]:
        stale: list[str] = []
        for sid, actor in list(self.actors.items()):
            if actor.closed or not actor.process.is_alive():
                stale.append(sid)
                try:
                    actor.connection.close()
                except OSError:
                    pass
                self.actors.pop(sid, None)
        if stale:
            stale_set = set(stale)
            for request_id, (_, sid) in list(self.starts.items()):
                if sid in stale_set or sid not in self.actors:
                    self.starts.pop(request_id, None)
        return stale

    def _active_count_locked(self) -> int:
        self._prune_locked()
        return sum(not a.closed and a.process.is_alive() for a in self.actors.values())

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
            self._prune_locked()
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
            if self._active_count_locked() >= self.max_sessions:
                raise SessionProtocolError("maximum live sessions reached")
            sid = uuid.uuid4().hex
            self.starts[request_id] = (fp, sid)
            actor = SessionActor(self.native_config, dict(
                session_id=sid, seed=int(payload.get("seed", 0)),
                max_media_ms=int(payload["max_media_ms"]), output_dir=str(self.output_root / sid)),
                timeout_seconds=self.timeout_seconds)
            self.actors[sid] = actor
            try:
                return actor.wait_ready()
            except Exception:
                actor.abort(reason="start_failed")
                self.actors.pop(sid, None)
                self.starts.pop(request_id, None)
                raise

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
            try:
                return self.actors[sid].call("advance", pcm16=pcm, sim_start_ms=int(payload["sim_start_ms"]),
                                             sequence=int(payload["sequence"]), request_id=payload["request_id"])
            finally:
                with self.lock:
                    self._prune_locked()
        if payload:
            raise SessionProtocolError("control operation takes no payload")
        if operation == "abort":
            if payload:
                raise SessionProtocolError("abort operation takes no payload")
            result = self.actors[sid].abort(reason="client_abort")
            with self.lock:
                self._prune_locked()
            return result
        if operation not in ("pause", "resume", "status", "finish"):
            raise SessionProtocolError("unknown operation")
        try:
            return self.actors[sid].call(operation)
        finally:
            with self.lock:
                self._prune_locked()

    def abort_all(self, *, reason: str) -> dict[str, Any]:
        with self.lock:
            actors = list(self.actors.items())
        aborted = {}
        for sid, actor in actors:
            aborted[sid] = actor.abort(reason=reason)
        with self.lock:
            stale = self._prune_locked()
            active_sessions = self._active_count_locked()
        return {"aborted_sessions": aborted, "pruned_session_ids": stale,
                "active_sessions": active_sessions, "reason": reason}

    def health_payload(self) -> dict[str, Any]:
        with self.lock:
            pruned = self._prune_locked()
            active = [sid for sid, actor in self.actors.items() if not actor.closed and actor.process.is_alive()]
            actor_statuses = {sid: actor.summary() for sid, actor in self.actors.items()}
        try:
            service_pgid = os.getpgid(os.getpid())
        except OSError:
            service_pgid = None
        return {"protocol": "mu-v14-stateful-logical-v1", "model": self.native_config["model"],
                "active_sessions": len(active), "session_ids": active,
                "actor_statuses": actor_statuses, "pruned_session_ids": pruned,
                "service_pid": os.getpid(), "service_pgid": service_pgid,
                "hostname": socket.gethostname(),
                "native_gpu_validation": "not_performed_by_this_implementation",
                "model_step_performed": False}


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
            data = manager.health_payload()
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
                elif parts in (["admin", "reap"], ["admin", "abort_all"]):
                    reason = str(payload.get("reason", "admin_reap"))
                    result = manager.abort_all(reason=reason)
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
