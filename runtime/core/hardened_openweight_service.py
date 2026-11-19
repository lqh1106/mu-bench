#!/usr/bin/env python3
"""Fail-closed wrapper around the native μ-Bench logical-time service.

The historical service keeps dead actors and failed ``start`` reservations in
memory and returns the same HTTP status for every protocol/backend exception.
This wrapper retains its model implementation but adds actor reaping,
process-group cleanup, and structured error responses.  It is intentionally
versioned and does not modify an existing service in place.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import signal
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _load_legacy_service(legacy_src: Path):
    sys.path.insert(0, str(legacy_src.resolve()))
    from experiments.mu_bench.logical_time.openweight import service as legacy  # type: ignore

    return legacy


def _terminate_process_group(pid: int) -> dict[str, Any]:
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return {"pid": pid, "status": "already_gone"}
    own_pgid = os.getpgrp()
    signals: list[str] = []
    try:
        if pgid != own_pgid:
            os.killpg(pgid, signal.SIGTERM)
            signals.append("SIGTERM")
        else:
            os.kill(pid, signal.SIGTERM)
            signals.append("SIGTERM(pid-only)")
    except (ProcessLookupError, PermissionError):
        pass
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        proc_stat = Path(f"/proc/{pid}/stat")
        if proc_stat.is_file():
            try:
                fields = proc_stat.read_text(encoding="utf-8").split()
                if len(fields) > 2 and fields[2] == "Z":
                    return {"pid": pid, "pgid": pgid, "signals": signals, "status": "terminated"}
            except Exception:
                pass
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return {"pid": pid, "pgid": pgid, "signals": signals, "status": "terminated"}
        time.sleep(0.2)
    try:
        if pgid != own_pgid:
            os.killpg(pgid, signal.SIGKILL)
            signals.append("SIGKILL")
        else:
            os.kill(pid, signal.SIGKILL)
            signals.append("SIGKILL(pid-only)")
    except (ProcessLookupError, PermissionError):
        pass
    return {"pid": pid, "pgid": pgid, "signals": signals, "status": "terminated"}


def build_server(legacy, manager, *, host: str, port: int, token: str):
    if len(token) < 24:
        raise ValueError("service token must contain at least 24 characters")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: object) -> None:
            # The service log remains useful without exposing bearer tokens or
            # audio payloads.
            print(
                json.dumps(
                    {"component": "mu-openweight-service", "message": fmt % args},
                    ensure_ascii=False,
                ),
                flush=True,
            )

        def authorized(self) -> bool:
            return secrets.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + token
            )

        def reply(self, status: int, payload: dict[str, Any]) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 24 * 1024 * 1024:
                raise legacy.SessionProtocolError("invalid request size")
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise legacy.SessionProtocolError("request body must be an object")
            return value

        def do_GET(self) -> None:  # noqa: N802
            if not self.authorized():
                self.reply(401, {"ok": False, "error_type": "Unauthorized", "error": "unauthorized"})
                return
            if self.path != "/health":
                self.reply(404, {"ok": False, "error_type": "NotFound", "error": "unknown endpoint"})
                return
            manager.reap()
            active = sum(
                not a.closed and a.process.is_alive()
                for a in manager.actors.values()
            )
            self.reply(
                200,
                {
                    "ok": True,
                    "protocol": "mu-v14-stateful-logical-v2",
                    "model": manager.native_config.get("model"),
                    "active_sessions": active,
                    "actor_count": len(manager.actors),
                    "start_reservation_count": len(manager.starts),
                    "model_step_performed": False,
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            if not self.authorized():
                self.reply(401, {"ok": False, "error_type": "Unauthorized", "error": "unauthorized"})
                return
            try:
                manager.reap()
                payload = self.body()
                parts = self.path.strip("/").split("/")
                if parts == ["sessions"]:
                    result = manager.start(payload)
                elif len(parts) == 3 and parts[0] == "sessions":
                    result = manager.call(parts[1], parts[2], payload)
                else:
                    raise legacy.SessionProtocolError("unknown endpoint")
                self.reply(200, {"ok": True, "result": result})
            except Exception as exc:  # noqa: BLE001
                manager.reap()
                diagnostic = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "path": self.path,
                    "model": manager.native_config.get("model"),
                    "active_sessions": sum(
                        not a.closed and a.process.is_alive()
                        for a in manager.actors.values()
                    ),
                    "traceback_tail": traceback.format_exc().splitlines()[-12:],
                }
                print(json.dumps(diagnostic, ensure_ascii=False), flush=True)
                self.reply(409, diagnostic)

    server = ThreadingHTTPServer((host, port), Handler)
    # A stuck request thread must not keep a dead model service alive during
    # shutdown; actors are explicitly cleaned up in ``main`` below.
    server.daemon_threads = True
    return server


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--legacy-src", type=Path, required=True)
    p.add_argument("--native-config", type=Path, required=True)
    p.add_argument("--allowed-root", action="append", required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--api-key-file", type=Path, required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--max-sessions", type=int, default=1)
    p.add_argument("--timeout-seconds", type=int, default=1800)
    p.add_argument("--allow-test-backend", action="store_true")
    args = p.parse_args()
    legacy = _load_legacy_service(args.legacy_src)

    class HardenedManager(legacy.SessionManager):
        def reap(self) -> list[dict[str, Any]]:
            removed: list[dict[str, Any]] = []
            with self.lock:
                dead = [
                    sid for sid, actor in self.actors.items()
                    if actor.closed or not actor.process.is_alive()
                ]
                for sid in dead:
                    actor = self.actors.pop(sid, None)
                    if actor is not None:
                        try:
                            actor.connection.close()
                        except Exception:
                            pass
                        removed.append({"sid": sid, "pid": actor.process.pid})
                    for request_id, (_, reserved_sid) in list(self.starts.items()):
                        if reserved_sid == sid:
                            self.starts.pop(request_id, None)
            return removed

        def _abort_actor(self, sid: str) -> None:
            with self.lock:
                actor = self.actors.get(sid)
                if actor is None:
                    return
                actor.closed = True
                pid = actor.process.pid
                for request_id, (_, reserved_sid) in list(self.starts.items()):
                    if reserved_sid == sid:
                        self.starts.pop(request_id, None)
                self.actors.pop(sid, None)
            _terminate_process_group(pid)

        def start(self, payload):
            self.reap()
            before = set(self.actors)
            try:
                return super().start(payload)
            except Exception:
                # A failed/ambiguous model load must not reserve a dead start
                # or occupy the sole session slot for the next case.
                for sid in set(self.actors) - before:
                    self._abort_actor(sid)
                self.reap()
                raise

        def call(self, sid, operation, payload):
            self.reap()
            try:
                result = super().call(sid, operation, payload)
            except Exception:
                # Native operation exceptions are terminal for this actor;
                # callers receive the structured error and can start a clean
                # next case instead of replaying an uncertain request.
                self._abort_actor(sid)
                raise
            if operation == "finish":
                self._abort_actor(sid)
            return result

        def shutdown_all(self) -> None:
            with self.lock:
                session_ids = list(self.actors)
            for sid in session_ids:
                self._abort_actor(sid)
            self.reap()

    manager = HardenedManager(
        native_config=json.loads(args.native_config.read_text(encoding="utf-8")),
        allowed_roots=args.allowed_root,
        output_root=str(args.output_root),
        max_sessions=args.max_sessions,
        timeout_seconds=args.timeout_seconds,
        allow_test_backend=args.allow_test_backend,
    )
    token = args.api_key_file.read_text(encoding="utf-8").strip()
    server = build_server(
        legacy, manager, host=args.host, port=args.port, token=token
    )
    print(
        json.dumps(
            {
                "event": "service_ready",
                "protocol": "mu-v14-stateful-logical-v2",
                "model": manager.native_config.get("model"),
                "pid": os.getpid(),
                "port": args.port,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        manager.shutdown_all()
        server.server_close()


if __name__ == "__main__":
    main()
