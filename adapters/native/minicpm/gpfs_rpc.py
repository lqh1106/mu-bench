#!/usr/bin/env python3
"""Atomic request/response RPC over a shared GPFS directory."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any
import uuid


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class GPFSRPCClient:
    def __init__(self, queue_dir: Path, timeout_seconds: float = 1800.0) -> None:
        self.queue_dir = queue_dir.resolve()
        self.requests_dir = self.queue_dir / "requests"
        self.responses_dir = self.queue_dir / "responses"
        self.timeout_seconds = timeout_seconds
        self.requests_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)

    def call(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        request_path = self.requests_dir / f"{request_id}.json"
        response_path = self.responses_dir / f"{request_id}.json"
        atomic_json(
            request_path,
            {
                "id": request_id,
                "method": method,
                "created_at": time.time(),
                "payload": payload,
            },
        )
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            if response_path.exists():
                response = load_json(response_path)
                response_path.unlink(missing_ok=True)
                if not response.get("ok", False):
                    error = response.get("error", "unknown RPC error")
                    traceback_text = response.get("traceback", "")
                    raise RuntimeError(f"{error}\n{traceback_text}".rstrip())
                return dict(response.get("payload", {}))
            time.sleep(0.05)
        raise TimeoutError(
            f"GPU RPC timed out after {self.timeout_seconds:.1f}s for "
            f"{method} request {request_id}"
        )
