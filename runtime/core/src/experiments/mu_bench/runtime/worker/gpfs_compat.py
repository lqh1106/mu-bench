"""Worker-side rollback bridge for whole-session GPFS RPC requests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time
import uuid

from experiments.mu_bench.runtime.contracts import SessionSubmission
from experiments.mu_bench.runtime.worker.service import WorkerRuntime


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


class GPFSCompatWorkerServer:
    """Serve SessionSpec pointer requests without placing media frames on GPFS."""

    def __init__(
        self,
        *,
        runtime: WorkerRuntime,
        queue_dir: Path,
        poll_seconds: float = 0.05,
    ) -> None:
        self.runtime = runtime
        self.queue_dir = queue_dir.resolve()
        self.poll_seconds = poll_seconds
        self.requests = self.queue_dir / "requests"
        self.processing = self.queue_dir / "processing"
        self.responses = self.queue_dir / "responses"
        for path in (self.requests, self.processing, self.responses):
            path.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def serve_forever(self) -> None:
        while not self._stop.is_set():
            found = False
            for request_path in sorted(self.requests.glob("*.json")):
                found = True
                claimed = self.processing / request_path.name
                try:
                    os.replace(request_path, claimed)
                except FileNotFoundError:
                    continue
                request_id = claimed.stem
                response_path = self.responses / f"{request_id}.json"
                try:
                    request = json.loads(claimed.read_text(encoding="utf-8"))
                    method = str(request.get("method"))
                    if method == "health":
                        payload = {
                            "ready": True,
                            "backend": self.runtime.capabilities.backend_name,
                        }
                    elif method == "run_session_spec":
                        submission = SessionSubmission.model_validate(
                            request.get("payload", {})
                        )
                        status = self.runtime.submit(submission)
                        while not status.state.terminal and not self._stop.is_set():
                            time.sleep(self.poll_seconds)
                            status = self.runtime.status(status.session_id)
                        payload = status.model_dump(mode="json")
                    else:
                        raise ValueError(f"unknown compatibility method: {method}")
                    _atomic_json(
                        response_path,
                        {"id": request_id, "ok": True, "payload": payload},
                    )
                except Exception as exc:  # noqa: BLE001
                    _atomic_json(
                        response_path,
                        {
                            "id": request_id,
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                finally:
                    claimed.unlink(missing_ok=True)
            if not found:
                time.sleep(self.poll_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
