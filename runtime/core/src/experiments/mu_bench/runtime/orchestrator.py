"""Development-machine orchestration over interchangeable session transports."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol
import time

from experiments.mu_bench.runtime.contracts import (
    SessionResult,
    SessionSpec,
    SessionState,
    SessionSubmission,
    WorkerJobStatus,
)
from experiments.mu_bench.runtime.storage import GPFSArtifactStore


class SessionControlBackend(Protocol):
    """Small control-plane adapter; audio frames never cross this interface."""

    def submit(self, submission: SessionSubmission) -> WorkerJobStatus:
        """Submit one immutable GPFS SessionSpec."""

    def status(self, session_id: str) -> WorkerJobStatus:
        """Return the latest infrastructure state."""

    def cancel(self, session_id: str) -> WorkerJobStatus:
        """Request cancellation without deleting artifacts."""


class UnifiedOrchestrator:
    """Write SessionSpec, submit it, and admit only persisted valid results."""

    def __init__(
        self,
        *,
        store: GPFSArtifactStore,
        backend: SessionControlBackend,
        poll_seconds: float = 0.25,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self.store = store
        self.backend = backend
        self.poll_seconds = poll_seconds

    def submit(self, spec: SessionSpec, spec_path: str | Path) -> WorkerJobStatus:
        spec_file = self.store.write_spec(spec_path, spec)
        submission = SessionSubmission(
            spec_path=str(spec_file),
            result_path=spec.artifacts.result_json,
            idempotency_key=f"{spec.session_id}:attempt:{spec.attempt_id}",
        )
        return self.backend.submit(submission)

    def wait(
        self,
        session_id: str,
        *,
        timeout_seconds: float,
    ) -> SessionResult:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        deadline = time.monotonic() + timeout_seconds
        last: WorkerJobStatus | None = None
        while time.monotonic() < deadline:
            last = self.backend.status(session_id)
            if last.state.terminal:
                if not Path(last.result_path).exists():
                    raise RuntimeError(
                        f"terminal job {session_id} has no persisted result: "
                        f"{last.result_path}"
                    )
                result = self.store.load_result(last.result_path)
                if result.session_id != session_id:
                    raise RuntimeError("persisted result belongs to another session")
                if result.state != last.state:
                    raise RuntimeError("HTTP job state disagrees with GPFS result")
                return result
            time.sleep(self.poll_seconds)
        state = last.state if last is not None else SessionState.QUEUED
        raise TimeoutError(f"session {session_id} timed out in state {state}")

    def run(
        self,
        spec: SessionSpec,
        spec_path: str | Path,
        *,
        timeout_seconds: float,
    ) -> SessionResult:
        self.submit(spec, spec_path)
        return self.wait(spec.session_id, timeout_seconds=timeout_seconds)
