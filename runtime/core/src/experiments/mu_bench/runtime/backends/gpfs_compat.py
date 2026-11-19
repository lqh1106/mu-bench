"""Compatibility transport for the previous atomic GPFS RPC queue."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid

from experiments.mu_bench.runtime.contracts import (
    SessionError,
    SessionState,
    SessionSubmission,
    WorkerJobStatus,
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


class GPFSCompatBackend:
    """Expose the legacy queue through the new SessionControlBackend contract.

    This is a rollback path only.  It exchanges whole SessionSpec pointers and
    never places 200 ms media frames on GPFS.
    """

    def __init__(self, queue_dir: Path) -> None:
        self.queue_dir = queue_dir.resolve()
        self.requests_dir = self.queue_dir / "requests"
        self.processing_dir = self.queue_dir / "processing"
        self.responses_dir = self.queue_dir / "responses"
        self.jobs_dir = self.queue_dir / "session_jobs"
        for directory in (
            self.requests_dir,
            self.processing_dir,
            self.responses_dir,
            self.jobs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _job_path(self, session_id: str) -> Path:
        return self.jobs_dir / f"{session_id}.json"

    def _read_job(self, session_id: str) -> WorkerJobStatus:
        path = self._job_path(session_id)
        if not path.exists():
            raise KeyError(f"unknown session: {session_id}")
        return WorkerJobStatus.model_validate_json(path.read_text(encoding="utf-8"))

    def _write_job(self, status: WorkerJobStatus) -> None:
        _atomic_json(self._job_path(status.session_id), status.model_dump(mode="json"))

    def submit(self, submission: SessionSubmission) -> WorkerJobStatus:
        spec = json.loads(Path(submission.spec_path).read_text(encoding="utf-8"))
        session_id = str(spec["session_id"])
        existing_path = self._job_path(session_id)
        if existing_path.exists():
            existing = self._read_job(session_id)
            if existing.idempotency_key != submission.idempotency_key:
                raise RuntimeError("session id reused with another idempotency key")
            return existing
        now = time.time()
        status = WorkerJobStatus(
            session_id=session_id,
            state=SessionState.QUEUED,
            spec_path=submission.spec_path,
            result_path=submission.result_path,
            idempotency_key=submission.idempotency_key,
            queued_at_unix=now,
        )
        self._write_job(status)
        _atomic_json(
            self.requests_dir / f"{session_id}.json",
            {
                "id": session_id,
                "method": "run_session_spec",
                "created_at": now,
                "payload": submission.model_dump(mode="json"),
            },
        )
        return status

    def status(self, session_id: str) -> WorkerJobStatus:
        status = self._read_job(session_id)
        response_path = self.responses_dir / f"{session_id}.json"
        processing_path = self.processing_dir / f"{session_id}.json"
        if response_path.exists() and not status.state.terminal:
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if response.get("ok", False):
                status.state = SessionState.SUCCEEDED
            else:
                status.state = SessionState.INFRA_ERROR
                status.error = SessionError(
                    error_type="LegacyGPFSRPCError",
                    message=str(response.get("error", "unknown legacy RPC error")),
                    traceback=response.get("traceback"),
                    retryable=True,
                )
            status.completed_at_unix = time.time()
            self._write_job(status)
        elif processing_path.exists() and status.state == SessionState.QUEUED:
            status.state = SessionState.RUNNING
            status.started_at_unix = time.time()
            self._write_job(status)
        return status

    def cancel(self, session_id: str) -> WorkerJobStatus:
        status = self._read_job(session_id)
        if status.state == SessionState.QUEUED:
            (self.requests_dir / f"{session_id}.json").unlink(missing_ok=True)
            status.state = SessionState.CANCELLED
            status.completed_at_unix = time.time()
            self._write_job(status)
        return status
