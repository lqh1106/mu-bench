"""Development-machine control backend for Realtime API session executors."""

from __future__ import annotations

from experiments.mu_bench.runtime.contracts import (
    SessionSubmission,
    WorkerJobStatus,
)
from experiments.mu_bench.runtime.worker.service import WorkerRuntime


class InProcessBackend:
    """Use the same asynchronous session API without an HTTP network hop."""

    def __init__(self, runtime: WorkerRuntime) -> None:
        self.runtime = runtime

    def submit(self, submission: SessionSubmission) -> WorkerJobStatus:
        return self.runtime.submit(submission)

    def status(self, session_id: str) -> WorkerJobStatus:
        return self.runtime.status(session_id)

    def cancel(self, session_id: str) -> WorkerJobStatus:
        return self.runtime.cancel(session_id)
