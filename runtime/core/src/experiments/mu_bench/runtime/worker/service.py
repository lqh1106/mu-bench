"""Asynchronous HTTP control service around one persistent GPU model."""

from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import queue
import threading
import time
import traceback
from typing import Any, Protocol
from urllib.parse import urlparse

from experiments.mu_bench.runtime.contracts import (
    BackendCapabilities,
    SessionError,
    SessionResult,
    SessionState,
    SessionSubmission,
    WorkerJobStatus,
)
from experiments.mu_bench.runtime.storage import GPFSArtifactStore


class SessionExecutor(Protocol):
    """Loaded execution backend used by one serial GPU/API worker."""

    @property
    def capabilities(self) -> BackendCapabilities:
        """Return the adapter capabilities advertised by the worker."""

    def execute(
        self,
        spec: Any,
        *,
        cancel_event: threading.Event | None = None,
    ) -> SessionResult:
        """Run one complete immutable SessionSpec."""


@dataclass
class _QueuedJob:
    submission: SessionSubmission
    status: WorkerJobStatus
    cancel_event: threading.Event


class WorkerRuntime:
    """Own one loaded adapter and execute submitted sessions serially."""

    def __init__(
        self,
        *,
        store: GPFSArtifactStore,
        executor: SessionExecutor,
    ) -> None:
        self.store = store
        self.executor = executor
        self.capabilities = executor.capabilities
        self._jobs: dict[str, _QueuedJob] = {}
        self._idempotency: dict[str, str] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue[_QueuedJob | None] = queue.Queue()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @staticmethod
    def _log(event: str, **fields: Any) -> None:
        print(
            json.dumps(
                {"component": "mu-worker-runtime", "event": event, **fields},
                ensure_ascii=False,
            ),
            flush=True,
        )

    def submit(self, submission: SessionSubmission) -> WorkerJobStatus:
        spec = self.store.load_spec(submission.spec_path)
        if submission.result_path != spec.artifacts.result_json:
            raise ValueError("submission result_path disagrees with SessionSpec")
        with self._lock:
            prior_session = self._idempotency.get(submission.idempotency_key)
            if prior_session is not None:
                prior = self._jobs[prior_session]
                if prior.status.spec_path != submission.spec_path:
                    raise ValueError("idempotency key reused for another SessionSpec")
                return prior.status.model_copy(deep=True)
            if spec.session_id in self._jobs:
                raise ValueError("session_id already exists with another idempotency key")
            status = WorkerJobStatus(
                session_id=spec.session_id,
                state=SessionState.QUEUED,
                spec_path=submission.spec_path,
                result_path=submission.result_path,
                idempotency_key=submission.idempotency_key,
                queued_at_unix=time.time(),
            )
            job = _QueuedJob(
                submission=submission,
                status=status,
                cancel_event=threading.Event(),
            )
            self._jobs[spec.session_id] = job
            self._idempotency[submission.idempotency_key] = spec.session_id
            self._queue.put(job)
            self._log(
                "session_queued",
                session_id=spec.session_id,
                case_id=spec.case_id,
                phase=spec.phase.value,
                spec_path=submission.spec_path,
                result_path=submission.result_path,
            )
            return status.model_copy(deep=True)

    def status(self, session_id: str) -> WorkerJobStatus:
        with self._lock:
            try:
                return self._jobs[session_id].status.model_copy(deep=True)
            except KeyError as exc:
                raise KeyError(f"unknown session: {session_id}") from exc

    def cancel(self, session_id: str) -> WorkerJobStatus:
        with self._lock:
            job = self._jobs.get(session_id)
            if job is None:
                raise KeyError(f"unknown session: {session_id}")
            if not job.status.state.terminal:
                job.cancel_event.set()
                if job.status.state == SessionState.QUEUED:
                    spec = self.store.load_spec(job.submission.spec_path)
                    now = time.time()
                    result = SessionResult(
                        session_id=spec.session_id,
                        attempt_id=spec.attempt_id,
                        case_id=spec.case_id,
                        phase=spec.phase,
                        state=SessionState.CANCELLED,
                        model_name=spec.model_name,
                        backend_name=spec.backend_name,
                        execution_mode=self.capabilities.execution_mode,
                        causal_input_isolation=self.capabilities.causal_input_isolation,
                        queued_at_unix=job.status.queued_at_unix,
                        completed_at_unix=now,
                    )
                    self.store.write_result(job.submission.result_path, result)
                    job.status.state = SessionState.CANCELLED
                    job.status.completed_at_unix = now
                    self._log(
                        "session_cancelled",
                        session_id=spec.session_id,
                        state=SessionState.CANCELLED.value,
                    )
            return job.status.model_copy(deep=True)

    def _run(self) -> None:
        while not self._stopped.is_set():
            job = self._queue.get()
            if job is None:
                return
            with self._lock:
                if job.status.state == SessionState.CANCELLED:
                    continue
                job.status.state = SessionState.RUNNING
                job.status.started_at_unix = time.time()
                self._log(
                    "session_started",
                    session_id=job.status.session_id,
                    started_at_unix=job.status.started_at_unix,
                )
            try:
                spec = self.store.load_spec(job.submission.spec_path)
                result = self.executor.execute(spec, cancel_event=job.cancel_event)
                result.queued_at_unix = job.status.queued_at_unix
                if result.started_at_unix is None:
                    result.started_at_unix = job.status.started_at_unix
                self.store.write_result(job.submission.result_path, result)
                with self._lock:
                    job.status.state = result.state
                    job.status.completed_at_unix = result.completed_at_unix or time.time()
                    job.status.error = result.error
                    self._log(
                        "session_completed",
                        session_id=job.status.session_id,
                        state=result.state.value,
                        completed_at_unix=job.status.completed_at_unix,
                        result_path=job.status.result_path,
                    )
            except Exception as exc:  # noqa: BLE001
                now = time.time()
                error = SessionError(
                    error_type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    retryable=True,
                )
                try:
                    spec = self.store.load_spec(job.submission.spec_path)
                    result = SessionResult(
                        session_id=spec.session_id,
                        attempt_id=spec.attempt_id,
                        case_id=spec.case_id,
                        phase=spec.phase,
                        state=SessionState.INFRA_ERROR,
                        model_name=spec.model_name,
                        backend_name=spec.backend_name,
                        execution_mode=self.capabilities.execution_mode,
                        causal_input_isolation=self.capabilities.causal_input_isolation,
                        queued_at_unix=job.status.queued_at_unix,
                        started_at_unix=job.status.started_at_unix,
                        completed_at_unix=now,
                        error=error,
                    )
                    self.store.write_result(job.submission.result_path, result)
                finally:
                    with self._lock:
                        job.status.state = SessionState.INFRA_ERROR
                        job.status.completed_at_unix = now
                        job.status.error = error
                        self._log(
                            "session_failed",
                            session_id=job.status.session_id,
                            state=SessionState.INFRA_ERROR.value,
                            error_type=error.error_type,
                            message=error.message,
                        )

    def stop(self) -> None:
        self._stopped.set()
        self._queue.put(None)
        self._thread.join(timeout=10)


class WorkerHTTPServer:
    """Small JSON HTTP server suitable for the H online inference service."""

    def __init__(
        self,
        *,
        runtime: WorkerRuntime,
        host: str = "0.0.0.0",
        port: int = 8000,
        api_key: str | None = None,
    ) -> None:
        self.runtime = runtime
        self.api_key = api_key
        handler = self._handler_type()
        self.server = ThreadingHTTPServer((host, port), handler)

    def _handler_type(self) -> type[BaseHTTPRequestHandler]:
        runtime = self.runtime
        expected_key = self.api_key

        class Handler(BaseHTTPRequestHandler):
            server_version = "mu-bench-worker/1.3"

            def log_message(self, format: str, *args: object) -> None:
                print(
                    json.dumps(
                        {
                            "component": "mu-worker-http",
                            "client": self.client_address[0],
                            "message": format % args,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            def _authorized(self) -> bool:
                if not expected_key:
                    return True
                return self.headers.get("Authorization") == f"Bearer {expected_key}"

            def _json(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length > 1024 * 1024:
                    raise ValueError("control request exceeds 1 MiB")
                return json.loads(self.rfile.read(length) or b"{}")

            def _guard(self) -> bool:
                if self._authorized():
                    return True
                self._json(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
                return False

            def do_GET(self) -> None:  # noqa: N802
                path = urlparse(self.path).path
                try:
                    if path == "/health/live":
                        self._json(HTTPStatus.OK, {"live": True})
                    elif path == "/health/ready":
                        self._json(
                            HTTPStatus.OK,
                            {
                                "ready": True,
                                "backend_name": runtime.capabilities.backend_name,
                                "model_name": runtime.capabilities.model_name,
                            },
                        )
                    elif not self._guard():
                        return
                    elif path == "/mu/v1/capabilities":
                        self._json(
                            HTTPStatus.OK,
                            runtime.capabilities.model_dump(mode="json"),
                        )
                    elif path.startswith("/mu/v1/sessions/"):
                        session_id = path.removeprefix("/mu/v1/sessions/")
                        self._json(
                            HTTPStatus.OK,
                            runtime.status(session_id).model_dump(mode="json"),
                        )
                    else:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                except KeyError as exc:
                    self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": f"{type(exc).__name__}: {exc}"},
                    )

            def do_POST(self) -> None:  # noqa: N802
                if not self._guard():
                    return
                path = urlparse(self.path).path
                try:
                    if path == "/mu/v1/sessions":
                        submission = SessionSubmission.model_validate(self._body())
                        status = runtime.submit(submission)
                        self._json(
                            HTTPStatus.ACCEPTED,
                            status.model_dump(mode="json"),
                        )
                    elif path.endswith("/cancel") and path.startswith(
                        "/mu/v1/sessions/"
                    ):
                        session_id = path.removeprefix("/mu/v1/sessions/").removesuffix(
                            "/cancel"
                        )
                        self._body()
                        self._json(
                            HTTPStatus.OK,
                            runtime.cancel(session_id).model_dump(mode="json"),
                        )
                    else:
                        self._json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                except KeyError as exc:
                    self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                except ValueError as exc:
                    self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
                except Exception as exc:  # noqa: BLE001
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": f"{type(exc).__name__}: {exc}"},
                    )

        return Handler

    def serve_forever(self) -> None:
        self.server.serve_forever(poll_interval=0.25)

    def shutdown(self) -> None:
        self.server.shutdown()
        self.runtime.stop()
