"""HTTP control-plane client for an online μ-Bench GPU Worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from experiments.mu_bench.runtime.contracts import (
    BackendCapabilities,
    RJobWorkerEndpoint,
    SessionSubmission,
    WorkerJobStatus,
)


class HTTPWorkerBackend:
    """Submit SessionSpec pointers; never upload audio through HTTP."""

    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        use_environment_proxy: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._opener = build_opener(
            ProxyHandler() if use_environment_proxy else ProxyHandler({})
        )

    @classmethod
    def from_rjob_endpoint(
        cls,
        endpoint_path: str | Path,
        *,
        timeout_seconds: float = 30.0,
    ) -> HTTPWorkerBackend:
        """Resolve one live RJob IP:port record persisted on GPFS."""
        path = Path(endpoint_path).resolve()
        endpoint = RJobWorkerEndpoint.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        if not endpoint.ready:
            raise RuntimeError(f"RJob endpoint is not ready: {path}")
        api_key = None
        if endpoint.api_key_file is not None:
            api_key = Path(endpoint.api_key_file).read_text(encoding="utf-8").strip()
            if not api_key:
                raise RuntimeError("RJob API key file is empty")
        return cls(
            endpoint.base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            use_environment_proxy=False,
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"worker HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"worker HTTP unavailable: {exc.reason}") from exc
        return json.loads(body.decode("utf-8")) if body else {}

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health/ready")

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities.model_validate(
            self._request("GET", "/mu/v1/capabilities")
        )

    def submit(self, submission: SessionSubmission) -> WorkerJobStatus:
        payload = self._request(
            "POST",
            "/mu/v1/sessions",
            submission.model_dump(mode="json"),
        )
        return WorkerJobStatus.model_validate(payload)

    def status(self, session_id: str) -> WorkerJobStatus:
        return WorkerJobStatus.model_validate(
            self._request("GET", f"/mu/v1/sessions/{session_id}")
        )

    def cancel(self, session_id: str) -> WorkerJobStatus:
        return WorkerJobStatus.model_validate(
            self._request("POST", f"/mu/v1/sessions/{session_id}/cancel", {})
        )
