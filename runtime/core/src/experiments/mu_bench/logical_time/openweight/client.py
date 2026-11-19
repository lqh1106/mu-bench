"""Development-host control client. The GPU never calls the MLLM.

step_pcm accepts arbitrary integral-ms chunks and retains sub-quantum input in
the remote session. Chunk files contain only newly revealed user samples.
There is deliberately no automatic replay/reconnect/new-session fallback.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid


class WorkerHTTPError(RuntimeError):
    """HTTP failure retaining the worker's structured response body."""

    def __init__(self, status: int, body: str, payload=None):
        self.status = int(status)
        self.body = body[:65536]
        self.payload = payload if isinstance(payload, dict) else None
        message = self.payload.get("error", self.body) if self.payload else self.body
        super().__init__(f"worker HTTP {self.status}: {message}")

    def to_dict(self):
        return {"status": self.status, "body": self.body, "payload": self.payload}


class SessionClient:
    def __init__(self, base_url: str, token: str, *, shared_chunk_dir: str | None = None,
                 timeout_seconds: float = 1800, admission_wait_seconds: float = 3600):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout_seconds
        self.session_id = None
        self.sequence = 0
        self.pending_request = None
        self.pending_start = None
        self.admission_wait_seconds = admission_wait_seconds
        self.admission_wait_events = []
        self.shared_chunk_dir = Path(shared_chunk_dir) if shared_chunk_dir else None
        if self.shared_chunk_dir:
            self.shared_chunk_dir.mkdir(parents=True, exist_ok=False)
        # Internal pod HTTP must not be redirected through an external proxy.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _post(self, path, payload):
        request = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                value = json.load(response)
        except urllib.error.HTTPError as exc:
            raw = exc.read(256 * 1024)
            body = raw.decode("utf-8", errors="replace")
            try:
                payload = json.loads(body) if body else None
            except json.JSONDecodeError:
                payload = None
            raise WorkerHTTPError(exc.code, body, payload) from exc
        if not value.get("ok"):
            raise RuntimeError(value)
        return value["result"]

    def start(self, *, seed: int, max_media_ms: int, request_id: str | None = None):
        if self.session_id is not None:
            raise RuntimeError("a client owns exactly one conversation")
        if self.pending_start is not None:
            raise RuntimeError("prior session start unresolved; never create another context automatically")
        payload = dict(seed=seed, max_media_ms=max_media_ms, request_id=request_id or uuid.uuid4().hex)
        self.pending_start = payload
        started = time.monotonic()
        while True:
            try:
                result = self._post("/sessions", payload)
                break
            except WorkerHTTPError as exc:
                # Only the new admission service can prove no actor/request
                # exists. Never retry a general HTTP conflict, timeout or EOF.
                if exc.status != 409:
                    raise
                body = exc.payload
                if not isinstance(body, dict):
                    raise exc
                safe = (body.get("ok") is False and body.get("error_type") == "AdmissionWaitError"
                        and body.get("session_created") is False and body.get("request_reserved") is False)
                if not safe:
                    raise exc
                elapsed = time.monotonic() - started
                record = dict(request_id=payload["request_id"], waiting_seconds=elapsed,
                              session_created=False, reason=body.get("error", "GPU admission wait"))
                self.admission_wait_events.append(record)
                if self.shared_chunk_dir:
                    with (self.shared_chunk_dir / "admission_wait.jsonl").open("a") as out:
                        out.write(json.dumps(record) + "\n")
                if elapsed >= self.admission_wait_seconds:
                    # This particular rejection explicitly proves no session
                    # was created. Caller may report capacity, never replay.
                    self.pending_start = None
                    raise RuntimeError("native admission capacity wait exhausted before session creation") from exc
                delay = body.get("retry_after_seconds", 5)
                if not isinstance(delay, (int, float)):
                    delay = 5
                time.sleep(min(max(1, delay), 30, self.admission_wait_seconds - elapsed))
        self.session_id = result["session_id"]
        self.pending_start = None
        return result

    def step_pcm(self, pcm16: bytes, sim_start_ms: int, *, request_id: str | None = None):
        if self.session_id is None:
            raise RuntimeError("start required")
        if self.pending_request is not None:
            raise RuntimeError("prior advance response unresolved; use retry_pending, not a new request")
        rid = request_id or uuid.uuid4().hex
        payload = dict(sim_start_ms=sim_start_ms, sequence=self.sequence, request_id=rid)
        if self.shared_chunk_dir:
            path = self.shared_chunk_dir / f"{self.sequence:08d}_{rid}.pcm"
            with path.open("xb") as stream:
                stream.write(pcm16)
            payload.update(pcm_path=str(path), pcm_sha256=hashlib.sha256(pcm16).hexdigest())
        else:
            payload["pcm16_b64"] = base64.b64encode(pcm16).decode()
        self.pending_request = payload
        return self.retry_pending()

    def retry_pending(self):
        """Explicit transport retry of EXACT same operation, not model replay."""
        if self.pending_request is None:
            raise RuntimeError("no unresolved advance")
        result = self._post(f"/sessions/{self.session_id}/advance", self.pending_request)
        self.sequence = result["sequence"]
        self.pending_request = None
        result["assistant_pcm16"] = base64.b64decode(result.pop("assistant_pcm16_b64"))
        return result

    def _control(self, operation):
        if self.session_id is None:
            raise RuntimeError("start required")
        return self._post(f"/sessions/{self.session_id}/{operation}", {})

    def pause(self): return self._control("pause")
    def resume(self): return self._control("resume")
    def finish(self): return self._control("finish")
    def abort(self): return self._control("abort")
    def status(self): return self._control("status")
