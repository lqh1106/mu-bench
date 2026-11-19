"""Synchronous HTTP transport for a persistent CosyVoice renderer.

The benchmark's planner and simulators deliberately consume a tiny synchronous
TTS interface.  This module preserves that interface while moving heavyweight
CosyVoice inference to a trusted CPU service.  Only the minimum speaker profile
needed for deterministic voice binding crosses HTTP; goals, permissions and
future event plans never do.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TTSHTTPError(RuntimeError):
    """HTTP failure with the server's structured diagnostic preserved."""

    def __init__(self, status: int, body: str, payload: Any | None = None):
        self.status = int(status)
        self.body = body[:65536]
        self.payload = payload if isinstance(payload, dict) else None
        message = self.payload.get("error", self.body) if self.payload else self.body
        super().__init__(f"TTS worker HTTP {self.status}: {message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "body": self.body,
            "payload": self.payload,
        }


class TTSServiceEndpoint(BaseModel):
    """GPFS-persisted discovery record for one CosyVoice service replica."""

    # annotate tts.json with bookkeeping keys (pgid, note, retargeted_*).
    # Unknown keys are ignored (dropped), never stored; all declared core
    # fields and validators below remain strict.
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal["mu-bench-tts-endpoint-v1"] = (
        "mu-bench-tts-endpoint-v1"
    )
    transport: Literal["http_pcm_s16le"] = "http_pcm_s16le"
    ready: bool
    host: str = Field(min_length=1)
    port: int = Field(gt=0, le=65_535)
    base_url: str = Field(min_length=1)
    hostname: str = Field(min_length=1)
    pid: int = Field(gt=0)
    api_key_file: str | None = None
    # Service discovery records may advertise the readiness probe path.
    # It is metadata for the runner's health check, not part of the synthesis
    # request, but older runtime schemas rejected it as an extra field.
    health_path: str = "/health/ready"
    started_at_unix: float
    updated_at_unix: float
    renderer_version: str = Field(min_length=1)
    validation_version: str = Field(min_length=1)
    sample_rate: int = Field(gt=0)
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    binding_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_dir: str
    replica_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_endpoint(self) -> "TTSServiceEndpoint":
        expected = f"http://{self.host}:{self.port}"
        if self.base_url.rstrip("/") != expected:
            raise ValueError("TTS base_url disagrees with host and port")
        if self.api_key_file is not None and not Path(self.api_key_file).is_absolute():
            raise ValueError("TTS api_key_file must be an absolute shared path")
        if not Path(self.cache_dir).is_absolute():
            raise ValueError("TTS cache_dir must be absolute")
        return self


class TTSProfile(BaseModel):
    """Only the user fields consumed by ``CosyVoiceTTS.register_profiles``."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    voice_id: str = Field(min_length=1)
    accent_label: str | None = None
    speaking_style: str | None = None


class TTSCaseRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["mu-bench-tts-register-v2"] = "mu-bench-tts-register-v2"
    run_namespace: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")
    case_id: str = Field(min_length=1)
    assignment_seed: int
    profiles: list[TTSProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_profiles(self) -> "TTSCaseRegistration":
        user_ids = [item.user_id for item in self.profiles]
        voice_ids = [item.voice_id for item in self.profiles]
        if len(user_ids) != len(set(user_ids)):
            raise ValueError("TTS registration user IDs must be unique")
        if len(voice_ids) != len(set(voice_ids)):
            raise ValueError("TTS registration voice IDs must be unique")
        return self

    def profile_digest(self) -> str:
        payload = self.model_dump(mode="json")
        payload["profiles"] = sorted(payload["profiles"], key=lambda item: item["user_id"])
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()


class TTSRegistrationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["mu-bench-tts-registration-result-v2"] = (
        "mu-bench-tts-registration-result-v2"
    )
    run_namespace: str
    case_id: str
    registration_id: str = Field(min_length=1)
    profile_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_rate: int = Field(gt=0)
    assigned_voice_catalog_ids: dict[str, str]
    provenance: dict[str, Any]


class TTSSynthesisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["mu-bench-tts-synthesis-v2"] = (
        "mu-bench-tts-synthesis-v2"
    )
    run_namespace: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")
    case_id: str = Field(min_length=1)
    registration_id: str = Field(min_length=1)
    voice_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=32_768)


class TTSCaseRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["mu-bench-tts-release-v1"] = "mu-bench-tts-release-v1"
    run_namespace: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")
    case_id: str = Field(min_length=1)
    registration_id: str | None = None


class RemoteCosyVoiceTTS:
    """Fail-closed synchronous client implementing the benchmark TTS duck type."""

    sample_rate = 16_000
    _max_pcm_bytes = 64 * 1024 * 1024

    def __init__(
        self,
        *,
        endpoint_file: Path,
        run_namespace: str,
        case_id: str,
        assignment_seed: int,
        timeout_seconds: float = 600.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("TTS HTTP timeout must be positive")
        self.endpoint_file = endpoint_file.expanduser().resolve()
        self.endpoint = TTSServiceEndpoint.model_validate_json(
            self.endpoint_file.read_text(encoding="utf-8")
        )
        if not self.endpoint.ready:
            raise RuntimeError(f"TTS endpoint is not ready: {self.endpoint_file}")
        self.case_id = case_id
        if not run_namespace or not all(c.isalnum() or c in "_.:-" for c in run_namespace):
            raise ValueError("invalid TTS run namespace")
        self.run_namespace = run_namespace
        self.assignment_seed = assignment_seed
        self.timeout_seconds = timeout_seconds
        self._queue_retry_count = 0
        self.base_url = self.endpoint.base_url.rstrip("/")
        self._opener = build_opener(ProxyHandler({}))
        self._api_key = self._load_api_key()
        self._registration: TTSRegistrationResult | None = None
        self._registered_voice_ids: set[str] = set()
        health = self._request_json("GET", "/health/ready")
        if health.get("ready") is not True:
            raise RuntimeError(f"TTS service is not ready: {health}")
        capabilities = self._request_json("GET", "/mu/v1/tts/capabilities")
        self._validate_capabilities(capabilities)

    def _load_api_key(self) -> str | None:
        if self.endpoint.api_key_file is None:
            return None
        path = Path(self.endpoint.api_key_file)
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            raise RuntimeError(f"TTS API key file is empty: {path}")
        return value

    def _headers(self, *, json_body: bool = False) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _open(self, request: Request) -> Any:
        deadline = time.monotonic() + self.timeout_seconds
        for attempt in range(3):
            try:
                remaining = max(0.1, deadline - time.monotonic())
                return self._opener.open(request, timeout=remaining)
            except HTTPError as exc:
                raw = exc.read(64 * 1024)
                detail = raw.decode("utf-8", errors="replace")
                try:
                    payload = json.loads(detail) if detail else None
                except json.JSONDecodeError:
                    payload = None
                queue_rejected = (
                    exc.code == 503
                    and isinstance(payload, dict)
                    and payload.get("error") == "shared_service_queue_timeout"
                )
                delay = 5 * (attempt + 1)
                if queue_rejected and attempt < 2 and deadline - time.monotonic() > delay:
                    self._queue_retry_count += 1
                    logging.getLogger(__name__).warning(
                        "TTS shared queue retry case=%s attempt=%s request_sha256=%s",
                        self.case_id, attempt + 1,
                        hashlib.sha256(request.data or b"").hexdigest(),
                    )
                    time.sleep(delay)
                    continue
                raise TTSHTTPError(exc.code, detail, payload) from exc
            except (URLError, TimeoutError, OSError) as exc:
                raise RuntimeError(f"TTS worker unavailable: {exc}") from exc
        raise RuntimeError("TTS queue retry budget exhausted")

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = None
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=body,
            headers=self._headers(json_body=payload is not None),
            method=method,
        )
        with self._open(request) as response:
            response_body = response.read(4 * 1024 * 1024 + 1)
        if len(response_body) > 4 * 1024 * 1024:
            raise RuntimeError("TTS JSON response exceeds 4 MiB")
        try:
            parsed = json.loads(response_body.decode("utf-8")) if response_body else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("TTS worker returned invalid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("TTS worker JSON response is not an object")
        return parsed

    def _validate_capabilities(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != "mu-bench-tts-capabilities-v1":
            raise RuntimeError("TTS worker returned an incompatible capability schema")
        for field in ("renderer_version", "validation_version", "catalog_sha256", "binding_sha256"):
            if payload.get(field) != getattr(self.endpoint, field):
                raise RuntimeError(f"TTS endpoint/capabilities mismatch for {field}")
        if int(payload.get("sample_rate", 0)) != self.sample_rate:
            raise RuntimeError(
                f"TTS worker sample rate is {payload.get('sample_rate')}, expected {self.sample_rate}"
            )
        if payload.get("run_namespace_supported") is not True:
            raise RuntimeError("TTS worker does not support run namespaces")

    @staticmethod
    def _profile_payload(user: Any) -> TTSProfile:
        voice = getattr(user, "voice", None)
        return TTSProfile(
            user_id=str(getattr(user, "user_id")),
            voice_id=str(getattr(voice, "voice_id")),
            accent_label=getattr(voice, "accent_label", None),
            speaking_style=getattr(voice, "speaking_style", None),
        )

    def register_profiles(self, users: list[Any]) -> None:
        request = TTSCaseRegistration(
            run_namespace=self.run_namespace,
            case_id=self.case_id,
            assignment_seed=self.assignment_seed,
            profiles=[self._profile_payload(user) for user in users],
        )
        payload = self._request_json(
            "POST", "/mu/v1/tts/cases/register", request.model_dump(mode="json")
        )
        result = TTSRegistrationResult.model_validate(payload)
        if result.run_namespace != self.run_namespace or result.case_id != self.case_id:
            raise RuntimeError("TTS registration returned a different case_id")
        if result.profile_digest != request.profile_digest():
            raise RuntimeError("TTS registration profile digest mismatch")
        if result.sample_rate != self.sample_rate:
            raise RuntimeError("TTS registration sample rate mismatch")
        voice_ids = {profile.voice_id for profile in request.profiles}
        if set(result.assigned_voice_catalog_ids) != voice_ids:
            raise RuntimeError("TTS registration voice assignment is incomplete")
        self._registration = result
        self._registered_voice_ids = voice_ids

    def _validate_pcm(self, pcm: bytes, headers: dict[str, str]) -> bytes:
        normalized = {key.casefold(): value for key, value in headers.items()}
        content_type = normalized.get("content-type", "").split(";", 1)[0].strip()
        if content_type != "application/octet-stream":
            raise RuntimeError(f"TTS worker returned unexpected content type: {content_type}")
        try:
            sample_rate = int(normalized["x-sample-rate"])
        except (KeyError, ValueError) as exc:
            raise RuntimeError("TTS worker omitted a valid X-Sample-Rate") from exc
        if sample_rate != self.sample_rate:
            raise RuntimeError(
                f"TTS PCM sample rate is {sample_rate}, expected {self.sample_rate}"
            )
        expected_sha256 = normalized.get("x-pcm-sha256", "").casefold()
        actual_sha256 = hashlib.sha256(pcm).hexdigest()
        if not hmac.compare_digest(expected_sha256, actual_sha256):
            raise RuntimeError("TTS PCM SHA256 validation failed")
        if not pcm or len(pcm) % 2:
            raise RuntimeError("TTS worker returned empty or non-PCM16-aligned audio")
        return pcm

    def synthesize(self, text: str, voice_id: str) -> bytes:
        if self._registration is None:
            raise RuntimeError("TTS profiles must be registered before synthesis")
        if voice_id not in self._registered_voice_ids:
            raise ValueError(f"voice profile was not registered: {voice_id}")
        payload = TTSSynthesisRequest(
            run_namespace=self.run_namespace,
            case_id=self.case_id,
            registration_id=self._registration.registration_id,
            voice_id=voice_id,
            text=text,
        ).model_dump(mode="json")
        request = Request(
            f"{self.base_url}/mu/v1/tts/synthesize",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={**self._headers(json_body=True), "Accept": "application/octet-stream"},
            method="POST",
        )
        with self._open(request) as response:
            pcm = response.read(self._max_pcm_bytes + 1)
            headers = dict(response.headers.items())
        if len(pcm) > self._max_pcm_bytes:
            raise RuntimeError("TTS PCM response exceeds 64 MiB")
        return self._validate_pcm(pcm, headers)

    def release_case(self) -> dict[str, Any]:
        if self._registration is None:
            return {"released": False, "reason": "not_registered"}
        payload = TTSCaseRelease(
            run_namespace=self.run_namespace,
            case_id=self.case_id,
            registration_id=self._registration.registration_id,
        ).model_dump(mode="json")
        try:
            return self._request_json("POST", "/mu/v1/tts/cases/release", payload)
        finally:
            self._registration = None
            self._registered_voice_ids = set()

    def close(self) -> dict[str, Any]:
        return self.release_case()

    def provenance(self) -> dict[str, Any]:
        if self._registration is None:
            raise RuntimeError("TTS profiles must be registered before provenance")
        return {
            **self._registration.provenance,
            "backend": "cosyvoice_http",
            "transport": "synchronous_raw_pcm_s16le_http",
            "endpoint_file": str(self.endpoint_file),
            "endpoint_base_url": self.base_url,
            "service_replica_id": self.endpoint.replica_id,
            "profile_digest": self._registration.profile_digest,
            "assigned_voice_catalog_ids": dict(
                self._registration.assigned_voice_catalog_ids
            ),
            "api_key_present": self._api_key is not None,
            "environment_proxy_bypassed": True,
            "shared_queue_retries": self._queue_retry_count,
        }
