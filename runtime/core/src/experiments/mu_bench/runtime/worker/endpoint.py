"""Publish directly addressable RJob worker endpoints through GPFS."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import time
import uuid

from experiments.mu_bench.runtime.contracts import (
    BackendCapabilities,
    RJobWorkerEndpoint,
)


def discover_ipv4(advertise_host: str | None = None) -> str:
    """Return an explicit host or the primary non-loopback RJob IPv4 address."""
    if advertise_host:
        return advertise_host
    for variable in ("POD_IP", "RJOB_POD_IP"):
        value = os.environ.get(variable)
        if value:
            return value
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as handle:
            handle.connect(("10.255.255.255", 1))
            value = str(handle.getsockname()[0])
            if value and not value.startswith("127."):
                return value
    except OSError:
        pass
    addresses = socket.gethostbyname_ex(socket.gethostname())[2]
    for value in addresses:
        if value and not value.startswith("127."):
            return value
    raise RuntimeError("could not discover a non-loopback RJob IPv4 address")


def publish_endpoint(
    path: Path,
    *,
    capabilities: BackendCapabilities,
    port: int,
    ready: bool,
    advertise_host: str | None = None,
    api_key_file: Path | None = None,
    started_at_unix: float,
) -> RJobWorkerEndpoint:
    """Atomically publish one endpoint record visible to the development host."""
    host = discover_ipv4(advertise_host)
    now = time.time()
    endpoint = RJobWorkerEndpoint(
        ready=ready,
        host=host,
        port=port,
        base_url=f"http://{host}:{port}",
        hostname=socket.gethostname(),
        pid=os.getpid(),
        backend_name=capabilities.backend_name,
        model_name=capabilities.model_name,
        api_key_file=str(api_key_file.resolve()) if api_key_file else None,
        started_at_unix=started_at_unix,
        updated_at_unix=now,
        rjob_name=os.environ.get("RJOB_NAME") or os.environ.get("JOB_NAME"),
        pod_name=os.environ.get("POD_NAME") or os.environ.get("HOSTNAME"),
    )
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(endpoint.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return endpoint


def load_api_key(
    *,
    api_key_env: str,
    api_key_file: Path | None,
) -> str | None:
    """Load a bearer token without placing it in the RJob command line."""
    if api_key_file is not None:
        value = api_key_file.read_text(encoding="utf-8").strip()
        if not value:
            raise RuntimeError(f"empty API key file: {api_key_file}")
        return value
    return os.environ.get(api_key_env)
