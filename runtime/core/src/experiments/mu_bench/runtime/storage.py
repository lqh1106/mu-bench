"""GPFS artifact plane with atomic JSON publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
import uuid

from experiments.mu_bench.runtime.contracts import SessionResult, SessionSpec


class GPFSArtifactStore:
    """Read and atomically publish session artifacts below explicit roots."""

    def __init__(self, allowed_roots: list[Path] | tuple[Path, ...]) -> None:
        if not allowed_roots:
            raise ValueError("at least one GPFS root is required")
        self.allowed_roots = tuple(root.resolve() for root in allowed_roots)

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            raise ValueError(f"artifact path must be absolute: {candidate}")
        resolved = candidate.resolve(strict=False)
        if not any(resolved == root or root in resolved.parents for root in self.allowed_roots):
            raise ValueError(f"artifact path is outside configured GPFS roots: {resolved}")
        return resolved

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)

    def write_json(self, path: str | Path, payload: dict[str, Any]) -> Path:
        resolved = self.resolve(path)
        self._atomic_text(
            resolved,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
        return resolved

    def read_json(self, path: str | Path) -> dict[str, Any]:
        resolved = self.resolve(path)
        return json.loads(resolved.read_text(encoding="utf-8"))

    def write_spec(self, path: str | Path, spec: SessionSpec) -> Path:
        return self.write_json(path, spec.model_dump(mode="json"))

    def load_spec(self, path: str | Path) -> SessionSpec:
        return SessionSpec.model_validate(self.read_json(path))

    def write_result(self, path: str | Path, result: SessionResult) -> Path:
        return self.write_json(path, result.model_dump(mode="json"))

    def load_result(self, path: str | Path) -> SessionResult:
        return SessionResult.model_validate(self.read_json(path))

    def ensure_output_dir(self, path: str | Path) -> Path:
        resolved = self.resolve(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    def sha256(self, path: str | Path) -> str:
        resolved = self.resolve(path)
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
