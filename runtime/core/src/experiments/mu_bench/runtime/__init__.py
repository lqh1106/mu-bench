from pkgutil import extend_path
import os as _path_os, sys as _path_sys
__path__ = sorted(extend_path(__path__, __name__), key=lambda p: min(
    [i for i,b in enumerate(_path_sys.path) if b and p.startswith(_path_os.path.abspath(b).rstrip(_path_os.sep)+_path_os.sep)]
    + [len(_path_sys.path)]))
"""Unified μ-Bench execution runtime.

The runtime separates the development-machine control plane, GPFS artifact
plane, and model-local media plane.  Backend transports may differ, but every
backend consumes :class:`SessionSpec` and produces :class:`SessionResult`.
"""

from experiments.mu_bench.runtime.contracts import (
    ArtifactManifest,
    BackendCapabilities,
    CausalInputIsolation,
    ExecutionMode,
    MediaClockSpec,
    SessionPhase,
    SessionResult,
    SessionSpec,
    SessionState,
    SessionSubmission,
    WorkerJobStatus,
)
from experiments.mu_bench.runtime.orchestrator import UnifiedOrchestrator
from experiments.mu_bench.runtime.storage import GPFSArtifactStore

__all__ = [
    "ArtifactManifest",
    "BackendCapabilities",
    "CausalInputIsolation",
    "ExecutionMode",
    "GPFSArtifactStore",
    "MediaClockSpec",
    "SessionPhase",
    "SessionResult",
    "SessionSpec",
    "SessionState",
    "SessionSubmission",
    "UnifiedOrchestrator",
    "WorkerJobStatus",
]
