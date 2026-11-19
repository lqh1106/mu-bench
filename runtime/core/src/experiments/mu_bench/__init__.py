from pkgutil import extend_path
import os as _path_os, sys as _path_sys
__path__ = sorted(extend_path(__path__, __name__), key=lambda p: min(
    [i for i,b in enumerate(_path_sys.path) if b and p.startswith(_path_os.path.abspath(b).rstrip(_path_os.sep)+_path_os.sep)]
    + [len(_path_sys.path)]))
"""Preferred public namespace for μ-Bench.

The implementation remains in ``experiments.tau_multiuser_voice`` during the
v1.3 compatibility window so existing 800-case IDs, artifacts, and imports do
not break.
"""

from experiments.tau_multiuser_voice import (
    AuthorizationOracle,
    BENCHMARK_DISPLAY_NAME,
    BENCHMARK_SLUG,
    BENCHMARK_VERSION,
    CapabilityCategory,
    CaseCatalog,
    MultiUserEvaluator,
    MultiUserVoiceSimulator,
    create_virtual_environment,
)
from experiments.tau_multiuser_voice.version import (
    RELEASE_DATE,
    RELEASE_STATUS,
    RUNTIME_TRANSPORT_VERSION,
    VOICE_PIPELINE_VERSION,
)

__all__ = [
    "AuthorizationOracle",
    "BENCHMARK_DISPLAY_NAME",
    "BENCHMARK_SLUG",
    "BENCHMARK_VERSION",
    "RELEASE_DATE",
    "RELEASE_STATUS",
    "RUNTIME_TRANSPORT_VERSION",
    "VOICE_PIPELINE_VERSION",
    "CapabilityCategory",
    "CaseCatalog",
    "MultiUserEvaluator",
    "MultiUserVoiceSimulator",
    "create_virtual_environment",
]
