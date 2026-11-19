from pkgutil import extend_path
import os as _path_os, sys as _path_sys
__path__ = sorted(extend_path(__path__, __name__), key=lambda p: min(
    [i for i,b in enumerate(_path_sys.path) if b and p.startswith(_path_os.path.abspath(b).rstrip(_path_os.sep)+_path_os.sep)]
    + [len(_path_sys.path)]))
"""μ-Bench multi-user full-duplex benchmark (legacy import namespace)."""

from experiments.tau_multiuser_voice.authorization import AuthorizationOracle
from experiments.tau_multiuser_voice.cases import CaseCatalog
from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator
from experiments.tau_multiuser_voice.models import CapabilityCategory
from experiments.tau_multiuser_voice.simulator import MultiUserVoiceSimulator
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.version import (
    BENCHMARK_DISPLAY_NAME,
    BENCHMARK_SLUG,
    BENCHMARK_VERSION,
    CLOSED_SOURCE_SYSTEM_PROMPT_POLICY,
    OPEN_WEIGHT_SYSTEM_PROMPT_POLICY,
    RELEASE_DATE,
    RELEASE_STATUS,
    RUNTIME_TRANSPORT_VERSION,
    SCORING_CONTRACT_VERSION,
    WEAK_COMPREHENSION_SCHEDULE_VERSION,
    VOICE_PIPELINE_VERSION,
)

__all__ = [
    "AuthorizationOracle",
    "BENCHMARK_DISPLAY_NAME",
    "BENCHMARK_SLUG",
    "BENCHMARK_VERSION",
    "CLOSED_SOURCE_SYSTEM_PROMPT_POLICY",
    "OPEN_WEIGHT_SYSTEM_PROMPT_POLICY",
    "RELEASE_DATE",
    "RELEASE_STATUS",
    "RUNTIME_TRANSPORT_VERSION",
    "SCORING_CONTRACT_VERSION",
    "WEAK_COMPREHENSION_SCHEDULE_VERSION",
    "VOICE_PIPELINE_VERSION",
    "CapabilityCategory",
    "CaseCatalog",
    "MultiUserEvaluator",
    "MultiUserVoiceSimulator",
    "create_virtual_environment",
]
