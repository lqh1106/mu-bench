from pkgutil import extend_path
import os as _path_os, sys as _path_sys
__path__ = sorted(extend_path(__path__, __name__), key=lambda p: min(
    [i for i,b in enumerate(_path_sys.path) if b and p.startswith(_path_os.path.abspath(b).rstrip(_path_os.sep)+_path_os.sep)]
    + [len(_path_sys.path)]))
"""Control-plane clients for μ-Bench session execution."""

from experiments.mu_bench.runtime.backends.gpfs_compat import GPFSCompatBackend
from experiments.mu_bench.runtime.backends.http_worker import HTTPWorkerBackend
from experiments.mu_bench.runtime.backends.inprocess import InProcessBackend
from experiments.mu_bench.runtime.backends.openai_realtime import OpenAIRealtimeSessionExecutor

__all__ = [
    "GPFSCompatBackend",
    "HTTPWorkerBackend",
    "InProcessBackend",
    "OpenAIRealtimeSessionExecutor",
]
