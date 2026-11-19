"""Single-session logical-time adapters for μ-Bench.

The package root is intentionally lazy.  Open-weight workers can import their
own logical-time transports without having tau2 and the OpenAI adapter's
dependencies installed in the GPU environment.
"""

from __future__ import annotations
from pkgutil import extend_path
import os as _path_os, sys as _path_sys
__path__ = sorted(extend_path(__path__, __name__), key=lambda p: min(
    [i for i,b in enumerate(_path_sys.path) if b and p.startswith(_path_os.path.abspath(b).rstrip(_path_os.sep)+_path_os.sep)]
    + [len(_path_sys.path)]))


from typing import Any


__all__ = ["LogicalTimeOpenAIAdapter", "make_logical_adapter"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .openai_adapter import LogicalTimeOpenAIAdapter, make_logical_adapter

        exports = {
            "LogicalTimeOpenAIAdapter": LogicalTimeOpenAIAdapter,
            "make_logical_adapter": make_logical_adapter,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
