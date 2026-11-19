"""Single-session logical-time adapters for μ-Bench.

The package root is intentionally lazy.  Open-weight workers can import their
own logical-time transports without having tau2 and the OpenAI adapter's
dependencies installed in the GPU environment.
"""

from __future__ import annotations

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
