"""Dependency-light τ Tool compatibility for μ-Bench schema audits.

Production tau2 environments provide ``tau2.environment.tool.as_tool``.  This
fallback keeps the benchmark importable and its JSON schemas testable in a
lean handoff environment; it implements only the surface used by μ-Bench.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, create_model


class FunctionTool:
    """Small callable wrapper exposing the tau2 ``openai_schema`` contract."""

    def __init__(self, func: Callable[..., Any], predefined: dict[str, Any]) -> None:
        self._func = func
        self._predefined = predefined
        self.name = func.__name__
        signature = inspect.signature(func)
        try:
            hints = get_type_hints(func)
        except (NameError, TypeError):
            hints = {}
        fields: dict[str, tuple[Any, Any]] = {}
        for name, parameter in signature.parameters.items():
            if name in predefined:
                continue
            annotation = hints.get(name, parameter.annotation)
            if annotation is inspect.Parameter.empty:
                annotation = Any
            default = (
                ...
                if parameter.default is inspect.Parameter.empty
                else parameter.default
            )
            fields[name] = (annotation, default)
        self.params: type[BaseModel] = create_model(
            f"{self.name}_parameters", **fields
        )

    @property
    def openai_schema(self) -> dict[str, Any]:
        doc = inspect.getdoc(self._func) or self.name
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": doc,
                "parameters": self.params.model_json_schema(),
            },
        }

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._func(*args, **{**kwargs, **self._predefined})


def as_tool(func: Callable[..., Any], **predefined: Any) -> FunctionTool:
    """Wrap one function with the subset of tau2 Tool behavior μ-Bench uses."""

    return FunctionTool(func, predefined)
