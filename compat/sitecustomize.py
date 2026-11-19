"""Compatibility shim for the runner's Python 3.10 process.

The benchmark models import enum.StrEnum, which is available in Python 3.11.
The runner itself only needs the str+Enum behavior, so expose the equivalent
before application modules are imported.
"""

import enum


if not hasattr(enum, "StrEnum"):
    class StrEnum(str, enum.Enum):
        def __str__(self) -> str:
            return self.value

    enum.StrEnum = StrEnum
