"""Persistent, demand-driven native model sessions for logical-time evaluation.

No module in this package starts a GPU worker on import.
"""

from .session import NativeOutput, StatefulSession, SessionProtocolError
from .client import SessionClient

__all__ = ["NativeOutput", "StatefulSession", "SessionProtocolError", "SessionClient"]
