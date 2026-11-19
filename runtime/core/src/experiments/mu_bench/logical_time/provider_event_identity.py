"""Instance-scoped preservation of OpenAI response identity at the wire edge.

The pinned tau2 parser discards ``response.created.response.id``. Adding an ID
lookup in the media adapter alone cannot recover it. This module leaves the
shared tau2 package and other providers untouched, replacing only the receiver
on the newly constructed, not-yet-connected provider owned by this run.
"""
from __future__ import annotations

import asyncio
import json
from types import MethodType
from typing import Any

import websockets
from tau2.voice.audio_native.openai.events import (
    ResponseCreatedEvent,
    UnknownEvent,
    parse_realtime_event,
)
from tau2.voice.audio_native.openai.provider import OpenAIRealtimeProvider


class ResponseCreatedWithIdentity(ResponseCreatedEvent):
    response_id: str | None = None
    # Only lifecycle identifiers/status, never the complete provider payload.
    raw: dict[str, Any]


def parse_identity_preserving_event(raw: dict[str, Any]):
    """Use the actual upstream parser for every event except created metadata."""
    parsed = parse_realtime_event(raw)
    if raw.get("type") != "response.created":
        return parsed
    response = raw.get("response")
    if not isinstance(response, dict):
        response = {}
    response_id = response.get("id")
    if not isinstance(response_id, str) or not response_id.strip():
        response_id = None
    safe_response = {"id": response_id}
    if isinstance(response.get("status"), str):
        safe_response["status"] = response["status"]
    return ResponseCreatedWithIdentity(
        type="response.created", event_id=parsed.event_id,
        response_id=response_id,
        raw={"type": "response.created", "event_id": parsed.event_id,
             "response": safe_response},
    )


async def _identity_receiver_loop(self) -> None:
    """Same queue/order/cancellation semantics as the pinned tau2 receiver."""
    assert self.ws is not None
    assert self._event_queue is not None
    while self.is_connected:
        try:
            raw_message = await self.ws.recv()
            event = parse_identity_preserving_event(json.loads(raw_message))
            await self._event_queue.put(event)
        except asyncio.CancelledError:
            raise
        except websockets.ConnectionClosed as exc:
            # The close reason can contain gateway data. Preserve the code,
            # not unbounded strings or credentials, in persisted diagnostics.
            await self._event_queue.put(UnknownEvent(type="error", raw={
                "error": f"WebSocket closed while receiving events: code={exc.code}",
                "exception_type": type(exc).__name__, "close_code": exc.code,
            }))
            break
        except Exception as exc:
            await self._event_queue.put(UnknownEvent(type="error", raw={
                "error": "provider event receive/parse failed",
                "exception_type": type(exc).__name__,
            }))


def install_response_identity_capture(provider: OpenAIRealtimeProvider) -> None:
    """Install before ``connect``; fail closed rather than replace a live loop.

    The factory should call this on ``base_adapter.provider`` before ownership
    is transferred to LogicalTimeOpenAIAdapter. Sender, endpoint, credentials,
    session configuration and cancellation policy are unchanged.
    """
    if not isinstance(provider, OpenAIRealtimeProvider):
        raise TypeError("identity capture requires an OpenAIRealtimeProvider instance")
    if getattr(provider, "_mu_response_identity_capture", False):
        return
    receiver = getattr(provider, "_receiver_task", None)
    if getattr(provider, "ws", None) is not None or (receiver is not None and not receiver.done()):
        raise RuntimeError("install response identity capture before connecting the provider")
    provider._receiver_loop = MethodType(_identity_receiver_loop, provider)
    provider._mu_response_identity_capture = True
