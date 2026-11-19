"""xAI wire facade for the existing single-session logical-time contract.

Reuse tau2's xAI audio/tool wire methods and μ-Bench's played-prefix ledger.
Do not use xAI's old typed parser: it drops VAD timestamps and truncate ACKs.
Registration stays server-VAD with explicit auto-response flags, never a new
manual-commit enrollment protocol. Missing/ignored flags fail closed.
"""
from __future__ import annotations

import asyncio
import inspect
from collections import deque
import json
import os
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import websockets

from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
from tau2.voice.audio_native.openai.discrete_time_adapter import DiscreteTimeOpenAIAdapter
from tau2.voice.audio_native.openai.events import (
    ResponseCreatedEvent, SessionUpdatedEvent, parse_realtime_event,
)
from tau2.voice.audio_native.xai.provider import XAIAudioFormat, XAIRealtimeProvider

from adapters.providers.openai.adapter import LogicalTimeOpenAIAdapter


class GrokProtocolError(RuntimeError):
    """Provider contract was not demonstrated; never silently change protocol."""


class GrokSessionUpdatedEvent(SessionUpdatedEvent):
    """Retain xAI's acknowledged settings instead of dropping them on parsing."""

    raw: dict


class GrokResponseCreatedEvent(ResponseCreatedEvent):
    """A response identity must survive parsing; counts cannot identify turns."""

    response_id: str
    status: str | None = None


def _response_identity(raw):
    response = raw.get("response")
    nested = response.get("id") if isinstance(response, dict) else None
    direct = raw.get("response_id")
    if nested and direct and nested != direct:
        raise GrokProtocolError("Grok lifecycle event has conflicting response IDs")
    value = nested or direct
    if not isinstance(value, str) or not value.strip():
        raise GrokProtocolError("Grok lifecycle event is missing a response ID")
    return value


def grok_lifecycle_metadata(raw, *, secrets=()):
    """Whitelist wire evidence, never conversation text, audio, or credentials.

    Unknown event names remain visible to diagnose new terminal aliases without
    treating them as terminal. Only scalar IDs, status and clock fields survive.
    """
    def safe(value):
        if isinstance(value, str):
            for secret in secrets:
                if secret:
                    value = value.replace(secret, "[REDACTED]")
            return value[:512]
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return None

    out = {}
    for key in ("type", "event_id", "response_id", "item_id", "previous_item_id",
                "audio_start_ms", "audio_end_ms", "content_index", "output_index",
                "status"):
        if key in raw:
            out[key] = safe(raw[key])
    for name, fields in (("response", ("id", "status")),
                         ("item", ("id", "type", "role", "status", "call_id")),
                         ("error", ("type", "code"))):
        obj = raw.get(name)
        if isinstance(obj, dict):
            out[name] = {key: safe(obj[key]) for key in fields if key in obj}
    response = raw.get("response")
    details = response.get("status_details") if isinstance(response, dict) else None
    if isinstance(details, dict):
        out["response"]["status_details"] = {
            key: safe(details[key]) for key in ("type", "reason") if key in details}
    elif isinstance(details, str):
        out["response"]["status_details"] = safe(details)
    session = raw.get("session")
    if isinstance(session, dict):
        out["session"] = {"id": safe(session.get("id"))}
        turn = session.get("turn_detection")
        if isinstance(turn, dict):
            out["session"]["turn_detection"] = {key: safe(turn[key]) for key in (
                "type", "create_response", "interrupt_response") if key in turn}
    return out


def grok_websocket_url(realtime_url, model):
    parsed = urlsplit(realtime_url)
    schemes = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}
    if parsed.scheme not in schemes or not parsed.netloc or parsed.username or parsed.password:
        raise GrokProtocolError("invalid Grok Realtime endpoint")
    path = parsed.path.rstrip("/")
    if not path.endswith("/realtime"):
        path += "/realtime" if path.endswith("/v1") else "/v1/realtime"
    query = [(key, value) for key, value in parse_qsl(parsed.query, keep_blank_values=True)
             if key != "model"]
    query.append(("model", model))
    return urlunsplit((schemes[parsed.scheme], parsed.netloc, path, urlencode(query), ""))


def grok_turn_detection(enabled):
    return {"type": "server_vad", "threshold": 0.5, "prefix_padding_ms": 300,
            "silence_duration_ms": 500, "create_response": enabled,
            "interrupt_response": enabled}


def verify_grok_auto_response_ack(raw, enabled):
    session = raw.get("session") if isinstance(raw, dict) else None
    turn = session.get("turn_detection") if isinstance(session, dict) else None
    if not isinstance(turn, dict) or turn.get("type") != "server_vad":
        raise GrokProtocolError("Grok did not acknowledge server_vad registration policy")
    # This single case records boolean echoes without gating on them.
    return turn


def normalize_grok_event(raw):
    """Normalize names only, preserving IDs/timestamps and cumulative ASR.

    A missing VAD timestamp cannot be replaced with zero: doing so would give
    an apparently exact, but invented, within-tick interruption frontier.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("type"), str):
        raise GrokProtocolError("Grok event is not a typed JSON object")
    data = dict(raw)
    data["type"] = {
        "response.audio.delta": "response.output_audio.delta",
        "response.audio.done": "response.output_audio.done",
        "response.audio_transcript.delta": "response.output_audio_transcript.delta",
        "response.audio_transcript.done": "response.output_audio_transcript.done",
        "conversation.item.added": "conversation.item.created",
    }.get(data["type"], data["type"])
    if data["type"] == "response.created":
        response = data.get("response") or {}
        return GrokResponseCreatedEvent(type="response.created", event_id=data.get("event_id"),
            response_id=_response_identity(data), status=response.get("status"))
    if data["type"] in {"response.done", "response.cancelled"}:
        response_id = _response_identity(data)
        if data["type"] == "response.done":
            response = dict(data.get("response") or {})
            response["id"] = response_id
            if response.get("status") not in {"completed", "cancelled", "failed", "incomplete"}:
                raise GrokProtocolError("Grok response.done lacks a terminal status")
            data["response"] = response
        else:
            data["response_id"] = response_id
    if data["type"] == "input_audio_buffer.speech_started":
        value = data.get("audio_start_ms")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise GrokProtocolError("Grok speech_started is missing a valid audio_start_ms")
    if data["type"] == "conversation.item.truncated":
        value = data.get("audio_end_ms")
        if (not data.get("item_id") or isinstance(value, bool)
                or not isinstance(value, int) or value < 0):
            raise GrokProtocolError("Grok truncate acknowledgement lacks item-relative frontier")
    if data["type"] == "session.updated":
        return GrokSessionUpdatedEvent(type="session.updated", event_id=data.get("event_id"), raw=data)
    # In particular, input_audio_transcription.updated is cumulative, not a
    # delta. Leave it as an unknown diagnostic event, never append it as text.
    return parse_realtime_event(data)


class GrokLogicalProvider(XAIRealtimeProvider):
    """Thin native-xAI transport with events in the shared ledger's types."""

    def __init__(self, *, realtime_url, realtime_key, model, bridge=None):
        if not isinstance(realtime_key, str) or not realtime_key.strip():
            raise GrokProtocolError("missing explicit Grok API credential")
        super().__init__(api_key=realtime_key, model=model, voice="eve",
                         audio_format=XAIAudioFormat.PCMU, sample_rate=8000)
        self._url = grok_websocket_url(realtime_url, model)
        self._deferred = deque()
        self._open_http_tunnel = getattr(bridge, "open_http_tunnel", None)
        self._wire_receive_sequence = 0
        self._lifecycle_audit_callback = None

    async def connect(self):
        if self.is_connected:
            return
        # Do not fall back to another model/provider, reconnect an active
        # conversation, log credentials, or silently resume a lost session.
        tunnel = None
        try:
            proxy_url = os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
            if proxy_url:
                if not callable(self._open_http_tunnel):
                    raise GrokProtocolError("Grok requires the validated PJLab CONNECT bridge")
                tunnel = await asyncio.to_thread(self._open_http_tunnel, self._url, proxy_url)
            headers = {"Authorization": f"Bearer {self.api_key}"}
            parameters = inspect.signature(websockets.connect).parameters
            if "additional_headers" in parameters:
                wire_options = {"additional_headers": headers, "proxy": None}
            elif "extra_headers" in parameters:
                wire_options = {"extra_headers": headers}
            else:
                raise GrokProtocolError("Unknown websockets authorization API")
            self.ws = await websockets.connect(self._url, **wire_options,
                sock=tunnel, max_size=16 * 1024 * 1024, open_timeout=30, close_timeout=5)
            raw = await self._receive_raw(asyncio.get_running_loop().time() + 20)
            if raw.get("type") not in {"session.created", "conversation.created"}:
                raise GrokProtocolError("Grok handshake did not create a session")
            self.session_id = (raw.get("session") or raw.get("conversation") or {}).get("id")
            if not self.session_id:
                raise GrokProtocolError("Grok handshake is missing a session/conversation ID")
        except Exception:
            if self.ws is not None:
                await self.disconnect()
            elif tunnel is not None:
                tunnel.close()
            raise

    async def _receive_raw(self, deadline):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        message = await asyncio.wait_for(self.ws.recv(), timeout=remaining)
        if not isinstance(message, str):
            raise GrokProtocolError("Grok unexpectedly switched to binary transport")
        try:
            raw = json.loads(message)
        except (ValueError, TypeError):
            raise GrokProtocolError("Grok sent malformed event JSON") from None
        if not isinstance(raw, dict):
            raise GrokProtocolError("Grok sent a non-object event")
        self._wire_receive_sequence += 1
        event_type = raw.get("type", "")
        # Sequence counts every received JSON frame, including omitted audio
        # deltas/pings. Lifecycle evidence therefore retains exact wire order.
        if (self._lifecycle_audit_callback is not None and event_type != "ping"
                and not str(event_type).endswith(".delta")):
            self._lifecycle_audit_callback("grok_raw_lifecycle",
                direction="received", wire_sequence=self._wire_receive_sequence,
                payload=grok_lifecycle_metadata(raw, secrets=(self.api_key, self._url)))
        if raw.get("type") == "error":
            # Do not propagate provider error text that could echo a key/URL.
            code = str((raw.get("error") or {}).get("code", "unknown"))
            for secret in (self.api_key, self._url):
                code = code.replace(secret, "[REDACTED]")
            raise GrokProtocolError(f"Grok provider error code={code}")
        return raw

    async def send_tool_result(self, call_id: str, result: str, request_response: bool = True) -> None:
        """Return a custom function result, then ask Grok to speak.

        xAI's speech-to-speech contract is function_call_output followed by
        response.create. A bare tool result does not continue the turn.
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        await self.ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": result,
            },
        }))
        if request_response:
            await self.ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "instructions": (
                        "The tool result is now available. Continue the user's request: "
                        "call an appropriate business tool if needed, or answer in audio. Do not call "
                        "report_speaker_attribution again until the user speaks again."
                    )
                },
            }))

    async def configure_session(self, *, system_prompt, tools, vad_config=None,
                                modality="audio", audio_format=None, **kwargs):
        if modality != "audio" or audio_format != TELEPHONY_AUDIO_FORMAT:
            raise GrokProtocolError("Grok logical adapter requires audio, PCMU mono 8 kHz")
        # xAI's turn_detection/voice/tools are top-level session fields, unlike
        # OpenAI's GA audio.input.turn_detection. No identity/policy changes.
        audio = self._build_audio_config()
        # Participant names/codes are learned from audio, never ASR keyterms.
        session = {"voice": self.voice, "instructions": system_prompt,
                   "tools": self._format_tools_for_api(tools),
                   "audio": audio,
                   "turn_detection": grok_turn_detection(True)}
        try:
            await self.ws.send(json.dumps({"type": "session.update", "session": session}))
            deadline = asyncio.get_running_loop().time() + 10
            while True:
                raw = await self._receive_raw(deadline)
                if raw.get("type") == "session.updated":
                    verify_grok_auto_response_ack(raw, True)
                    for direction in ("input", "output"):
                        audio = (raw.get("session") or {}).get("audio") or {}
                        fmt = (audio.get(direction) or {}).get("format") or {}
                        if fmt.get("type") != "audio/pcmu":
                            raise GrokProtocolError(f"Grok did not acknowledge PCMU {direction} audio")
                    return
                self._deferred.append(normalize_grok_event(raw))
        except Exception:
            await self.disconnect()
            raise

    async def receive_event_until(self, deadline):
        if self._deferred:
            return self._deferred.popleft()
        try:
            raw = await self._receive_raw(deadline)
        except asyncio.TimeoutError:
            return None
        return normalize_grok_event(raw)

    async def receive_events_for_duration(self, duration_seconds):
        deadline = asyncio.get_running_loop().time() + duration_seconds
        events = []
        while asyncio.get_running_loop().time() < deadline:
            event = await self.receive_event_until(deadline)
            if event is None:
                break
            events.append(event)
        return events

    async def truncate_item(self, *, item_id, content_index, audio_end_ms):
        if (not item_id or content_index != 0 or isinstance(audio_end_ms, bool)
                or not isinstance(audio_end_ms, int) or audio_end_ms < 0):
            raise GrokProtocolError("invalid Grok item-relative truncate request")
        await self.ws.send(json.dumps({"type": "conversation.item.truncate",
            "item_id": item_id, "content_index": content_index, "audio_end_ms": audio_end_ms}))


class LogicalTimeGrokAdapter(LogicalTimeOpenAIAdapter):
    """The common clock/interrupt ledger with only native session control."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._grok_created_response_ids = set()
        self.provider._lifecycle_audit_callback = self._audit

    async def _process_event(self, result, event):
        # xAI server-VAD interruption can omit response.done for the response
        # that was barged into. Close exactly the responses still active before
        # forwarding speech_started; a later named terminal event remains a
        # safe duplicate. Without this closure is_output_drained() stays false
        # forever, pacing waits in silence, and xAI eventually times out.
        if event.type == "input_audio_buffer.speech_started" and self._active_response_ids:
            interrupted = sorted(self._active_response_ids)
            for response_id in interrupted:
                self._mark_response_terminal(response_id, "interrupted_by_user_speech")
            self._audit(
                "grok_speech_started_closed_active_responses",
                response_ids=interrupted,
                trigger_event_type=event.type,
                audio_start_ms=getattr(event, "audio_start_ms", None),
            )

        if event.type == "response.created":
            # A later named response proves that the provider moved past the
            # interrupted item. xAI can omit a duplicate conversation.item
            # truncated ACK forever; retaining that frontier in the drain gate
            # would deadlock ordinary response-opportunity pacing.
            if self._pending_truncate_acks:
                stale = self.pending_truncate_acks
                self._pending_truncate_acks.clear()
                self._audit(
                    "grok_stale_truncate_acks_cleared",
                    stale_frontiers=stale,
                    trigger_event_type=event.type,
                    new_response_id=getattr(event, "response_id", None),
                    policy="later_named_response_precedes_missing_ack",
                )
            response_id = getattr(event, "response_id", None)
            if not isinstance(response_id, str) or not response_id:
                raise GrokProtocolError("Grok created event lost its response ID")
            duplicate = response_id in self._grok_created_response_ids
            terminal_before_created = response_id in self._terminal_response_ids
            self._grok_created_response_ids.add(response_id)
            if not terminal_before_created:
                self._active_response_ids.add(response_id)
            if self._pending_truncate_acks:
                self._record_pre_ack_provider_event(event)
            # Keep tau2's event forwarding/logging, bypass only the shared
            # anonymous +1 counter. A duplicate or late created cannot reopen
            # a completed response or invent a second in-flight response.
            await DiscreteTimeOpenAIAdapter._process_event(self, result, event)
            self._audit("response_created", response_id=response_id,
                status=getattr(event, "status", None), duplicate=duplicate,
                terminal_before_created=terminal_before_created,
                active_response_ids=sorted(self._active_response_ids))
            return
        await super()._process_event(result, event)

    def _mark_response_terminal(self, response_id, status):
        if not isinstance(response_id, str) or not response_id:
            raise GrokProtocolError("Grok terminal event lost its response ID")
        duplicate = response_id in self._terminal_response_ids
        self._terminal_response_ids.add(response_id)
        self._active_response_ids.discard(response_id)
        # Only the explicitly named response ends. Never decrement a generic
        # count or treat an item truncate ACK / VAD onset as generation end.
        self._audit("response_terminal", response_id=response_id, status=status,
            duplicate=duplicate, active_response_ids=sorted(self._active_response_ids))

    async def _async_configure_auto_response(self, enabled, timeout_seconds):
        await self.provider.ws.send(json.dumps({"type": "session.update", "session": {
            "turn_detection": grok_turn_detection(enabled)}}))
        self._audit("auto_response_update_sent", enabled=enabled,
                    transport="grok_native_websocket_ordered_at_logical_boundary")
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        deferred_count = 0
        while asyncio.get_running_loop().time() < deadline:
            event = await self.provider.receive_event_until(deadline)
            if event is None:
                break
            if event.type == "session.updated":
                turn = verify_grok_auto_response_ack(getattr(event, "raw", None) or {}, enabled)
                self._audit("auto_response_update_ack", enabled=enabled,
                            provider="xai", flags_echo_required=False,
                            flags_echo_verified=all(turn.get(field) is enabled for field in
                                                    ("create_response", "interrupt_response")),
                            echoed_create_response=turn.get("create_response"),
                            echoed_interrupt_response=turn.get("interrupt_response"),
                            deferred_event_count=deferred_count)
                return
            self._raise_if_fatal_provider_event(event)
            self._deferred_provider_events.append(event)
            deferred_count += 1
            self._audit("configuration_interleaved_event_deferred", event_type=event.type,
                        enabled=enabled)
        self._audit("auto_response_update_timeout", enabled=enabled, fail_closed=True)
        raise GrokProtocolError("Grok auto-response session.updated acknowledgement timed out")


def make_grok_logical_adapter(*, realtime_url, realtime_key, model,
                              tick_duration_ms, audio_format, bridge):
    """Factory without monkey-patching the GPT runner or shared tau2 modules."""
    if model != "grok-voice-think-fast-2.0":
        raise GrokProtocolError("this run requires the pinned grok-voice-think-fast-2.0 model")
    if audio_format != TELEPHONY_AUDIO_FORMAT:
        raise GrokProtocolError("Grok track requires the unchanged PCMU telephony format")
    provider = GrokLogicalProvider(realtime_url=realtime_url, realtime_key=realtime_key,
                                   model=model, bridge=bridge)
    adapter = LogicalTimeGrokAdapter(tick_duration_ms=tick_duration_ms,
        send_audio_instant=False, provider=provider, audio_format=audio_format)
    adapter.model = model
    adapter._owns_provider = True
    adapter._audit("grok_provider_facade_installed", model=model, voice="eve",
        audio_codec="audio/pcmu", sample_rate=8000,
        interruption_strategy="shared_played_prefix_ledger_server_vad",
        enrollment_strategy="unchanged_server_vad_explicit_auto_response_flags")
    return adapter
