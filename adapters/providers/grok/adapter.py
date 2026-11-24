"""xAI wire facade for the existing single-session logical-time contract.

Reuse tau2's xAI audio/tool wire methods and μ-Bench's played-prefix ledger.
Do not use xAI's old typed parser: it drops VAD timestamps and truncate ACKs.
Registration uses native server VAD and the shared spoken instructions.
No undocumented auto-response switches or model-specific task coaching are added.
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
from tau2.voice.audio_native.openai.events import (
    ResponseCreatedEvent, SessionUpdatedEvent, ConversationItemTruncatedEvent, parse_realtime_event,
)
from tau2.voice.audio_native.xai.provider import XAIAudioFormat, XAIRealtimeProvider

from adapters.providers.openai.adapter import LogicalTimeOpenAIAdapter


class GrokProtocolError(RuntimeError):
    """Provider contract was not demonstrated; never silently change protocol."""


class GrokSessionUpdatedEvent(SessionUpdatedEvent):
    """Retain xAI's acknowledged settings instead of dropping them on parsing."""

    raw: dict


class GrokTruncatedEvent(ConversationItemTruncatedEvent):
    """Keep the content index for exact request/ACK correlation."""

    content_index: int


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
                "status", "call_id", "name"):
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
        out["session"] = {key: safe(session.get(key)) for key in ("id", "model")}
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


def grok_turn_detection(enabled=None):
    # The public xAI contract supplies native VAD. It does not document the
    # OpenAI create_response/interrupt_response switches as silence controls.
    return {"type": "server_vad"}


def verify_grok_auto_response_ack(raw, enabled):
    session = raw.get("session") if isinstance(raw, dict) else None
    turn = session.get("turn_detection") if isinstance(session, dict) else None
    if not isinstance(turn, dict) or turn.get("type") != "server_vad":
        raise GrokProtocolError("Grok did not acknowledge server_vad registration policy")
    # Confirm native VAD only; registration silence is not a demonstrated flag.
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
        content_index = data.get("content_index")
        if isinstance(content_index, bool) or not isinstance(content_index, int) or content_index < 0:
            raise GrokProtocolError("Grok truncate acknowledgement lacks content_index")
        return GrokTruncatedEvent(type=data["type"], event_id=data.get("event_id"),
            item_id=data["item_id"], content_index=content_index, audio_end_ms=value)
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
        self._reader_task = None
        self._incoming = asyncio.Queue()
        self._queued_wire_bytes = 0
        self._receive_error = None
        self._closing = False
        self.max_queued_wire_bytes = 256 * 2**20

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
            if raw.get("type") == "error":
                error = raw.get("error") or {}
                raise GrokProtocolError("Grok handshake error code="
                    + str(error.get("code", "unknown")) + ": " + str(error.get("message", "")))
            if raw.get("type") not in {"session.created", "conversation.created"}:
                raise GrokProtocolError("Grok handshake did not create a session")
            self.session_id = (raw.get("session") or raw.get("conversation") or {}).get("id")
            if not self.session_id:
                raise GrokProtocolError("Grok handshake is missing a session/conversation ID")
            self._closing = False
            self._receive_error = None
            self._reader_task = asyncio.create_task(self._receiver_loop())
        except Exception:
            if self.ws is not None:
                await self.disconnect()
            elif tunnel is not None:
                tunnel.close()
            raise

    async def disconnect(self):
        self._closing = True
        if self._reader_task is not None:
            self._reader_task.cancel()
            await asyncio.gather(self._reader_task, return_exceptions=True)
            self._reader_task = None
        if self.ws is not None:
            await self.ws.close()
            self.ws = None
        if self._lifecycle_audit_callback is not None:
            self._lifecycle_audit_callback("grok_transport_closed",
                pending_wire_events=self._incoming.qsize(),
                pending_wire_bytes=self._queued_wire_bytes)

    async def _receiver_loop(self):
        # Own ws.recv even while simulator computation freezes logical media.
        # Only retain wire events here: never play audio, advance time, execute
        # a function, request a response or truncate cloud history in this task.
        try:
            while not self._closing:
                raw, size = await self._read_wire()
                if self._queued_wire_bytes + size > self.max_queued_wire_bytes:
                    raise GrokProtocolError("Grok unconsumed receive queue exceeded limit")
                self._queued_wire_bytes += size
                self._incoming.put_nowait((raw, size))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closing:
                message = type(exc).__name__ + ": " + str(exc)
                for secret in (self.api_key, self._url):
                    message = message.replace(secret, "[REDACTED]")
                self._receive_error = message[:1500]
                # Sentinel wakes an idle consumer without fabricating an API event.
                self._incoming.put_nowait((None, 0))

    @property
    def has_pending_events(self):
        return not self._incoming.empty()

    async def _receive_raw(self, deadline):
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        if self._reader_task is None:
            raw, _ = await asyncio.wait_for(self._read_wire(), timeout=remaining)
            return raw
        if self._incoming.empty() and self._receive_error:
            raise GrokProtocolError(self._receive_error)
        raw, size = await asyncio.wait_for(self._incoming.get(), timeout=remaining)
        self._queued_wire_bytes -= size
        if raw is None:
            raise GrokProtocolError(self._receive_error or "Grok transport stopped")
        return raw

    async def _read_wire(self):
        message = await self.ws.recv()
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
            # Preserve correlation and error classification for the common
            # continuation scheduler, while redacting provider-echoed secrets.
            raw = dict(raw)
            error = dict(raw.get("error") or {})
            for key, value in list(error.items()):
                if isinstance(value, str):
                    for secret in (self.api_key, self._url):
                        value = value.replace(secret, "[REDACTED]")
                    error[key] = value
            raw["error"] = error
        return raw, len(message)

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
            await self.request_response("grok_tool_" + call_id)

    async def request_response(self, event_id):
        if not self.is_connected:
            raise RuntimeError("Not connected to API")
        # Native continuation only: the same public protocol already delivered
        # to every model supplies task behavior and attribution requirements.
        await self.ws.send(json.dumps({"type": "response.create", "event_id": event_id}))

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
                if raw.get("type") == "error":
                    error = raw.get("error") or {}
                    raise GrokProtocolError("Grok session configuration failed: "
                                            + str(error.get("code", "unknown")) + ": "
                                            + str(error.get("message", "")))
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
        self.provider._lifecycle_audit_callback = self._audit
        self._grok_confirmed_truncates = set()
        self._grok_pending_truncate_at = {}
        self._grok_truncate_ack_timeout_ms = 10000

    def connect(self, *args, **kwargs):
        # A new connection owns a fresh ledger. No reconnect/history replay is added.
        if self._connected:
            self.disconnect()
        self._grok_confirmed_truncates.clear()
        self._grok_pending_truncate_at.clear()
        self._pending_truncate_acks.clear()
        return super().connect(*args, **kwargs)

    async def _request_item_truncate(self, **request):
        key = (request["item_id"], request["content_index"], request["audio_end_ms"])
        state = ("confirmed" if key in self._grok_confirmed_truncates else
                 "pending" if key in self._grok_pending_truncate_at else None)
        if state:
            self._audit("grok_duplicate_truncate_skipped", item_id=key[0],
                content_index=key[1], audio_end_ms=key[2], prior_state=state,
                new_wait_created=False)
            return
        await super()._request_item_truncate(**request)
        self._grok_pending_truncate_at[key] = self._current_logical_time_ms

    def _check_truncate_ack_timeout(self):
        expired = [{"item_id":key[0], "content_index":key[1], "audio_end_ms":key[2]}
            for key, started in self._grok_pending_truncate_at.items()
            if self._current_logical_time_ms - started >= self._grok_truncate_ack_timeout_ms]
        if expired:
            self._audit("grok_truncate_ack_timeout", targets=expired,
                timeout_logical_ms=self._grok_truncate_ack_timeout_ms,
                server_history_alignment_verified=False, technical_error=True)
            raise GrokProtocolError("Grok truncate acknowledgement timed out after 10s of media time")

    async def _execute_tick(self, *args, **kwargs):
        await super()._execute_tick(*args, **kwargs)
        self._check_truncate_ack_timeout()

    async def _process_event(self, result, event):
        # xAI server-VAD interruption can omit response.done for the response
        # that was barged into. Close exactly the responses still active before
        # forwarding speech_started; a later named terminal event remains a
        # safe duplicate. Without this closure is_output_drained() stays false
        # forever, pacing waits in silence, and xAI eventually times out.
        if event.type == "input_audio_buffer.speech_started" and self._active_response_ids:
            interrupted = sorted(self._active_response_ids)
            self._suppressed_response_ids.update(interrupted)
            for response_id in interrupted:
                self._mark_response_terminal(response_id, "interrupted_by_user_speech")
            self._audit(
                "grok_speech_started_closed_active_responses",
                response_ids=interrupted,
                trigger_event_type=event.type,
                audio_start_ms=getattr(event, "audio_start_ms", None),
            )

        if isinstance(event, GrokTruncatedEvent):
            result.events.append(event)
            key = (event.item_id, event.content_index, event.audio_end_ms)
            if key in self._grok_pending_truncate_at:
                self._record_truncate_ack(item_id=event.item_id, audio_end_ms=event.audio_end_ms)
                self._grok_pending_truncate_at.pop(key)
                self._grok_confirmed_truncates.add(key)
                self._audit("grok_truncate_target_confirmed", item_id=key[0],
                    content_index=key[1], audio_end_ms=key[2])
            else:
                self._audit("grok_duplicate_truncate_ack" if key in self._grok_confirmed_truncates
                    else "grok_unmatched_truncate_ack", item_id=key[0],
                    content_index=key[1], audio_end_ms=key[2], new_wait_created=False)
            return
        # A new response is not a truncate confirmation. Keep accepting late ACKs.
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
        self._audit("auto_response_request", requested=enabled, supported=False,
                    provider_action="none", enrollment_replies_preserved=True,
                    registration_policy="shared_spoken_instructions_native_server_vad")

    def is_output_drained(self):
        return (not self.provider.has_pending_events
                and super().is_output_drained())


def make_grok_logical_adapter(*, realtime_url, realtime_key, model,
                              tick_duration_ms, audio_format, bridge):
    """Factory without monkey-patching the GPT runner or shared tau2 modules."""
    if model not in {"grok-voice-3.0", "grok-voice-2.0", "grok-voice-think-fast-2.0"}:
        raise GrokProtocolError("Grok requires an explicit supported version, not a floating alias")
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
        enrollment_strategy="spoken_instructions_native_server_vad",
        continuation_strategy="shared_acknowledged_output_epoch_scheduler",
        transport_receive="continuous_bounded_queue", cloud_pause_supported=False)
    return adapter
