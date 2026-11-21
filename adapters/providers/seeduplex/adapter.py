"""Seeduplex 3.0 native wire protocol over the common logical-media contract.

The shared Gemini wrapper supplies only the tick/played-audio ledger and bounded
receive queue. No Gemini wire messages, tools or resumption logic are used.
Cloud inference is not frozen by simulator pauses. Registration and all task
information come from the common runner; no privileged text is injected here.
"""
from __future__ import annotations

import asyncio
import audioop
import base64
from copy import deepcopy
from dataclasses import dataclass
import inspect
import json
import time
import uuid
from urllib.parse import urlsplit

import websockets
from tau2.data_model.message import ToolCall
from tau2.data_model.usage import UsageRecord
from adapters.providers.gemini.adapter import GeminiLogicalAdapter

ENDPOINT = 'wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue'
MODEL = '1.2.6.1'


class SeeduplexProtocolError(RuntimeError):
    pass


@dataclass
class SeeduplexEvent:
    type: str
    event_id: str | None = None


class SeeduplexLogicalAdapter(GeminiLogicalAdapter):
    def __init__(self, *, key, model, bridge, tick_duration_ms=200, proxy=None,
                 realtime_url=ENDPOINT, voice='zh_female_vv_jupiter_bigtts'):
        if model != MODEL:
            raise ValueError('Seeduplex 3.0 requires the explicit model 1.2.6.1')
        u = urlsplit(realtime_url)
        if u.scheme != 'wss' or not u.netloc or u.username or u.password:
            raise ValueError('Seeduplex requires an explicit secure WebSocket endpoint')
        if not key or not voice:
            raise ValueError('Seeduplex credential and voice are required')
        super().__init__(key=key, model=model, bridge=bridge,
                         tick_duration_ms=tick_duration_ms, proxy=proxy)
        self.session_id = None  # Only a real session.created establishes identity.
        self._url, self._voice = realtime_url, voice
        self._nonce, self._serial = uuid.uuid4().hex, 0
        self._closed_ack = None
        self._muted = False
        self._closing = False
        self._batches = []
        self._results = {}
        self._sent_results = {}
        self._response_id = None
        self._suppressed = set()
        self._seen_usage = set()
        self._seen_tool_payloads = {}
        self._wire_sequence = 0

    def connect(self, system_prompt, tools, vad_config=None, modality='audio'):
        if modality != 'audio' or vad_config is not None:
            raise ValueError('Seeduplex uses native audio activity detection')
        declarations = [deepcopy(t.openai_schema['function']) for t in tools]
        self._setup = {'type': 'session.create', 'session': {
            'model': self.model, 'instructions': system_prompt,
            'audio': {'input': {'format': {'type': 'pcm', 'rate': 16000}},
                      'output': {'format': {'type': 'pcm_s16le', 'rate': 24000},
                                 'voice': self._voice}},
            'tools': [dict(type='function', **f) for f in declarations]}}
        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(self._connect_transport(), timeout=75)
            self._connected = True
            self.audit('adapter_contract', provider='seeduplex', model=self.model,
                       input_wire='pcm16_16000', output_wire='pcm16_24000',
                       benchmark_interface='ulaw_8000', tick_ms=self.tick_duration_ms,
                       input_packet_ms=20, cloud_pause_supported=False,
                       cloud_pause_support_basis='no_documented_freeze_resume_interface',
                       native_history_truncation_supported=False,
                       tools=[f['name'] for f in declarations],
                       native_client_activity_policy='official_demo_clear_queue_on_asr_started',
                       initial_context_and_registration='common_runner_unchanged',
                       extra_roster=False, hotwords=False, web_search=False,
                       session_id_origin='provider_session_created')
        except Exception:
            try:
                self._bg_loop.run_coroutine(self._disconnect(), timeout=10)
            finally:
                self._bg_loop.stop()
            raise

    async def _send(self, message):
        message = dict(message)
        self._serial += 1
        message.setdefault('event_id', 'mu_seed_' + self._nonce + '_' + str(self._serial))
        await self._ws.send(json.dumps(message, ensure_ascii=False))
        return message['event_id']

    async def _connect_transport(self, resume=False):
        if resume:
            raise SeeduplexProtocolError('Automatic session stitching is not enabled')
        sock = None
        try:
            if self.proxy:
                sock = await asyncio.wait_for(asyncio.to_thread(
                    self.bridge.open_http_tunnel, self._url, self.proxy), 35)
            options = dict(sock=sock, open_timeout=30, close_timeout=3,
                           max_size=16 * 2**20, max_queue=64)
            if 'additional_headers' in inspect.signature(websockets.connect).parameters:
                options.update(additional_headers={'X-Api-Key': self.key}, proxy=None)
            else:
                options['extra_headers'] = {'X-Api-Key': self.key}
            self._ws = await websockets.connect(self._url, **options)
            await self._send(self._setup)
            raw = json.loads(await asyncio.wait_for(self._ws.recv(), 25))
            if raw.get('type') != 'session.created' or not raw.get('session', {}).get('id'):
                raise SeeduplexProtocolError(self._safe('Session creation failed: ' + json.dumps(raw)))
            self.session_id = raw['session']['id']
            self._closed_ack = asyncio.Event()
            self._reader = asyncio.create_task(self._receive())
            self.audit('session_setup_complete', model=self.model, session_id=self.session_id,
                       fresh_session=True, sent_input_replay=False)
        except Exception:
            if self._ws is None and sock is not None:
                sock.close()
            raise

    def _safe(self, value):
        return str(value).replace(self.key, '[REDACTED]')[:2000]

    async def _receive(self):
        try:
            async for raw in self._ws:
                if not isinstance(raw, str):
                    raise SeeduplexProtocolError('Expected a JSON text frame')
                msg = json.loads(raw)
                if not isinstance(msg, dict) or not isinstance(msg.get('type'), str):
                    raise SeeduplexProtocolError('Expected a typed event object')
                self._wire_sequence += 1
                self.audit('wire_event', sequence=self._wire_sequence, type=msg['type'],
                           event_id=msg.get('event_id'), response_id=msg.get('response_id'),
                           payload_bytes=len(raw), paused=self._paused is not None)
                if msg['type'] == 'session.closed':
                    self._closed_ack.set()
                    if not self._closing:
                        self._transport_error = 'Seeduplex closed the active session'
                    continue
                self._messages.append((time.monotonic(), msg, len(raw)))
                self._queued_bytes += len(raw)
                if self._queued_bytes > 256 * 2**20:
                    raise SeeduplexProtocolError('Unconsumed receive queue exceeded 256 MiB')
            if not self._closing:
                self._transport_error = 'Seeduplex WebSocket closed'
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._transport_error = self._safe(type(exc).__name__ + ': ' + str(exc))

    async def _ensure_transport(self):
        if self._transport_error:
            raise SeeduplexProtocolError(self._transport_error)

    def disconnect(self):
        try:
            if self._bg_loop.is_running:
                self._bg_loop.run_coroutine(self._disconnect(), timeout=15)
        finally:
            if self._bg_loop.is_running:
                self._bg_loop.stop()
            self._connected = False

    def resume_logical_clock(self, reason, logical_time_ms):
        if self._paused is None or self._paused[:2] != (reason, logical_time_ms):
            raise RuntimeError('logical pause frontier changed')
        elapsed = (time.monotonic() - self._paused[2]) * 1000
        self._bg_loop.run_coroutine(self._end_pause_signal(), timeout=15)
        self._paused = None
        self.audit('logical_resume', reason=reason, logical_time_ms=logical_time_ms,
                   wall_elapsed_ms=elapsed, buffered_wire_bytes=self._queued_bytes,
                   provider_action='unmute_if_previously_muted',
                   cloud_generation_frozen=False, audio_bytes_replayed=0)

    async def _disconnect(self):
        await self._end_pause_signal()
        self._closing = True
        try:
            if self._ws is not None and self._closed_ack is not None:
                await self._send({'type': 'session.close'})
                try:
                    await asyncio.wait_for(self._closed_ack.wait(), 3)
                    self.audit('session_close_ack', received=True)
                except asyncio.TimeoutError:
                    self.audit('session_close_ack', received=False)
        finally:
            if self._reader:
                self._reader.cancel()
                await asyncio.gather(self._reader, return_exceptions=True)
            if self._ws is not None:
                await self._ws.close()

    def configure_auto_response(self, enabled):
        self.audit('auto_response_request', requested=enabled, supported=False,
                   provider_action='none', enrollment_replies_preserved=True)

    def pause_logical_clock(self, reason, logical_time_ms):
        if self._paused is not None:
            raise RuntimeError('nested logical pause')
        self._paused = (reason, logical_time_ms, time.monotonic())
        self._bg_loop.run_coroutine(self._begin_pause_signal(), timeout=15)
        self.audit('logical_pause', reason=reason, logical_time_ms=logical_time_ms,
                   provider_action='delayed_input_mute_after_1s',
                   cloud_generation_frozen=False, response_cancel_sent=False)

    async def _signal_long_audio_gap(self):
        try:
            await asyncio.sleep(1)
            if self._paused is not None:
                await self._send({'type': 'input_audio_mute.commit'})
                self._muted = True
                self.audit('pause_input_muted', audio_bytes_sent=0,
                           logical_clock_advanced=False, cloud_generation_frozen=False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._transport_error = self._safe(exc)

    async def _end_pause_signal(self):
        if self._pause_signal_task is not None:
            self._pause_signal_task.cancel()
            await asyncio.gather(self._pause_signal_task, return_exceptions=True)
            self._pause_signal_task = None
        if self._muted and not self._closing:
            await self._send({'type': 'input_audio_unmute.commit'})
            self._muted = False
            self.audit('pause_input_unmuted', audio_bytes_replayed=0)

    def is_output_drained(self):
        return not (self._turn_active or self._buffered_agent_audio or self._messages
                    or self._pending_tool_results or self._batches)

    def send_tool_result(self, call_id, result, request_response=True, is_error=False):
        if call_id not in self._calls:
            raise SeeduplexProtocolError('Unmatched tool result: ' + call_id)
        self._pending_tool_results.append((call_id, result, request_response, is_error))

    async def _flush_pending_tool_results(self):
        for cid, value, request_response, is_error in self._pending_tool_results:
            text = value if isinstance(value, str) else json.dumps(value)
            if cid in self._sent_results:
                if self._sent_results[cid] != text:
                    raise SeeduplexProtocolError('Conflicting repeated tool result: ' + cid)
                continue
            if cid in self._results and self._results[cid] != text:
                raise SeeduplexProtocolError('Conflicting queued tool result: ' + cid)
            self._results[cid] = text
        self._pending_tool_results.clear()
        for batch in list(self._batches):
            if not all(cid in self._results for cid in batch):
                continue
            items = [{'call_id': cid, 'role': 'tool',
                      'content': [{'type': 'input_text', 'text': self._results[cid]}]}
                     for cid in batch]
            event_id = await self._send({'type': 'conversation.item.create', 'items': items})
            for cid in batch:
                self._sent_results[cid] = self._results.pop(cid)
            self._batches.remove(batch)
            self.audit('tool_results_sent', call_ids=list(batch), event_id=event_id,
                       response_create_sent=False, continuation='native_after_all_results')

    async def _execute_tick(self, user_audio, tick_number, result, tick_start):
        pcm = audioop.ulaw2lin(user_audio, 2)
        pcm, self._input_state = audioop.ratecv(pcm, 2, 1, 8000, 16000, self._input_state)
        async def send():
            for offset in range(0, len(pcm), 640):
                packet = pcm[offset:offset + 640]
                await self._send({'type': 'input_audio_buffer.append',
                                  'audio': base64.b64encode(packet).decode()})
                await asyncio.sleep(len(packet) / 32000)
        async def receive():
            deadline = tick_start + self.tick_duration_ms / 1000
            while True:
                while self._messages:
                    arrival, msg, size = self._messages.popleft()
                    self._queued_bytes -= size
                    self._process_message(result, msg, arrival, tick_start)
                await self._ensure_transport()
                left = deadline - asyncio.get_running_loop().time()
                if left <= 0:
                    return
                await asyncio.sleep(min(.005, left))
        await asyncio.gather(send(), receive())

    def _clear_unplayed(self, result, arrival, tick_start, response_id=None):
        # Without provider word timestamps, receipt time is the only known local
        # frontier. Never rewind already-played ticks or invent cloud truncation.
        budget = max(0, min(self.bytes_per_tick, int((arrival-tick_start)*8000)))
        kept, dropped = [], 0
        for audio, item in result.agent_audio_chunks:
            affected = response_id is None or item == response_id
            if affected:
                prefix = audio[:budget];budget -= len(prefix)
                dropped += len(audio)-len(prefix)
                if prefix:kept.append((prefix,item))
            else:kept.append((audio,item))
        result.agent_audio_chunks[:] = kept
        buffered=[]
        for audio,item in self._buffered_agent_audio:
            if response_id is None or item == response_id:dropped+=len(audio)
            else:buffered.append((audio,item))
        self._buffered_agent_audio[:] = buffered
        result.truncated_audio_bytes += dropped
        interrupted = dropped > 0 or (self._turn_active and (response_id is None or response_id == self._response_id))
        if not interrupted:
            self.audit('old_cancel_no_playback_effect', response_id=response_id)
            return
        result.was_truncated = True
        result.vad_events.append('interrupted')
        self.audit('native_interruption', response_id=response_id,
                   dropped_unplayed_bytes=dropped, provider_history_truncate_sent=False,
                   native_history_alignment_unverified=True)

    def _process_message(self, result, msg, arrival, tick_start):
        typ = msg['type'];rid=msg.get('response_id')
        result.events.append(SeeduplexEvent(typ, msg.get('event_id')))
        if typ == 'error':
            raise SeeduplexProtocolError(self._safe('Seeduplex API: '+json.dumps(msg)))
        if typ == 'conversation.item.input_audio_transcription.started':
            # Official Seeduplex client clears queued playback on this native
            # signal. This is not an oracle detector or an injected cancel.
            if self._turn_active or result.agent_audio_chunks or self._buffered_agent_audio:
                self._clear_unplayed(result, arrival, tick_start)
            result.vad_events.append('speech_started')
        elif typ == 'conversation.item.input_audio_transcription.completed':
            result.vad_events.append('speech_stopped')
        elif typ == 'response.canceled':
            self._clear_unplayed(result, arrival, tick_start, rid)
            if rid:self._suppressed.add(rid)
            if not rid or rid == self._response_id:self._turn_active=False
        elif typ in {'response.output_audio.started','response.output_audio.delta'}:
            if rid in self._suppressed:
                self.audit('cancelled_audio_discarded', response_id=rid)
                return
            if typ == 'response.output_audio.started' or self._item is None:
                if rid != self._response_id or self._item is None:
                    self._output_state = None
                    self._turn += 1
                    self._item = rid or 'seeduplex-local-audio-'+str(self._turn)
                    self._response_id = rid
                    self.audit('audio_turn_started', response_id=rid, item_id=self._item,
                               local_fallback_id=not bool(rid))
            self._turn_active=True
            if typ.endswith('.delta'):
                raw=base64.b64decode(msg['delta'], validate=True)
                if len(raw)%2:raise SeeduplexProtocolError('Odd PCM16 output length')
                converted,self._output_state=audioop.ratecv(raw,2,1,24000,8000,self._output_state)
                result.agent_audio_chunks.append((audioop.lin2ulaw(converted,2),self._item))
        elif typ == 'response.function_call_arguments.done':
            batch=[]
            for fc in msg.get('items',[]):
                cid,name=fc.get('call_id'),fc.get('name');args=fc.get('arguments')
                if not cid or not name:raise SeeduplexProtocolError('Tool call lacks real call_id/name')
                if isinstance(args,str):args=json.loads(args)
                if not isinstance(args,dict):raise SeeduplexProtocolError('Tool arguments must be an object')
                signature=json.dumps([name,args],sort_keys=True)
                if cid in self._seen_tool_payloads:
                    if self._seen_tool_payloads[cid]!=signature:raise SeeduplexProtocolError('Conflicting duplicate call_id')
                    continue
                self._seen_tool_payloads[cid]=signature;self._calls[cid]=(cid,name)
                result.tool_calls.append(ToolCall(id=cid,name=name,arguments=args));batch.append(cid)
                self.audit('native_tool_call',call_id=cid,name=name)
            if batch:self._batches.append(tuple(batch));self._turn_active=True
        elif typ == 'conversation.item.added':
            self.audit('conversation_item_ack',call_ids=[x.get('call_id') for x in msg.get('items',[]) if x.get('call_id')])
        elif typ == 'response.done':
            self._turn_active=False
            usage=(msg.get('response') or {}).get('usage')
            if usage:
                token=msg.get('event_id') or json.dumps(msg,sort_keys=True)
                if token not in self._seen_usage:
                    self._seen_usage.add(token)
                    self.record_usage(UsageRecord.from_openai_realtime_usage(
                        usage,provider='seeduplex',model=self.model,
                        scope_id=rid or msg.get('event_id')))
                self.audit('usage_metadata',usage=usage)
        elif typ == 'response.output_audio.done':
            self._turn_active=False
            # Multiple sentence segments may share a response ID. Preserve the
            # resampler across them; a new actual ID establishes a new turn.


def make_seeduplex_logical_adapter(**kwargs):
    return SeeduplexLogicalAdapter(**kwargs)
