"""Gemini Live wire transport with μ-Bench's existing 8 kHz logical-media interface.

No text roster, future user text, synthetic cancel, or replay of sent audio.
Cloud inference is NOT frozen by a local simulator pause.
"""
import asyncio
import audioop
import base64
from collections import deque
import json
import ssl
import time
import uuid

import websockets
from tau2.data_model.message import ToolCall
from tau2.voice.audio_native.adapter import DiscreteTimeAdapter
from tau2.voice.audio_native.async_loop import BackgroundAsyncLoop
from tau2.voice.audio_native.gemini.events import BaseGeminiEvent


class GeminiLogicalAdapter(DiscreteTimeAdapter):
    def __init__(self, *, key, model, bridge, tick_duration_ms=200, proxy):
        super().__init__(tick_duration_ms, send_audio_instant=False)
        self.key, self.model, self.bridge, self.proxy = key, model, bridge, proxy
        self.provider = self
        self.session_id = 'client-gemini-' + uuid.uuid4().hex
        self.audit_records = []
        self._bg_loop = BackgroundAsyncLoop()
        self._connected = False
        self._paused = None
        self._pause_signal_task = None
        self._messages = deque()
        self._queued_bytes = 0
        self._reader = self._ws = None
        self._transport_error = None
        self._handle = None
        self._goaway = False
        self._deadline = None
        self._resumptions = 0
        self._input_state = self._output_state = None
        self._turn_active = False
        self._calls = {}
        self._seen_calls = set()
        self._cancelled_calls = set()
        self._played = {}
        self._turn = 0
        self._item = None

    def audit(self, kind, **values):
        self.audit_records.append(dict(kind=kind, wall_time=time.time(), **values))

    @property
    def is_connected(self):
        return self._connected

    def connect(self, system_prompt, tools, vad_config=None, modality='audio'):
        assert modality == 'audio'
        declarations = []
        for tool in tools:
            f = tool.openai_schema['function']
            declarations.append(dict(name=f['name'], description=f.get('description', ''),
                                     parametersJsonSchema=f['parameters'], behavior='BLOCKING'))
        self._setup = dict(model='models/' + self.model.removeprefix('models/'),
                           generationConfig=dict(responseModalities=['AUDIO']),
                           systemInstruction=dict(parts=[dict(text=system_prompt)]),
                           realtimeInputConfig=dict(automaticActivityDetection=dict(disabled=False),
                                                    activityHandling='START_OF_ACTIVITY_INTERRUPTS'),
                           sessionResumption={}, contextWindowCompression=dict(slidingWindow={}))
        if declarations:
            self._setup['tools'] = [dict(functionDeclarations=declarations)]
        self._bg_loop.start()
        try:
            self._bg_loop.run_coroutine(self._connect_transport(), timeout=90)
            self._connected = True
            self.audit('adapter_contract', input_wire='pcm16_16000', output_wire='pcm16_24000',
                       benchmark_interface='ulaw_8000', tick_ms=self.tick_duration_ms,
                       input_packet_ms=20, native_vad=True, cloud_pause_supported=False,
                       native_history_truncation_supported=False, context_compression='native_sliding_window',
                       tool_behavior='BLOCKING', tools=[f['name'] for f in declarations],
                       session_id_origin='client_generated_not_provider_receipt')
        except Exception:
            self._bg_loop.stop()
            raise

    async def _connect_transport(self, resume=False):
        url = ('wss://generativelanguage.googleapis.com/ws/'
               'google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent')
        sock = await asyncio.wait_for(asyncio.to_thread(self.bridge.open_http_tunnel, url, self.proxy), 40) if self.proxy else None
        self._ws = await websockets.connect(url, sock=sock, ssl=ssl.create_default_context(),
                                           extra_headers={'x-goog-api-key': self.key}, open_timeout=30,
                                           close_timeout=3, max_size=32 * 2**20, max_queue=64)
        setup = dict(self._setup)
        setup['sessionResumption'] = dict(handle=self._handle) if resume else {}
        await self._ws.send(json.dumps(dict(setup=setup)))
        first = json.loads(await asyncio.wait_for(self._ws.recv(), 30))
        if 'setupComplete' not in first:
            raise RuntimeError('Gemini setup failed: ' + str(first).replace(self.key, '[REDACTED]')[:1000])
        self._transport_error = None
        self._goaway = False
        self._deadline = None
        self._reader = asyncio.create_task(self._receive())
        self.audit('session_resumed' if resume else 'session_setup_complete', model=self.model,
                   physical_connection=self._resumptions + 1, sent_input_replay=False)

    async def _receive(self):
        try:
            async for raw in self._ws:
                msg = json.loads(raw)
                now = time.monotonic()
                update = msg.get('sessionResumptionUpdate')
                if update is not None:
                    if update.get('resumable') and update.get('newHandle'):
                        self._handle = update['newHandle']
                    self.audit('session_resumption_available', resumable=bool(update.get('resumable')))
                    msg.pop('sessionResumptionUpdate')
                if 'goAway' in msg:
                    self._goaway = True
                    left = float(msg['goAway'].get('timeLeft', '30s').rstrip('s'))
                    self._deadline = now + max(0, left - 5)
                    self.audit('go_away', time_left_seconds=left)
                if msg:
                    self._messages.append((now, msg, len(raw)))
                    self._queued_bytes += len(raw)
                    if self._queued_bytes > 256 * 2**20:
                        raise RuntimeError('Gemini unconsumed receive queue exceeded 256 MiB')
            self._transport_error = 'Gemini WebSocket closed'
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._transport_error = (type(exc).__name__ + ': ' + str(exc)).replace(self.key, '[REDACTED]')[:2000]

    async def _ensure_transport(self):
        due = self._goaway and (self._transport_error or not self._turn_active
                               or (self._deadline is not None and time.monotonic() >= self._deadline))
        if due:
            if not self._handle or self._resumptions >= 16:
                raise RuntimeError('Gemini planned reconnect has no usable native handle or reached limit')
            if self._reader:
                self._reader.cancel()
                await asyncio.gather(self._reader, return_exceptions=True)
            await self._ws.close()
            self._resumptions += 1
            await self._connect_transport(resume=True)
        elif self._transport_error:
            raise RuntimeError(self._transport_error)

    def disconnect(self):
        if self._bg_loop.is_running:
            self._bg_loop.run_coroutine(self._disconnect(), timeout=15)
            self._bg_loop.stop()
        self._connected = False

    async def _disconnect(self):
        await self._end_pause_signal()
        if self._reader:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
        if self._ws:
            await self._ws.close()

    def configure_auto_response(self, enabled):
        # No Gemini equivalent to create_response=false. The benchmark's spoken
        # enrollment instruction is retained; all actual replies remain evidence.
        self.audit('auto_response_request', requested=enabled, supported=False,
                   provider_action='none', enrollment_replies_preserved=True)

    def pause_logical_clock(self, reason, logical_time_ms):
        if self._paused is not None:
            raise RuntimeError('nested logical pause')
        self._paused = (reason, logical_time_ms, time.monotonic())
        self.audit('logical_pause', reason=reason, logical_time_ms=logical_time_ms,
                   provider_action='flush_ready_tool_results_then_audio_stream_end_1s_and_every_30s', cloud_generation_frozen=False)
        self._bg_loop.run_coroutine(self._begin_pause_signal(), timeout=15)

    async def _begin_pause_signal(self):
        # These results already exist; no user/model computation is performed here.
        await self._flush_pending_tool_results()
        self._pause_signal_task = asyncio.create_task(self._signal_long_audio_gap())

    async def _signal_long_audio_gap(self):
        try:
            await asyncio.sleep(1.0)
            while self._paused is not None and not self._transport_error:
                await self._ws.send(json.dumps(dict(realtimeInput=dict(audioStreamEnd=True))))
                self.audit('pause_audio_stream_end', logical_time_ms=self._paused[1],
                           audio_bytes_sent=0, logical_clock_advanced=False, heartbeat_interval_seconds=30)
                await asyncio.sleep(30)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._transport_error = (type(exc).__name__ + ': ' + str(exc)).replace(self.key, '[REDACTED]')[:2000]

    async def _end_pause_signal(self):
        if self._pause_signal_task is not None:
            self._pause_signal_task.cancel()
            await asyncio.gather(self._pause_signal_task, return_exceptions=True)
            self._pause_signal_task = None

    def resume_logical_clock(self, reason, logical_time_ms):
        if self._paused is None or self._paused[:2] != (reason, logical_time_ms):
            raise RuntimeError('logical pause frontier changed')
        elapsed = (time.monotonic() - self._paused[2]) * 1000
        self._bg_loop.run_coroutine(self._end_pause_signal(), timeout=15)
        self._paused = None
        self.audit('logical_resume', reason=reason, logical_time_ms=logical_time_ms,
                   wall_elapsed_ms=elapsed, buffered_wire_bytes=self._queued_bytes,
                   provider_action='none', cloud_generation_frozen=False)

    def is_output_drained(self):
        meaningful = any(m.get('serverContent') or m.get('toolCall') for _, m, _ in list(self._messages))
        return not (self._turn_active or self._buffered_agent_audio or self._pending_tool_results or meaningful)

    def run_tick(self, user_audio, tick_number=None):
        if self._paused is not None:
            raise RuntimeError('cannot advance logical clock during simulator pause')
        if not self.is_connected:
            raise RuntimeError('not connected')
        if tick_number is None:
            tick_number = self._tick_count
        self._tick_count = tick_number + 1
        result = self._bg_loop.run_coroutine(self._run(user_audio, tick_number), timeout=120)
        # The user simulator and judges use ASR of these retained waveforms, never
        # generated-but-unplayed output transcription or model reasoning text.
        result.proportional_transcript = ''
        for chunk, item in result.agent_audio_chunks:
            self._played[item] = self._played.get(item, 0) + len(chunk)
        return result

    async def _run(self, audio, tick):
        await self._ensure_transport()
        return await self._async_run_tick(audio, tick)

    def send_tool_result(self, call_id, result, request_response=True, is_error=False):
        if call_id not in self._calls:
            raise RuntimeError('unmatched Gemini tool result: ' + call_id)
        self._pending_tool_results.append((call_id, result, request_response, is_error))

    async def _flush_pending_tool_results(self):
        responses = []
        for cid, result, _, is_error in self._pending_tool_results:
            original, name = self._calls[cid]
            if original in self._cancelled_calls:
                self.audit('cancelled_tool_result_not_sent', call_id=cid)
                continue
            response = dict(name=name, response={('error' if is_error else 'output'): result})
            if original:
                response['id'] = original
            responses.append(response)
        if responses:
            await self._ws.send(json.dumps(dict(toolResponse=dict(functionResponses=responses))))
            self.audit('tool_results_sent', call_ids=[r.get('id') for r in responses], count=len(responses))
        self._pending_tool_results.clear()

    async def _execute_tick(self, user_audio, tick_number, result, tick_start):
        pcm = audioop.ulaw2lin(user_audio, 2)
        pcm, self._input_state = audioop.ratecv(pcm, 2, 1, 8000, 16000, self._input_state)
        async def send():
            for offset in range(0, len(pcm), 640):
                packet = pcm[offset:offset + 640]
                await self._ws.send(json.dumps(dict(realtimeInput=dict(audio=dict(
                    mimeType='audio/pcm;rate=16000', data=base64.b64encode(packet).decode())))))
                await asyncio.sleep(len(packet) / 32000)
        async def receive():
            deadline = tick_start + self.tick_duration_ms / 1000
            while True:
                while self._messages:
                    arrival, msg, size = self._messages.popleft()
                    self._queued_bytes -= size
                    self._process_message(result, msg, arrival, tick_start)
                if self._transport_error and not self._goaway:
                    raise RuntimeError(self._transport_error)
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return
                await asyncio.sleep(min(.005, remaining))
        await asyncio.gather(send(), receive())

    def _process_message(self, result, msg, arrival, tick_start):
        if 'error' in msg:
            raise RuntimeError('Gemini API: ' + str(msg['error']).replace(self.key, '[REDACTED]'))
        if 'toolCallCancellation' in msg:
            ids = msg['toolCallCancellation'].get('ids', [])
            self._cancelled_calls.update(ids)
            result.tool_calls[:] = [c for c in result.tool_calls if c.id not in ids]
            self.audit('native_tool_call_cancellation', call_ids=ids)
        sc = msg.get('serverContent', {})
        if sc.get('interrupted'):
            # No server audio_start_ms is provided. Keep only the locally played
            # prefix up to receipt time, clamped to this tick. Never rewind old ticks.
            keep = max(0, min(self.bytes_per_tick, int((arrival - tick_start) * 8000)))
            kept, dropped = [], 0
            for audio, item in result.agent_audio_chunks:
                prefix = audio[:keep]
                if prefix:
                    kept.append((prefix, item))
                keep -= len(prefix)
                dropped += len(audio) - len(prefix)
            result.agent_audio_chunks[:] = kept
            dropped += sum(len(b) for b, _ in self._buffered_agent_audio)
            self._buffered_agent_audio.clear()
            result.truncated_audio_bytes += dropped
            result.was_truncated = True
            result.vad_events.append('interrupted')
            result.events.append(BaseGeminiEvent(type='interruption'))
            self.audit('native_interruption', tick=result.tick_number, dropped_unplayed_bytes=dropped,
                       provider_history_truncate_sent=False, native_history_alignment_unverified=True)
            self._turn_active = False
            self._item = None
            self._output_state = None
        for part in sc.get('modelTurn', {}).get('parts', []):
            inline = part.get('inlineData')
            if inline and inline.get('data'):
                mime = inline.get('mimeType', '')
                if not mime.startswith('audio/pcm') or ('rate=' in mime and 'rate=24000' not in mime):
                    raise RuntimeError('unexpected Gemini output format: ' + mime)
                if self._item is None:
                    self._turn += 1
                    self._item = f'gemini-turn-{self._turn}'
                    result.events.append(BaseGeminiEvent(type='response.created'))
                self._turn_active = True
                raw = base64.b64decode(inline['data'], validate=True)
                if len(raw) % 2:
                    raise RuntimeError('odd-length Gemini PCM16 payload')
                converted, self._output_state = audioop.ratecv(raw, 2, 1, 24000, 8000, self._output_state)
                result.agent_audio_chunks.append((audioop.lin2ulaw(converted, 2), self._item))
                result.events.append(BaseGeminiEvent(type='audio.delta'))
        for fc in msg.get('toolCall', {}).get('functionCalls', []):
            original = fc.get('id') or ''
            if original and (original in self._seen_calls or original in self._cancelled_calls):
                continue
            cid = original or 'gemini-tool-' + uuid.uuid4().hex
            self._seen_calls.add(cid)
            self._calls[cid] = (original, fc['name'])
            result.tool_calls.append(ToolCall(id=cid, name=fc['name'], arguments=fc.get('args', {})))
            result.events.append(BaseGeminiEvent(type='function_call.done'))
            self._turn_active = True
            self.audit('native_tool_call', call_id=cid, name=fc['name'])
        if sc.get('turnComplete'):
            self._turn_active = False
            self._item = None
            result.events.append(BaseGeminiEvent(type='turn.complete'))
        if 'usageMetadata' in msg:
            self.audit('usage_metadata', usage=msg['usageMetadata'])


def make_gemini_logical_adapter(**kwargs):
    return GeminiLogicalAdapter(**kwargs)
