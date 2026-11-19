"""MOSS native front-end functions, with unchanged PCM/playback/session semantics.

Use custom tools directly; no client delegation or external reasoning agent.
Arguments are executed only by the benchmark's existing tool/authorization engine.
"""
from __future__ import annotations
import asyncio
from collections import deque
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

from adapters.providers.moss.reconnect import MossLiveLogicalAdapter
from adapters.providers.moss.base import MossProtocolError


class MossToolAdapter(MossLiveLogicalAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configuration = {}
        self._calls = deque()
        self._call_records = {}
        self._result_acks = {}
        self._results = {}
        self._cancelled = set()
        self._continuations = {}
        self.tool_effects = {}

    def _audit(self, kind, **kwargs):
        if kind == 'moss_session_started':
            kwargs['custom_prompt'] = bool(getattr(self, '_configuration', {}).get('instructions'))
            kwargs['custom_prompt_scope'] = 'tool_contract_and_authored_initial_context'
            kwargs['session_rules_in_system_text'] = False
        return super()._audit(kind, **kwargs)

    def _session_config(self):
        return deepcopy(self._configuration)

    def connect(self, system_prompt, tools, modality='audio'):
        if modality != 'audio' or self.is_connected:
            raise MossProtocolError('MOSS requires one fresh audio session')
        declarations = []
        for tool in tools:
            schema = deepcopy(tool.openai_schema)
            function = schema.get('function', schema)
            name = function['name']
            read_only = name == 'report_speaker_attribution' or self.tool_effects.get(name) == 'read'
            declarations.append(dict(type='function', name=name,
                description=function.get('description') or name,
                parameters=function.get('parameters', {'type': 'object', 'properties': {}}),
                execution_mode='async_query' if read_only else 'async_effect', read_only=read_only))
        if len({d['name'] for d in declarations}) != len(declarations):
            raise MossProtocolError('Duplicate registered function name')
        self._configuration = dict(tools=declarations)
        if system_prompt:
            self._configuration['instructions'] = system_prompt
        self._audit('moss_tools_configuration', tools=[d['name'] for d in declarations],
            native_custom_functions=True, delegation=False,
            asynchronous_wait_reason='simulator wall pauses must not time out a synchronous function',
            session_rules_transport='spoken_tts_registration')
        self._call(self._connect())
        self.is_connected = True

    async def _on_session_configured(self, data):
        expected = {t['name'] for t in self._configuration.get('tools', [])}
        actual = set(data.get('tools') or [])
        if actual != expected:
            raise MossProtocolError('Server did not acknowledge the exact custom tool set')
        self._audit('moss_tools_configuration_acknowledged', tools=sorted(actual))

    def configure_auto_response(self, enabled, **kwargs):
        # The public MOSS API has no create_response boolean. Never claim an
        # acknowledgement or silence that the service did not provide.
        self._audit('moss_registration_response_policy', requested_enabled=bool(enabled),
            provider_action='none', actual_policy='native_audio_with_spoken_registration_rules',
            server_auto_response_disabled=False)

    def _accept_call(self, data):
        call_id, name = data.get('call_id'), data.get('name')
        if not isinstance(call_id, str) or not call_id or not isinstance(name, str):
            raise MossProtocolError('Function event lacks call_id/name')
        arguments = data.get('arguments', '{}')
        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
        if not isinstance(arguments, dict):
            raise MossProtocolError('Function arguments must be an object')
        digest = hashlib.sha256(json.dumps(dict(name=name, arguments=arguments),
                              sort_keys=True).encode()).hexdigest()
        old = self._call_records.get(call_id)
        if old:
            if old['sha256'] != digest:
                raise MossProtocolError('Repeated call_id changed arguments')
            self._audit('moss_tool_duplicate_ignored', call_id=call_id)
            return
        self._call_records[call_id] = dict(sha256=digest, name=name, task_id=data.get('task_id'),
                                          revision=data.get('revision'))
        self._calls.append(SimpleNamespace(id=call_id, name=name, arguments=arguments))
        self._audit('moss_tool_call_received', call_id=call_id, name=name,
            arguments=arguments, execution_mode=data.get('execution_mode'),
            task_id=data.get('task_id'), revision=data.get('revision'))

    async def _on_protocol_event(self, action, data):
        if action == 'response.function_call_arguments.done':
            self._accept_call(data)
        elif action == 'response.done':
            for call in (data.get('response') or {}).get('output', []):
                if call.get('type') == 'function_call':
                    self._accept_call(call)
        elif action == 'conversation.item.created':
            item = data.get('item') or {}
            call_id = item.get('call_id')
            if item.get('type') == 'function_call_output' and call_id in self._result_acks:
                self._result_acks[call_id].set()
                self._audit('moss_tool_result_acknowledged', call_id=call_id, item_id=item.get('id'))
        elif action == 'response.function_call.cancel_requested':
            call_id = data.get('call_id')
            self._cancelled.add(call_id)
            self._audit('moss_tool_cancel_requested', call_id=call_id,
                reason=data.get('reason'), result_already_submitted=call_id in self._results,
                business_effects_automatically_rolled_back=False)
        elif action == 'response.interrupted':
            self._active.discard((data.get('run_id'), data.get('turn_id')))

    def run_tick(self, user_audio, tick_number):
        result = super().run_tick(user_audio, tick_number)
        calls = []
        while self._calls:
            call = self._calls.popleft()
            if call.id in self._cancelled:
                self._audit('moss_cancelled_pending_call_not_executed', call_id=call.id)
            else:
                calls.append(call)
        result.tool_calls = calls
        return result

    async def _submit_result(self, call_id, result, request_response):
        event = asyncio.Event()
        self._result_acks[call_id] = event
        try:
            if call_id in self._cancelled:
                self._audit('moss_late_tool_result_discarded', call_id=call_id)
                return
            await self._send('conversation.item.create', item=dict(
                type='function_call_output', call_id=call_id, output=result))
            self._audit('moss_tool_result_submitted', call_id=call_id)
            await asyncio.wait_for(event.wait(), 30)
            if request_response and call_id not in self._cancelled:
                await self._send('response.create', response=dict(call_ids=[call_id]))
                self._audit('moss_tool_continuation_requested', call_id=call_id,
                            after_matching_result_ack=True)
        except Exception as exc:
            self._fatal = 'Tool result transport failed: ' + type(exc).__name__
            self._audit('moss_tool_result_transport_failed', call_id=call_id,
                        error_type=type(exc).__name__)
        finally:
            self._continuations.pop(call_id, None)

    async def _queue_result(self, call_id, result, request_response):
        old = self._results.get(call_id)
        if old is not None:
            if old != result:
                raise MossProtocolError('One call_id cannot receive different results')
            self._audit('moss_duplicate_result_not_replayed', call_id=call_id)
            return
        self._results[call_id] = result
        self._continuations[call_id] = asyncio.create_task(
            self._submit_result(call_id, result, request_response))

    def send_tool_result(self, call_id, result, request_response=True, is_error=False):
        if call_id not in self._call_records:
            raise MossProtocolError('Cannot submit a result for an unknown call_id')
        # Return promptly: the receiver keeps draining media and confirms the
        # output asynchronously. Completion waits for this receipt/continuation.
        self._call(self._queue_result(call_id, result, request_response))

    def is_output_drained(self):
        return super().is_output_drained() and not self._calls and not self._continuations

    async def _disconnect(self):
        for task in list(self._continuations.values()):
            task.cancel()
        await asyncio.gather(*self._continuations.values(), return_exceptions=True)
        self._continuations.clear()
        await super()._disconnect()
