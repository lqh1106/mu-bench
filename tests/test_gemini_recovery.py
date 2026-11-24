"""Offline regressions for Gemini recovery boundaries and input-gap handling."""
import asyncio
from collections import deque
import importlib.util
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import AsyncMock, patch
import sys

ROOT = Path(os.environ.get('MUBENCH_TEST_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
from mubench.config import activate, profile
activate(profile('gemini', 'tool_free'))
path = Path(os.environ.get('MUBENCH_TEST_GEMINI_ADAPTER', ROOT / 'adapters/providers/gemini/adapter.py'))
spec = importlib.util.spec_from_file_location('gemini_recovery_subject', path)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)

class Wire:
    def __init__(self, updates=()):
        self.updates = updates
        self.send = AsyncMock()
        self.close = AsyncMock()
    def __aiter__(self):
        async def messages():
            for update in self.updates:
                yield json.dumps(update)
        return messages()

class GeminiRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def probe(self):
        p = subject.GeminiLogicalAdapter(key='offline-fixture', model='gemini-3.8-live', bridge=None, proxy=None)
        p._ws = Wire()
        p._connect_transport = AsyncMock()
        p._goaway = True
        p._handle = 'checkpoint-A'
        p._resumable = True
        p._deadline = time.monotonic() + 30
        return p

    async def test_non_resumable_update_disables_cached_handle(self):
        p = self.probe()
        p._ws = Wire([{'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'A'}},
                      {'sessionResumptionUpdate': {'resumable': False, 'newHandle': ''}}])
        await p._receive()
        self.assertEqual(p._handle, 'A')
        self.assertFalse(p._resumable)
        with self.assertRaisesRegex(RuntimeError, 'no safe native resume boundary'):
            await p._ensure_transport()
        p._connect_transport.assert_not_awaited()
        p._ws.close.assert_not_awaited()

    async def test_unsafe_boundary_waits_before_deadline(self):
        p = self.probe(); p._resumable = False
        await p._ensure_transport()
        p._connect_transport.assert_not_awaited()
        p._ws.close.assert_not_awaited()

    async def test_each_unsettled_boundary_blocks_forced_reconnect(self):
        mutations = [lambda p: setattr(p, '_wire_turn_active', True),
                     lambda p: setattr(p, '_turn_active', True),
                     lambda p: p._unanswered_calls.add('call-1'),
                     lambda p: p._pending_tool_results.append(('call-1', {}, True, False)),
                     lambda p: p._buffered_agent_audio.append((b'abc', 'turn-1')),
                     lambda p: p._messages.append((0, {'toolCallCancellation': {'ids': ['call-1']}}, 1))]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                p = self.probe(); mutate(p); p._deadline = time.monotonic() - 1
                with self.assertRaisesRegex(RuntimeError, 'technical interruption'):
                    await p._ensure_transport()
                p._connect_transport.assert_not_awaited()

    async def test_safe_boundary_uses_native_resume_once_without_replay(self):
        p = self.probe()
        await p._ensure_transport()
        p._ws.close.assert_awaited_once()
        p._connect_transport.assert_awaited_once_with(resume=True)
        p._ws.send.assert_not_awaited()
        self.assertEqual(p._resumptions, 1)

    async def test_later_valid_handle_reenables_recovery(self):
        p = self.probe()
        p._ws = Wire([{'sessionResumptionUpdate': {'resumable': False}},
                      {'sessionResumptionUpdate': {'resumable': True, 'newHandle': 'B'}}])
        await p._receive()
        self.assertTrue(p._resumable)
        self.assertEqual(p._handle, 'B')
        await p._ensure_transport()
        p._connect_transport.assert_awaited_once_with(resume=True)

    async def test_unplanned_disconnect_is_technical_not_empty_session(self):
        p = self.probe(); p._goaway = False; p._transport_error = 'fixture disconnect'
        with self.assertRaisesRegex(RuntimeError, 'fixture disconnect'):
            await p._ensure_transport()
        p._connect_transport.assert_not_awaited()

    async def test_stream_end_repeats_every_30s_without_media_or_clock_advance(self):
        p = self.probe(); p._paused = ('simulator', 1000, time.monotonic())
        intervals = []
        async def advance(seconds):
            intervals.append(seconds)
            if len(intervals) == 4:
                p._paused = None
        with patch.object(subject.asyncio, 'sleep', side_effect=advance):
            await p._signal_long_audio_gap()
        self.assertEqual(intervals, [1.0, 30, 30, 30])
        self.assertEqual(p._ws.send.await_count, 3)
        for call in p._ws.send.call_args_list:
            self.assertEqual(json.loads(call.args[0]), {'realtimeInput': {'audioStreamEnd': True}})
        self.assertEqual(p._tick_count, 0)
        self.assertFalse(p._played)
        self.assertTrue(all(r['audio_bytes_sent'] == 0 and not r['logical_clock_advanced']
                            for r in p.audit_records if r['kind'] == 'pause_audio_stream_end'))

    async def test_no_pause_signal_after_resume(self):
        p = self.probe(); p._paused = None
        with patch.object(subject.asyncio, 'sleep', new=AsyncMock()):
            await p._signal_long_audio_gap()
        p._ws.send.assert_not_awaited()

    async def test_tool_response_and_cancellation_settle_only_matching_calls(self):
        p = self.probe(); p._calls = {'a': ('a', 'lookup'), 'b': ('b', 'lookup')}
        p._unanswered_calls.update(['a', 'b']); p._cancelled_calls.add('b')
        p._pending_tool_results = [('a', {'ok': True}, True, False), ('b', {}, True, False)]
        await p._flush_pending_tool_results()
        response = json.loads(p._ws.send.call_args.args[0])['toolResponse']['functionResponses']
        self.assertEqual(response, [{'name': 'lookup', 'id': 'a', 'response': {'output': {'ok': True}}}])
        self.assertFalse(p._unanswered_calls)
        self.assertFalse(p._pending_tool_results)

if __name__ == '__main__':
    unittest.main()
