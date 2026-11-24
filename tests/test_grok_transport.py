"""Offline regressions for the real Grok transport and shared continuation state."""
import asyncio
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import AsyncMock
from types import SimpleNamespace
ROOT = Path(os.environ.get('MUBENCH_TEST_ROOT', Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT))
with contextlib.redirect_stdout(sys.stderr):
 from mubench.config import activate,profile
 activate(profile('xai', 'tool_free'))
 from tau2.data_model.audio import TELEPHONY_AUDIO_FORMAT
 from tau2.voice.audio_native.tick_result import TickResult
 from websockets.protocol import State
path = Path(os.environ.get('MUBENCH_TEST_GROK_ADAPTER', ROOT/'adapters/providers/grok/adapter.py'))
spec = importlib.util.spec_from_file_location('grok_transport_subject', path)
subject = importlib.util.module_from_spec(spec)
spec.loader.exec_module(subject)

class Wire:
 def __init__(self):
  self.state=State.OPEN;self.messages=asyncio.Queue();self.sent=[]
 async def recv(self):return await self.messages.get()
 async def send(self,message):self.sent.append(json.loads(message))
 async def close(self):self.state=State.CLOSED
 def feed(self,raw):self.messages.put_nowait(json.dumps(raw))

class GrokTransportTests(unittest.IsolatedAsyncioTestCase):
 def provider(self):
  p=subject.GrokLogicalProvider(realtime_url='wss://api.x.ai/v1/realtime',realtime_key='offline-fixture-secret',model='grok-voice-3.0');p.ws=Wire();return p
 def adapter(self):
  p=self.provider();a=subject.LogicalTimeGrokAdapter(tick_duration_ms=200,provider=p,audio_format=TELEPHONY_AUDIO_FORMAT);a.model='grok-voice-3.0';return a
 async def event(self,a,raw):
  result=TickResult(tick_number=0,audio_sent_bytes=1600,audio_sent_duration_ms=200,bytes_per_tick=1600,bytes_per_second=8000)
  await a._process_event(result,subject.normalize_grok_event(raw));return result
 async def ack(self,a,cid):
  await self.event(a,{'type':'conversation.item.added','item':{'id':'item-'+cid,'type':'function_call_output','call_id':cid,'output':'{}'}})
 async def done(self,a,rid):await self.event(a,{'type':'response.done','response':{'id':rid,'status':'completed'}})
 async def created(self,a,rid):await self.event(a,{'type':'response.created','response':{'id':rid,'status':'in_progress'}})
 async def output(self,a,ids):
  a._pending_tool_results.extend((cid,'{}',True,False) for cid in ids);await a._flush_pending_tool_results();a._continuation_not_before=0

 async def test_pinned_model_3_and_no_alias_fallback(self):
  a=subject.make_grok_logical_adapter(realtime_url='wss://api.x.ai/v1/realtime',realtime_key='offline-fixture-secret',model='grok-voice-3.0',tick_duration_ms=200,audio_format=TELEPHONY_AUDIO_FORMAT,bridge=None)
  self.assertIn('model=grok-voice-3.0',a.provider._url)
  with self.assertRaises(subject.GrokProtocolError):
   subject.make_grok_logical_adapter(realtime_url='wss://api.x.ai/v1/realtime',realtime_key='offline-fixture-secret',model='grok-voice-latest',tick_duration_ms=200,audio_format=TELEPHONY_AUDIO_FORMAT,bridge=None)

 async def test_outputs_once_wait_for_ack_and_active_response_then_coalesce(self):
  a=self.adapter();await self.created(a,'old');await self.output(a,['one','two']);await self.ack(a,'one');await a._maybe_request_continuation();self.assertFalse(any(x['type']=='response.create' for x in a.provider.ws.sent));await self.ack(a,'two');await a._maybe_request_continuation();self.assertFalse(any(x['type']=='response.create' for x in a.provider.ws.sent));await self.done(a,'old');await a._maybe_request_continuation();await a._maybe_request_continuation()
  self.assertEqual([x['type'] for x in a.provider.ws.sent],['conversation.item.create','conversation.item.create','response.create'])
  self.assertNotIn('response',a.provider.ws.sent[-1])
  await self.created(a,'new');self.assertFalse(a._continuation_inflight);self.assertEqual(a._continuation_covered_epoch,2)
  await self.done(a,'new');await a._maybe_request_continuation();self.assertEqual(len(a.provider.ws.sent),3)

 async def test_server_auto_response_covers_committed_result(self):
  a=self.adapter();await self.output(a,['one']);await self.ack(a,'one');await self.created(a,'auto');await self.done(a,'auto');await a._maybe_request_continuation();self.assertFalse(any(x['type']=='response.create' for x in a.provider.ws.sent))

 async def test_identified_collision_deferred_without_replaying_output(self):
  a=self.adapter();await self.output(a,['one']);await self.ack(a,'one');await a._maybe_request_continuation();eid=a.provider.ws.sent[-1]['event_id']
  await self.event(a,{'type':'error','error':{'type':'invalid_request_error','code':'conversation_already_has_active_response','event_id':eid,'message':'Conversation already has an active response in progress: racing_response. Wait.'}})
  await a._maybe_request_continuation();self.assertEqual(sum(x['type']=='response.create' for x in a.provider.ws.sent),1)
  await self.done(a,'racing_response');await a._maybe_request_continuation();self.assertEqual(sum(x['type']=='response.create' for x in a.provider.ws.sent),2);self.assertEqual(sum(x['type']=='conversation.item.create' for x in a.provider.ws.sent),1)

 async def test_unrelated_provider_error_stays_fatal(self):
  a=self.adapter()
  with self.assertRaises(RuntimeError):await self.event(a,{'type':'error','error':{'code':'insufficient_quota','message':'fixture quota error'}})

 async def test_receiver_queues_during_pause_without_media_or_tool_execution(self):
  a=self.adapter();p=a.provider;a._paused=True;p._reader_task=asyncio.create_task(p._receiver_loop())
  for i in range(80):p.ws.feed({'type':'response.output_audio.delta','response_id':'r','item_id':'i','delta':'AAAA','event_id':str(i)})
  p.ws.feed({'type':'response.function_call_arguments.done','call_id':'call1','name':'get_reservation','arguments':'{}'})
  for _ in range(100):
   if p._incoming.qsize()==81:break
   await asyncio.sleep(.002)
  self.assertEqual(p._incoming.qsize(),81);self.assertFalse(a.is_output_drained());self.assertEqual(a._tick_count,0);self.assertFalse(p.ws.sent);self.assertFalse(a._pending_tool_results)
  deadline=asyncio.get_running_loop().time()+2
  for _ in range(81):await p.receive_event_until(deadline)
  self.assertFalse(p.has_pending_events);self.assertEqual(p._queued_wire_bytes,0);await p.disconnect()

 async def test_queue_overflow_is_technical_error_not_silent_loss(self):
  p=self.provider();p.max_queued_wire_bytes=1;p._reader_task=asyncio.create_task(p._receiver_loop());p.ws.feed({'type':'response.created','response':{'id':'x'}})
  with self.assertRaisesRegex(subject.GrokProtocolError,'queue exceeded'):
   await p.receive_event_until(asyncio.get_running_loop().time()+2)
  await p.disconnect()

 async def test_wire_error_preserves_correlation_and_redacts_secret(self):
  p=self.provider();p.ws.feed({'type':'error','error':{'code':'test','event_id':'mu_continuation_1','message':'fixture '+p.api_key}})
  raw,_=await p._read_wire();self.assertEqual(raw['error']['event_id'],'mu_continuation_1');self.assertNotIn(p.api_key,json.dumps(raw));self.assertIn('[REDACTED]',raw['error']['message']);self.assertEqual(subject.normalize_grok_event(raw).related_event_id,'mu_continuation_1')

 async def test_public_function_schema_registered_unchanged(self):
  p=self.provider();schema={'type':'object','properties':{'caller_id':{'type':'string'}},'required':['caller_id']};tool=SimpleNamespace(openai_schema={'function':{'name':'get_reservation','description':'fixture contract','parameters':schema}})
  p.ws.feed({'type':'session.updated','session':{'turn_detection':{'type':'server_vad'},'audio':{'input':{'format':{'type':'audio/pcmu'}},'output':{'format':{'type':'audio/pcmu'}}}}})
  await p.configure_session(system_prompt='fixture protocol',tools=[tool],audio_format=TELEPHONY_AUDIO_FORMAT)
  session=p.ws.sent[0]['session'];self.assertEqual(session['tools'],[{'type':'function','name':'get_reservation','description':'fixture contract','parameters':schema}]);self.assertEqual(session['turn_detection'],{'type':'server_vad'});self.assertNotIn('transcription_keyterms',session)

 async def test_registration_flags_are_not_claimed_or_sent(self):
  a=self.adapter();await a._async_configure_auto_response(False,1);self.assertFalse(a.provider.ws.sent);self.assertEqual(subject.grok_turn_detection(False),{'type':'server_vad'});self.assertFalse(a.audit_records[-1]['supported'])

 async def truncate(self,a,item='old-item',end=200):
  await a._request_item_truncate(item_id=item,content_index=0,audio_end_ms=end,prior_item_played_bytes=end*8,current_tick_item_played_bytes=0)
 async def truncated(self,a,item='old-item',end=200,index=0):
  await self.event(a,{'type':'conversation.item.truncated','item_id':item,'content_index':index,'audio_end_ms':end})
 async def test_pending_and_confirmed_duplicates_do_not_add_waits(self):
  a=self.adapter();await self.truncate(a);await self.truncate(a)
  self.assertEqual(a.pending_truncate_acks,{'old-item':[200]});self.assertEqual(len(a.provider.ws.sent),1)
  await self.truncated(a);await self.truncate(a);self.assertFalse(a.pending_truncate_acks);self.assertEqual(len(a.provider.ws.sent),1)
  self.assertEqual([r['prior_state'] for r in a.audit_records if r['kind']=='grok_duplicate_truncate_skipped'],['pending','confirmed'])
 async def test_late_ack_after_new_response_remains_matchable(self):
  a=self.adapter();await self.truncate(a);await self.created(a,'new');self.assertEqual(a.pending_truncate_acks,{'old-item':[200]})
  await self.truncated(a);self.assertFalse(a.pending_truncate_acks);self.assertTrue(next(r for r in a.audit_records if r['kind']=='truncate_ack')['matched'])
  self.assertFalse(any(r['kind']=='grok_stale_truncate_acks_cleared' for r in a.audit_records))
 async def test_ack_wrong_content_index_does_not_clear_wait(self):
  a=self.adapter();await self.truncate(a);await self.truncated(a,index=1);self.assertTrue(a.pending_truncate_acks);await self.truncated(a);self.assertFalse(a.pending_truncate_acks)
 async def test_duplicate_ack_does_not_create_or_consume_other_wait(self):
  a=self.adapter();await self.truncate(a);await self.truncated(a);await self.truncate(a,end=100);await self.truncated(a);self.assertEqual(a.pending_truncate_acks,{'old-item':[100]});await self.truncated(a,end=100);self.assertFalse(a.pending_truncate_acks)
 async def test_missing_ack_has_finite_media_timeout_without_forged_success(self):
  a=self.adapter();await self.truncate(a);a._current_logical_time_ms=9999;a._check_truncate_ack_timeout();a._current_logical_time_ms=10000
  with self.assertRaisesRegex(subject.GrokProtocolError,'10s of media time'):a._check_truncate_ack_timeout()
  self.assertTrue(a.pending_truncate_acks);self.assertFalse(a._grok_confirmed_truncates)
 async def test_actual_barge_in_does_not_retruncate_fully_heard_old_item(self):
  a=self.adapter();a._last_played_item_id='old-item';a._played_bytes_by_item['old-item']=1600
  await self.event(a,{'type':'input_audio_buffer.speech_started','audio_start_ms':0});await self.truncated(a)
  await self.event(a,{'type':'input_audio_buffer.speech_started','audio_start_ms':0})
  self.assertEqual(sum(r['type']=='conversation.item.truncate' for r in a.provider.ws.sent),1);self.assertFalse(a.pending_truncate_acks)
 async def test_ledgers_are_per_adapter_session(self):
  a=self.adapter();await self.truncate(a);await self.truncated(a);b=self.adapter();await self.truncate(b);self.assertEqual(len(b.provider.ws.sent),1);self.assertTrue(b.pending_truncate_acks)

 async def test_user_barge_in_closes_only_active_response_and_suppresses_late_audio(self):
  a=self.adapter();await self.created(a,'old');await self.event(a,{'type':'input_audio_buffer.speech_started','audio_start_ms':0,'item_id':'u'});self.assertNotIn('old',a._active_response_ids);self.assertIn('old',a._suppressed_response_ids)
  result=await self.event(a,{'type':'response.output_audio.delta','response_id':'old','item_id':'late','delta':'AAAA'});self.assertFalse(result.agent_audio_chunks);self.assertTrue(any(x['type']=='conversation.item.delete' for x in a.provider.ws.sent));self.assertFalse(any(x['type']=='response.cancel' for x in a.provider.ws.sent))

if __name__=='__main__':unittest.main()
