"""Native Seeduplex wire semantics, without network/model/judge calls."""
from pathlib import Path
import sys,unittest,asyncio,base64,json,time
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mubench.config import activate
activate(ROOT/'runtime/runners/tool_free')
from adapters.providers.seeduplex.adapter import SeeduplexLogicalAdapter,SeeduplexProtocolError
from tau2.voice.audio_native.tick_result import TickResult

def adapter():
 a=SeeduplexLogicalAdapter(key='test-secret',model='1.2.6.1',bridge=SimpleNamespace(),proxy=None)
 a.sent=[]
 async def send(msg):a.sent.append(msg);return 'event-test'
 a._send=send
 return a

def result():return TickResult(tick_number=1,audio_sent_bytes=1600,audio_sent_duration_ms=200,bytes_per_tick=1600,bytes_per_second=8000)
def deliver(a,r,msg,arrival=1.0):a._process_message(r,msg,arrival,1.0)

class Events(unittest.TestCase):
 def test_pin_and_secure_endpoint(self):
  with self.assertRaises(ValueError):SeeduplexLogicalAdapter(key='x',model='other',bridge=None,proxy=None)
  with self.assertRaises(ValueError):SeeduplexLogicalAdapter(key='x',model='1.2.6.1',bridge=None,realtime_url='ws://bad',proxy=None)
 def test_asr_onset_does_not_synthesize_cancel(self):
  a=adapter();r=result();deliver(a,r,{'type':'conversation.item.input_audio_transcription.started'})
  self.assertFalse(r.was_truncated);self.assertEqual(a.sent,[])
 def test_official_onset_clears_only_unplayed_prefix(self):
  a=adapter();a._played={'old':8000};r=result();r.agent_audio_chunks=[(b'x'*1600,'old')];a._buffered_agent_audio=[(b'y'*400,'old')]
  deliver(a,r,{'type':'conversation.item.input_audio_transcription.started'},1.05)
  self.assertEqual(len(r.agent_audio_data),400);self.assertEqual(a._played,{'old':8000});self.assertEqual(a.sent,[])
  self.assertTrue(a.audit_records[-1]['native_history_alignment_unverified']);self.assertFalse(a.audit_records[-1]['provider_history_truncate_sent'])
 def test_late_cancel_cannot_clear_new_response(self):
  a=adapter();a._response_id='new';a._turn_active=True;r=result();r.agent_audio_chunks=[(b'n'*1600,'new')]
  deliver(a,r,{'type':'response.canceled','response_id':'old'})
  self.assertEqual(len(r.agent_audio_data),1600);self.assertTrue(a._turn_active);self.assertFalse(r.was_truncated)
  deliver(a,r,{'type':'response.output_audio.delta','response_id':'old','delta':base64.b64encode(b'\0'*1920).decode()})
  self.assertEqual(len(r.agent_audio_data),1600)
 def test_call_deduplication_and_original_arguments(self):
  a=adapter();r=result();msg={'type':'response.function_call_arguments.done','items':[{'call_id':'one','name':'change','arguments':'{"caller_id":"p02","target":"p01"}'}]}
  deliver(a,r,msg);deliver(a,r,msg);self.assertEqual(len(r.tool_calls),1);self.assertEqual(r.tool_calls[0].arguments,{'caller_id':'p02','target':'p01'})
  changed=dict(msg,items=[dict(msg['items'][0],arguments='{"caller_id":"p01"}')])
  with self.assertRaises(SeeduplexProtocolError):deliver(a,r,changed)
 def test_usage_retains_original_billing_data(self):
  a=adapter();r=result();m={'type':'response.done','event_id':'u1','response':{'usage':{'input_tokens':8,'output_tokens':4,'input_token_details':{'audio_tokens':8},'output_token_details':{'audio_tokens':4}}}}
  deliver(a,r,m);deliver(a,r,m);self.assertEqual(len(a.get_usage_records()),1);self.assertEqual(a.get_usage_records()[0].input_audio_tokens,8)
 def test_errors_redact_credentials(self):
  a=adapter()
  with self.assertRaisesRegex(SeeduplexProtocolError,'REDACTED'):deliver(a,result(),{'type':'error','error':{'message':'test-secret'}})
 def test_auto_response_is_not_faked(self):
  a=adapter();a.configure_auto_response(False);self.assertEqual(a.sent,[]);self.assertFalse(a.audit_records[-1]['supported'])

class AsyncWire(unittest.IsolatedAsyncioTestCase):
 async def test_all_parallel_results_sent_together_exactly_once(self):
  a=adapter();r=result();deliver(a,r,{'type':'response.function_call_arguments.done','items':[{'call_id':c,'name':'lookup','arguments':'{}'} for c in ['a','b']]})
  a.send_tool_result('a','{"ok":true}');await a._flush_pending_tool_results();self.assertEqual(a.sent,[])
  a.send_tool_result('b','{"ok":false}');await a._flush_pending_tool_results();self.assertEqual(len(a.sent),1);self.assertEqual(a.sent[0]['type'],'conversation.item.create');self.assertEqual([x['call_id'] for x in a.sent[0]['items']],['a','b']);self.assertTrue(all(x['role']=='tool' for x in a.sent[0]['items']))
  a.send_tool_result('a','{"ok":true}');await a._flush_pending_tool_results();self.assertEqual(len(a.sent),1)
 async def test_pause_only_mutes_and_never_cancels_or_replays(self):
  a=adapter();a._paused=('planning',200,time.monotonic());await a._begin_pause_signal();await asyncio.sleep(1.03)
  self.assertEqual([x['type'] for x in a.sent],['input_audio_mute.commit']);self.assertEqual(a._paused[1],200)
  await a._end_pause_signal();self.assertEqual([x['type'] for x in a.sent],['input_audio_mute.commit','input_audio_unmute.commit'])
 async def test_short_pause_does_not_modify_wire_state(self):
  a=adapter();a._paused=('planning',200,time.monotonic());await a._begin_pause_signal();await a._end_pause_signal();self.assertEqual(a.sent,[])
 async def test_played_output_capped_and_input_paced(self):
  a=adapter();data=base64.b64encode(b'\0'*19200).decode();msg={'type':'response.output_audio.delta','response_id':'r','delta':data};a._messages.append((time.monotonic(),msg,len(data)));a._queued_bytes=len(data)
  start=time.monotonic();r=await a._async_run_tick(b'\xff'*1600,1);elapsed=time.monotonic()-start
  self.assertLessEqual(r.agent_audio_bytes,1600);self.assertTrue(a._buffered_agent_audio);self.assertGreaterEqual(elapsed,.18);self.assertTrue(all(x['type']=='input_audio_buffer.append' for x in a.sent));self.assertTrue(all(0<len(base64.b64decode(x['audio']))<=640 for x in a.sent));self.assertEqual(r.proportional_transcript,'')
 async def test_background_receive_queues_during_pause(self):
  a=adapter();a._paused=('planning',200,time.monotonic());a._closed_ack=asyncio.Event()
  class WS:
   def __aiter__(self):return self
   async def __anext__(self):
    if getattr(self,'done',False):raise StopAsyncIteration
    self.done=True;return json.dumps({'type':'response.output_text.delta','delta':'not played'})
  a._ws=WS();await a._receive();self.assertEqual(len(a._messages),1);self.assertEqual(a._played,{});self.assertTrue(a.audit_records[-1]['paused'])

if __name__=='__main__':unittest.main()
