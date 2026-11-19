"""Audio registration and native MOSS function wire contract."""
from pathlib import Path
import sys,unittest,asyncio
from collections import deque
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mubench.config import activate
activate(ROOT/'runtime/runners/tool_augmented_audio')
from adapters.providers.moss.tools import MossToolAdapter
from experiments.tau_multiuser_voice.spoken_protocol import with_spoken_session_rules,ATTRIBUTION_RULES
from experiments.tau_multiuser_voice.cases import CaseCatalog
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from run_tau_mu_voice_realtime import build_system_prompt,required_tool_names

class PublicEntryTests(unittest.TestCase):
 def test_imports(self):
  import experiments.mu_bench.logical_time.openai_adapter
  import experiments.mu_bench.logical_time.grok_adapter
  import experiments.mu_bench.logical_time.gemini_adapter
 def test_session_rules_only_in_speech(self):
  case=with_spoken_session_rules(CaseCatalog.load_default()[0]);env=create_virtual_environment(case.environment)
  prompt=build_system_prompt(case,env,None,required_tool_names(case))
  self.assertIn(ATTRIBUTION_RULES,case.voice_enrollment.session_preamble_template)
  self.assertNotIn('Session registration protocol:',prompt)
  self.assertNotIn('Attribution protocol (required',prompt)
  self.assertNotIn('The opening audio contains',prompt)
  self.assertNotIn('You are Nova',prompt)
 def test_no_grok_keyterms(self):
  import inspect
  from adapters.providers.grok.adapter import GrokLogicalProvider
  self.assertNotIn('transcription_keyterms',inspect.getsource(GrokLogicalProvider.configure_session))

class WireTests(unittest.IsolatedAsyncioTestCase):
 def adapter(self):
  a=MossToolAdapter.__new__(MossToolAdapter);a._calls=deque();a._call_records={};a._result_acks={};a._results={};a._cancelled=set();a._continuations={};a.audit=[];a._fatal=None;a._audit=lambda k,**v:a.audit.append(dict(kind=k,**v));return a
 async def test_call_id_deduplicates_summary(self):
  a=self.adapter();c=dict(call_id='call-1',name='get_booking',arguments='{"caller_id":"p01"}')
  await a._on_protocol_event('response.function_call_arguments.done',c)
  await a._on_protocol_event('response.done',{'response':{'output':[dict(c,type='function_call')]}})
  self.assertEqual(len(a._calls),1)
 async def test_continuation_waits_for_exact_ack(self):
  a=self.adapter();sent=[]
  async def send(action,**kw):sent.append((action,kw))
  a._send=send;t=asyncio.create_task(a._submit_result('call-1','{"recorded":true}',True));await asyncio.sleep(0)
  await a._on_protocol_event('conversation.item.created',{'item':{'type':'function_call_output','call_id':'other'}});await asyncio.sleep(0)
  self.assertEqual([x[0] for x in sent],['conversation.item.create'])
  await a._on_protocol_event('conversation.item.created',{'item':{'type':'function_call_output','call_id':'call-1','id':'out-1'}});await t
  self.assertEqual(sent[-1],('response.create',{'response':{'call_ids':['call-1']}}))
 async def test_cancelled_pending_result_is_not_sent(self):
  a=self.adapter();sent=[]
  async def send(action,**kw):sent.append(action)
  a._send=send;await a._on_protocol_event('response.function_call.cancel_requested',{'call_id':'call-1','reason':'cancelled'})
  await a._submit_result('call-1','{}',True);self.assertEqual(sent,[])

if __name__=='__main__':unittest.main()
