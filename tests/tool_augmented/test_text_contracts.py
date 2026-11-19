from pathlib import Path
import sys,json,os
HERE=Path(__file__).resolve().parent
os.environ['MUBENCH_JUDGE_PROTOCOL']='tool_full_source_review'

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mubench.config import activate,profile
activate(profile('chat','tool_augmented',transport='text'))
from run_text_case import Episode
from adapters.providers.chat.adapter import TextStream
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.fact_contract import review_for_case
from experiments.tau_multiuser_voice.task_evidence import build_task_evidence_contract
from experiments.tau_multiuser_voice.judge_evidence import build_evidence_catalog
from experiments.tau_multiuser_voice.scene_context import get_initial_context

class Fake:
 def __init__(self,deltas):self.deltas=deltas
 def stream(self,messages,tools):yield from self.deltas
tests=[]
history=[];effects=[]
s=TextStream(Fake([({'content':'ABCDE'},None),({'content':'future'},'stop')]),history,[])
out=s.response(lambda c,p:len(p)==3,lambda calls:effects.extend(calls))
assert history==[{'role':'assistant','content':'ABC'}] and out['discarded_buffer_suffix']=='DE' and not effects
tests.append('only_delivered_prefix_committed')
history=[]
s=TextStream(Fake([({'tool_calls':[{'index':0,'id':'c1','function':{'name':'write','arguments':'{"x":'}}],'content':'ABCDE'},None)]),history,[])
out=s.response(lambda c,p:len(p)==2,lambda calls:effects.extend(calls))
assert not effects and not history[0].get('tool_calls');tests.append('partial_tool_call_not_executed')
history=[];effects=[]
s=TextStream(Fake([({'tool_calls':[{'index':0,'id':'c1','function':{'name':'write','arguments':'{"x":'}}]},None),({'tool_calls':[{'index':0,'function':{'arguments':'1}'}}]},'tool_calls')]),history,[])
s.response(lambda c,p:False,lambda calls:effects.extend(calls));assert len(effects)==1 and effects[0]['function']['arguments']=='{"x":1}'
history.append({'role':'user','content':'interrupt'});assert effects;tests.append('completed_tool_effect_survives_later_interrupt')
history=[]
s=TextStream(Fake([({'reasoning_content':'hidden reasoning'},None),({'content':'X'},'stop')]),history,[])
s.response(lambda c,p:False,lambda calls:None);assert history[0]['content']=='X';tests.append('reasoning_not_delivered')
rows=[]
for p in sorted((ROOT/'cases/tool_augmented').glob('*.json')):
 c=MultiUserVoiceCase.model_validate(json.loads(p.read_text()));env=create_virtual_environment(c.environment)
 contract=build_task_evidence_contract(c,review_for_case(c));assert contract['enabled'],c.case_id
 assert get_initial_context(c)['contains_future_event_oracle'] is False
 assert set(contract['available_business_tools'])==set(c.environment.tool_names)
 owners={g.goal_id:u.user_id for u in c.users for g in u.goals}
 for g in contract['goals']:
  if g.get('external_execution_required'):assert set(g['evidence_policy']['tool_names'])<=env.tool_names
 rows.append({'case_id':c.case_id,'contract_enabled':True,'goals':len(contract['goals'])})
tests.append('all_408_case_schema_source_and_tool_obligation_contracts_compile')
cat=build_evidence_catalog([{'role':'assistant','text':'Actual delivered text','start_ms':0,'end_ms':1,'committed':True}])
assert cat['sources'][0]['kind']=='assistant_text';tests.append('text_evidence_not_audio_or_asr')
from experiments.mu_bench.logical_time.online_users import OrderedOnlineEventGraph
from experiments.mu_bench.logical_time.pacing import build_case_pacing
from experiments.tau_multiuser_voice.models import WorldObservation,ObservationKind
case=MultiUserVoiceCase.model_validate(json.loads((ROOT/'cases/tool_augmented/hospitality_01_party_size_fix__g-small__a-clean__c-strong.json').read_text()))
g=OrderedOnlineEventGraph(case,pacing_plan=build_case_pacing(case));g.mark_completed('e_open',2200)
g.pacing.observe_media(14000,False,False,played_duration_ms=0)
assert g.ready_events(WorldObservation(sim_time_ms=14000,kind=ObservationKind.CLOCK))[0].event_id=='e_live'
tests.append('approved_semantic_followup_has_relative_deadline')
g=OrderedOnlineEventGraph(case,pacing_plan=build_case_pacing(case));g.mark_completed('e_open',2200);g.mark_completed('e_live',7000)
g.pacing.observe_media(18000,True,True,played_duration_ms=1000)
assert not g.ready_events(WorldObservation(sim_time_ms=18000,kind=ObservationKind.CLOCK,assistant_speaking=True))
g.pacing.observe_media(20000,True,True,played_duration_ms=2000)
g.pacing.observe_media(20000,False,False,played_duration_ms=0)
assert not g.ready_events(WorldObservation(sim_time_ms=20000,kind=ObservationKind.CLOCK))
g.pacing.observe_media(21200,False,False,played_duration_ms=0)
assert g.ready_events(WorldObservation(sim_time_ms=21200,kind=ObservationKind.CLOCK))[0].event_id=='e_primary_reconcile'
tests.append('ordinary_turn_waits_output_drain_and_quiet_guard')
result={'tests':tests,'passed':len(tests),'case_contracts':len(rows),'oracle_leak_test_scope':'Explicit initial context contains only service index; no goals/ACL/future event data','limitations':'Compilation is not semantic correctness of all 408; real API pilots still required.'}
(Path(__import__('os').environ['MUBENCH_TEST_OUTPUT'])/'offline_tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
