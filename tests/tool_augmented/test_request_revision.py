from pathlib import Path
import sys,json,copy
H=Path(__file__).resolve().parent;

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mubench.config import activate,profile
activate(profile('chat','tool_augmented',transport='text'))
from run_text_case import Episode
from adapters.providers.chat.adapter import TextStream
from experiments.tau_multiuser_voice.models import *
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator
from experiments.tau_multiuser_voice.caller_protocol import apply_score_policy
from experiments.tau_multiuser_voice.authorization import AuthorizationOracle
from experiments.mu_bench.logical_time.online_users import OrderedOnlineEventGraph
from experiments.mu_bench.logical_time.pacing import build_case_pacing
passed=[];cases=[MultiUserVoiceCase.model_validate(json.loads(p.read_text())) for p in sorted((ROOT/'cases/tool_augmented').glob('*.json'))]
for c in cases:
 g=OrderedOnlineEventGraph(c,pacing_plan=build_case_pacing(c));now=0
 for index,e in enumerate(c.events):
  if index:
   spec=c.task_contract_revision['semantic_event_release'][e.event_id];now+=spec['timeout_ms']
  g.pacing.observe_media(now,False,False,played_duration_ms=0)
  ready=g.ready_events(WorldObservation(sim_time_ms=now,kind=ObservationKind.CONVERSATION_START if index==0 else ObservationKind.CLOCK))
  assert ready and ready[0].event_id==e.event_id,(c.case_id,e.event_id,now)
  now+=1000;g.mark_completed(e.event_id,now)
passed.append('all_408_graphs_reach_every_authored_event_without_model_output_or_tools')
c=next(c for c in cases if c.case_id=='airline_01_split_rebooking__g-small__a-clean__c-strong')
def fixture(initial=False,query=True,alternative=False):
 env=create_virtual_environment(c.environment);tr=MultiUserEpisodeTrace(case_id=c.case_id)
 tr.spoken_events=[SpokenEventTrace(event_id='e_open',proposal_id='open',speaker_id='primary',addressee='assistant',utterance='Move our group to the available alternative flight.',start_ms=0,end_ms=1000),SpokenEventTrace(event_id='e_primary_reconcile' if alternative else 'e_live',proposal_id='revision',speaker_id='primary' if alternative else 'live',addressee='assistant',utterance='Move Avery but keep Blake on the original flight.',start_ms=4000,end_ms=5000)]
 def call(name,args,ts):
  result=env.invoke(name,'primary',**args)
  tr.tool_actions.append(ToolActionTrace(tool_name=name,resource_id='BK-MU-001',timestamp_ms=ts,result_available_ms=ts+200,result_availability_basis='fixture_exact',sut_attributed_actor_id='primary',oracle_actor_id=None,caller_id='p01',authorization_basis='model_declared_caller',caller_provenance_status='supported',request_basis_status='supported',oracle_authorized=True,executed=True,arguments=args,returned_fact_paths=__import__('experiments.tau_multiuser_voice.business_acceptance',fromlist=['result_fact_paths']).result_fact_paths(result)))
  return result
 if query:call('get_booking_context',{'booking_id':'BK-MU-001'},1200)
 if initial:call('rebook_selected_travelers',{'booking_id':'BK-MU-001','rebook_user_ids':['primary','live'],'keep_original_user_ids':[],'target_flight_id':'FLT-MU-220'},2000)
 call('rebook_selected_travelers',{'booking_id':'BK-MU-001','rebook_user_ids':['primary'],'keep_original_user_ids':['live'],'target_flight_id':'FLT-MU-220'},6000)
 tr.final_environment_state=env.snapshot();tr.goal_results=[GoalResult(goal_id=g.goal_id,disposition=g.expected_disposition,evidence_verified=True) for u in c.users for g in u.goals]
 return tr,apply_score_policy(c,MultiUserEvaluator().evaluate(c,tr).model_dump(mode='json'))
for initial in (True,False):
 tr,score=fixture(initial=initial);assert score['task_success'],score['task_failures']
passed.append('lawful_initial_execution_then_repair_and_no_initial_execution_both_pass_TS')
tr,score=fixture(alternative=True);assert score['task_success']
passed.append('actual_later_revision_can_replace_missing_fixed_e_live')
tr,score=fixture(query=False);assert not score['expected_tools_success']
passed.append('real_query_dependency_is_not_removed')
tr,score=fixture();tr.tool_actions[-1].request_basis_status='contradicted'
assert not MultiUserEvaluator().evaluate(c,tr).expected_tools_success
passed.append('future_or_unsupported_request_basis_cannot_pass_positive_tool_obligation')
g=OrderedOnlineEventGraph(c,pacing_plan=build_case_pacing(c));g.mark_completed('e_open',1000);g.pacing.observe_media(2000,False,False,played_duration_ms=0)
result={'status':'updated','booking_id':'BK-MU-001','transaction_id':'TX-test'}
assert g.ready_events(WorldObservation(sim_time_ms=2000,kind=ObservationKind.TOOL_RESULT,tool_name='rebook_selected_travelers',tool_result=json.dumps(result)))[0].event_id=='e_live'
passed.append('actual_successful_rebooking_result_releases_objection_before_deadline')
for scene,wrong,correct,key,bad in [
 ('retail_02_shared_address_correction',{'house_number':81}, {'house_number':18},'corrected_house_number',18),
 ('travel_identity_01_passport_digit_correction',{'passport_number':'P8899011'},{'passport_number':'P8890011'},'corrected_candidate','P8890011')]:
 cc=next(x for x in cases if x.case_id.startswith(scene+'__g-small__a-clean__c-strong'))
 expected=next(a for a in cc.expected_tool_actions if a.must_execute and a.allowed and not a.tool_name.startswith('get_'))
 env=create_virtual_environment(cc.environment);args=copy.deepcopy(expected.argument_subset);args.update(wrong)
 result=env.invoke(expected.tool_name,expected.actor_id,**args);assert result['business_state'][key]!=bad
 args.update(correct);result=env.invoke(expected.tool_name,expected.actor_id,**args);assert result['business_state'][key]==bad
passed.append('pending_address_and_passport_return_actual_inputs_not_future_answer')
cc=next(x for x in cases if x.case_id.startswith('telecom_02_lost_phone_number_fix__g-small__a-clean__c-strong'))
env=create_virtual_environment(cc.environment);a=next(a for a in cc.expected_tool_actions if a.must_execute)
args=copy.deepcopy(a.argument_subset);args.update(lost_line_suffix='0182',preserve_active_line_suffix='0812');r=env.invoke(a.tool_name,'primary',**args)
assert r['business_state']['line_0182_status']=='suspended' and r['business_state']['line_0812_status']=='active'
r=env.invoke(a.tool_name,'primary',**a.argument_subset);assert r['business_state']['line_0812_status']=='suspended' and r['business_state']['line_0182_status']=='active'
passed.append('lost_line_wrong_initial_then_corrected_state_is_real_not_prefilled')
cc=next(x for x in cases if x.case_id.startswith('airline_03_payer_traveler_cancel__g-small__a-clean__c-strong'))
u=next(u for u in cc.users if u.user_id=='primary');rt=UserRuntimeState(user_id=u.user_id,active_grant_ids={g.grant_id for g in u.permissions if g.active_initially})
assert not AuthorizationOracle(cc).decide(actor_id='primary',action='cancel_traveler_itinerary',resource_id='BK-MU-003',runtime=rt).allowed
passed.append('first_action_exception_does_not_grant_unauthorized_payer_cancellation')
from experiments.tau_multiuser_voice.evaluation_policy import cited_result_temporal_check
cc=c;row={'assistant_evidence':[{'source_id':'speech'}],'tool_evidence':[{'source_id':'write'}]}
sources={'speech':{'kind':'assistant_text','source_id':'speech','start_ms':100,'end_ms':1000},'write':{'kind':'executed_tool_result','source_id':'write','tool_name':'rebook_selected_travelers','timestamp_ms':500,'result_available_ms':700,'result_availability_basis':'explicit'}}
errors,pending=cited_result_temporal_check(cc,row,sources);assert pending and not errors
sources['speech'].update(start_ms=800,end_ms=1200);assert cited_result_temporal_check(cc,row,sources)==([],[])
passed.append('write_report_overlap_is_pending_alignment_and_true_later_report_is_supported')
out={'cases':408,'scenes':51,'passed':len(passed),'checks':passed,'synthetic_fixtures':True,'api_calls':0}
(Path(__import__('os').environ['MUBENCH_TEST_OUTPUT'])/'request_revision_tests.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False))
