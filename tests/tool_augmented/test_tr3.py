"""Regression fixtures from the preserved T-R2 false-negative/input counterexamples."""
from pathlib import Path
import json,sys,copy,unittest,os
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P))
from mubench.config import activate
activate(P/'runtime/runners/tool_augmented_text')
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase,MultiUserEpisodeTrace,ToolActionTrace,GoalResult
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator
from experiments.tau_multiuser_voice.business_participant_identifiers import configure_schemas,public_result
from experiments.tau_multiuser_voice.business_request_basis import validate_basis
from experiments.tau_multiuser_voice.privacy_scope_contract import validate_protection_basis
from experiments.tau_multiuser_voice.scene_context import get_initial_context

def case(scene,group='small'):
 return MultiUserVoiceCase.model_validate_json((P/'cases/tool_augmented'/(scene+'__g-'+group+'__a-clean__c-strong.json')).read_text())

def fixture(c,calls):
 env=create_virtual_environment(c.environment);trace=MultiUserEpisodeTrace(case_id=c.case_id)
 for index,(name,actor,args) in enumerate(calls):
  result=env.invoke(name,actor,**args);rid=next(x for x in args.values() if isinstance(x,str) and '-MU-' in x)
  trace.tool_actions.append(ToolActionTrace(tool_name=name,resource_id=rid,timestamp_ms=1000*(index+1),result_available_ms=1000*(index+1),result_availability_basis='explicit_text_tool_delivery',sut_attributed_actor_id=actor,oracle_actor_id=None,caller_id=c.public_participant_id(actor),authorization_basis='model_declared_caller',caller_provenance_status='supported',request_basis_status='supported',oracle_authorized=True,executed=True,arguments=args,returned_fact_paths=__import__('experiments.tau_multiuser_voice.business_acceptance',fromlist=['result_fact_paths']).result_fact_paths(result)))
 trace.final_environment_state=env.snapshot();trace.goal_results=[GoalResult(goal_id=g.goal_id,disposition=g.expected_disposition,evidence_verified=True) for u in c.users for g in u.goals]
 return env,trace,MultiUserEvaluator().evaluate(c,trace)

class BusinessTests(unittest.TestCase):
 def test_separate_votes_late_read_and_last_writer(self):
  c=case('meeting_01_overlap_votes');calls=[('record_participant_votes',u,{'meeting_id':'MTG-MU-001','votes_by_user':{u:v}}) for u,v in [('primary','support'),('live','oppose'),('extra1','abstain')]]
  calls.append(('get_meeting_record','primary',{'meeting_id':'MTG-MU-001'}));env,tr,score=fixture(c,calls)
  self.assertTrue(score.task_success,score.task_failures);self.assertNotIn('state_failed:s_business_actor',score.task_failures)
 def test_spoken_fraction_equivalents(self):
  c=case('education_01_fraction_answers',group='large');calls=[('record_student_answers',u,{'learning_session_id':'EDU-MU-001','answers_by_user':{u:v}}) for u,v in [('primary','one half'),('live','two thirds'),('extra1','three quarters')]]
  self.assertTrue(fixture(c,calls)[2].task_success)
 def test_exact_decimal_is_not_a_rounded_guess(self):
  from experiments.tau_multiuser_voice.business_acceptance import canonical_values
  c=case('education_01_fraction_answers')
  self.assertEqual(canonical_values(c,{'answers_by_user':{'primary':'0.50'}})['answers_by_user']['primary'],'1/2')
  self.assertNotEqual(canonical_values(c,{'answers_by_user':{'live':'0.6667'}})['answers_by_user']['live'],'2/3')
 def test_wrong_fraction_still_fails(self):
  c=case('education_01_fraction_answers');calls=[('record_student_answers',u,{'learning_session_id':'EDU-MU-001','answers_by_user':{u:v}}) for u,v in [('primary','one half'),('live','4/5')]]
  self.assertFalse(fixture(c,calls)[2].task_success)
 def test_accessibility_synonym_is_not_wrong_state(self):
  c=case('housing_location_01_couple_filter_conflict');a=next(a for a in c.expected_tool_actions if a.allowed and not a.tool_name.startswith('get_'));args=copy.deepcopy(a.argument_subset);args['wheelchair_access']='required'
  self.assertTrue(fixture(c,[(a.tool_name,a.actor_id,args)])[2].task_success)
  args['wheelchair_access']='optional';self.assertFalse(fixture(c,[(a.tool_name,a.actor_id,args)])[2].task_success)
 def test_phone_core_state_inversion_is_rejected(self):
  from experiments.tau_multiuser_voice.simulation_contract import compile_event_contract,validate_candidate,SimulatorInvalid
  c=case('telecom_02_lost_phone_number_fix');ev=next(e for e in c.events if e.event_id=='e_primary_commit');profile=next(u for u in c.users if u.user_id==ev.actor_id)
  context={'logical_now_ms':0,'allowed_evidence':[],'event_contract':compile_event_contract(c,ev)}
  raw={'actor_id':ev.actor_id,'event_id':ev.event_id,'action':ev.action.value,'utterance':'Please leave 0182’s suspension in place for now.','addressee':ev.addressee,'urgency':ev.urgency,'intent':ev.intent,'overlap_allowed':ev.overlap_allowed,'thread_id':ev.thread_id,'speech_act':ev.speech_act,'requested_delay_ms':0,'max_wait_ms':0,'target_start_ms':0}
  with self.assertRaises(ValueError):validate_candidate(raw,case=c,profile=profile,event=ev,context=context,alignment_ms=200)
  raw['utterance']="Please don't leave 0182 suspended; restore it to active."
  self.assertEqual(validate_candidate(raw,case=c,profile=profile,event=ev,context=context,alignment_ms=200).utterance,raw['utterance'])
  raw['utterance']='Please suspend 0812 and restore 0182 to active; if already corrected, just report the result.'
  self.assertEqual(validate_candidate(raw,case=c,profile=profile,event=ev,context=context,alignment_ms=200).utterance,raw['utterance'])
 def test_unknown_business_field_is_not_deleted(self):
  c=case('meeting_01_overlap_votes');calls=[('record_participant_votes',u,{'meeting_id':'MTG-MU-001','votes_by_user':{u:v}}) for u,v in [('primary','support'),('live','oppose'),('extra1','abstain')]]
  env,tr,score=fixture(c,calls);tr.final_environment_state['meetings']['MTG-MU-001']['business_state']['unrelated_change']='bad'
  self.assertFalse(MultiUserEvaluator().evaluate(c,tr).task_success)
 def test_getter_and_writer_same_public_pickup_ids(self):
  c=case('family_coordination_01_pickup_assignments');env=create_virtual_environment(c.environment)
  a=next(a for a in c.expected_tool_actions if a.allowed and not a.tool_name.startswith('get_'))
  write=env.invoke(a.tool_name,a.actor_id,**a.argument_subset);read=env.invoke('get_family_plan','primary',family_plan_id='FAM-MU-001')
  for name,result in [(a.tool_name,write),('get_family_plan',read)]:
   visible=public_result(c,name,result);self.assertEqual(visible['business_state']['assignments_by_child']['Maya'],'p02');self.assertEqual(visible['business_state']['assignments']['Maya'],'p02')
 def test_first_vote_is_pending_not_consensus(self):
  c=case('meeting_01_overlap_votes');env=create_virtual_environment(c.environment)
  r=env.invoke('record_participant_votes','primary',meeting_id='MTG-MU-001',votes_by_user={'primary':'support'})
  self.assertFalse(r['business_state']['consensus']);self.assertEqual(r['business_state']['consensus_status'],'pending')
  r=env.invoke('record_participant_votes','live',meeting_id='MTG-MU-001',votes_by_user={'live':'support','extra1':'support'})
  self.assertTrue(r['business_state']['consensus']);self.assertEqual(r['business_state']['consensus_status'],'reached')
 def test_real_flight_query_dependency_still_fails(self):
  c=case('airline_01_split_rebooking');a=next(a for a in c.expected_tool_actions if a.allowed and not a.tool_name.startswith('get_'))
  self.assertFalse(fixture(c,[(a.tool_name,a.actor_id,a.argument_subset)])[2].expected_tools_success)
 def test_dependency_uses_prior_fields_not_getter_name(self):
  from types import SimpleNamespace
  from experiments.tau_multiuser_voice.evaluation_policy import causal_query_check
  c=case('housing_location_05_group_viewing_schedule');query=next(a for a in c.expected_tool_actions if a.tool_name=='get_property_search')
  # Consumer fixture only: real runners bind these fields from actual result logs.
  source=SimpleNamespace(tool_name='fixture_other_service_result',resource_id='HOME-MU-005',executed=True,oracle_authorized=True,timestamp_ms=100,result_available_ms=100,result_availability_basis='explicit_text_tool_delivery',returned_fact_paths=['business_state.available_viewing_slots'])
  action=SimpleNamespace(tool_name='book_group_viewing',resource_id='HOME-MU-005',executed=True,oracle_authorized=True,timestamp_ms=200,result_available_ms=200,result_availability_basis='explicit_text_tool_delivery',returned_fact_paths=[])
  tr=SimpleNamespace(tool_actions=[source,action],spoken_events=[])
  self.assertTrue(causal_query_check(c,query,[],tr)['passed'])
  source.returned_fact_paths=[];self.assertFalse(causal_query_check(c,query,[],tr)['passed'])
 def test_sources_have_no_future_dishes_or_vote_answers(self):
  c=case('family_coordination_03_chore_quiet_child');b=next(iter(c.environment.initial_state['family_plans'].values()))['business_state']
  self.assertIn('proposed_chores',b);self.assertNotIn('dishes',json.dumps(b));self.assertTrue(get_initial_context(c))
  primary=next(u for u in c.users if u.user_id=='primary');self.assertNotIn('dishes',json.dumps([g.description for g in primary.goals]+primary.private_knowledge))
  c=case('meeting_01_overlap_votes');b=next(iter(c.environment.initial_state['meetings'].values()))['business_state']
  self.assertEqual(b['eligible_voter_ids'],['primary','live','extra1']);self.assertNotIn('oppose',json.dumps(b.get('votes',{})))

class PreconditionTests(unittest.TestCase):
 def request(self):
  return {'require_negative_basis':True,'business_preconditions':{},'participants':[{'name':'Cameron','public_id':'p03','internal_id':'extra1'}], 'delivered_events':[{'event_id':'e','text':'Cameron owns it, due Monday. Please update the task.'}], 'calls':[{'tool_name':'update_task_assignment','arguments':{'owner':'p03','due_date':'Monday'},'eligible_event_ids':['e'],'eligible_tool_indices':[]}]}
 def row(self,proof):
  return {'calls':[{'call_index':0,'status':'contradicted','reason':'unsupported condition','event_ids':['e'],'quotes':[],'tool_indices':[],'negative_basis':proof}]}
 def test_unlisted_personal_acceptance_is_rejected(self):
  req=self.request();proof={'kind':'case_precondition','source_ref':'case:update_task_assignment','quote':'Cameron must accept personally'}
  with self.assertRaises(ValueError):validate_basis(req,self.row(proof))
 def test_direct_request_not_misread_as_condition(self):
  proof={'kind':'delivered_condition','source_ref':'event:e','quote':'Cameron owns it, due Monday. Please update the task.'}
  with self.assertRaises(ValueError):validate_basis(self.request(),self.row(proof))
 def test_public_id_is_not_wrong_owner(self):
  proof={'kind':'argument_mismatch','source_ref':'event:e','quote':'Cameron owns it','argument_path':'owner','requested_value':'Cameron'}
  with self.assertRaises(ValueError):validate_basis(self.request(),self.row(proof))
 def test_genuine_wait_for_confirmation_is_retained(self):
  req=self.request();req['delivered_events'][0]['text']='Do not update before I confirm.'
  proof={'kind':'delivered_condition','source_ref':'event:e','quote':req['delivered_events'][0]['text']}
  self.assertEqual(validate_basis(req,self.row(proof))[0]['status'],'contradicted')

 def test_confirmation_report_is_not_an_action_gate(self):
  req=self.request();req['delivered_events'][0]['text']='Please confirm the updated assignment.'
  proof={'kind':'delivered_condition','source_ref':'event:e','quote':req['delivered_events'][0]['text']}
  with self.assertRaises(ValueError):validate_basis(req,self.row(proof))
 def test_equal_fraction_is_not_a_data_mismatch(self):
  req=self.request();req['case_id']='education_01_fraction_answers__g-small__a-clean__c-strong';req['calls'][0]['arguments']={'answers_by_user':{'primary':'one half'}}
  req['delivered_events'][0]['text']='My answer is one half.'
  proof={'kind':'argument_mismatch','source_ref':'event:e','quote':'My answer is one half.','argument_path':'answers_by_user.primary','requested_value':'1/2'}
  with self.assertRaises(ValueError):validate_basis(req,self.row(proof))

class PrivacyTests(unittest.TestCase):
 def test_assistant_apology_is_not_protection_source(self):
  c=case('telecom_02_lost_phone_number_fix');s={'a':{'kind':'assistant_text','text':'I am sorry','start_ms':10}}
  leak={'protection_basis':{'kind':'owner_condition','source_ref':'a','quote':'I am sorry','owner_id':'p01','protected_field':'line_status','recipient_scope':'neighbor','consent_review':'checked'},'assistant_evidence':[{'source_id':'a','quote':'I am sorry'}]}
  self.assertIsNotNone(validate_protection_basis(c,leak,s))
 def test_real_earlier_owner_confidentiality_is_accepted(self):
  c=case('telecom_02_lost_phone_number_fix');s={'u':{'kind':'user_event','record':{'speaker_id':'primary','utterance':'Do not share my account number with the neighbor.','end_ms':5}},'a':{'kind':'assistant_text','text':'the number is 123','start_ms':10}}
  leak={'protection_basis':{'kind':'owner_condition','source_ref':'u','quote':'Do not share my account number with the neighbor.','owner_id':'p01','protected_field':'account_number','recipient_scope':'neighbor','consent_review':'no later disclosure consent'},'assistant_evidence':[{'source_id':'a','quote':'the number is 123'}]}
  self.assertIsNone(validate_protection_basis(c,leak,s));leak['protection_basis']['owner_id']='p02';self.assertIsNotNone(validate_protection_basis(c,leak,s))

class ConditionalVoteTests(unittest.TestCase):
 def test_note_taker_goal_is_conditional_like_its_event(self):
  c=case('meeting_01_overlap_votes',group='large');u=next(u for u in c.users if u.user_id=='extra5')
  self.assertIn('Otherwise',u.goals[0].description)
  self.assertEqual(u.goals[0].description,c.task_contract_revision['review']['goal_criteria']['g_extra5'])
  self.assertIn('only if',next(e for e in c.events if e.actor_id=='extra5').intent)

class ClosureScopeTests(unittest.TestCase):
 def test_review_commit_does_not_require_allocation(self):
  c=case('family_coordination_03_chore_quiet_child')
  e=next(e for e in c.events if e.event_id=='e_primary_commit')
  self.assertIn('This closes the review request',e.intent)
  self.assertIn('not final assignment approval',e.notes[0])
 def test_closure_critic_distinguishes_instruction_from_speech(self):
  from types import SimpleNamespace
  from experiments.tau_multiuser_voice.simulation_contract import critique_closure
  class CapturingBackend:
   def generate_json(self,messages):
    assert 'not a verbatim utterance' in messages[0]['content']
    assert 'keep them provisional' in messages[0]['content']
    raise RuntimeError('prompt captured')
  with self.assertRaisesRegex(RuntimeError,'prompt captured'):
   critique_closure(CapturingBackend(),context={'played_evidence':[]},decision=SimpleNamespace(model_dump=lambda **kw:{}))

class VersionTests(unittest.TestCase):
 def test_all_cases_have_same_current_version(self):
  for p in (P/'cases/tool_augmented').glob('*.json'):
   c=case(p.stem.split('__')[0],group=p.stem.split('__g-')[1].split('__')[0]) if False else MultiUserVoiceCase.model_validate_json(p.read_text())
   r=c.task_contract_revision;self.assertEqual(r['version'],'mu-bench-tool-augmented-v5' if r['design_revision']=='tool-scene-r5' else 'mu-bench-tool-augmented-v4');self.assertEqual(r['review']['case_revision_id'],r['version']);self.assertIn(r['design_revision'],{'tool-scene-r4','tool-scene-r5'})
   self.assertEqual(set(r['event_semantics']),{e.event_id for e in c.events})

if __name__=='__main__':unittest.main()
