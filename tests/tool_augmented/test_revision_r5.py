"""Request revisions, explicit clock formats, and observed return diagnostics."""
from pathlib import Path
import sys,json,copy,unittest,os
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from mubench.config import activate,profile
activate(profile('openai','tool_augmented'))
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase,MultiUserEpisodeTrace,ToolActionTrace,SpokenEventTrace,GoalResult,CapabilityResult
from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.tool_revision_r5 import resolve_case,enrich_judge_packet
from experiments.tau_multiuser_voice.business_values import canonical
from experiments.tau_multiuser_voice.caller_protocol import apply_score_policy

def case(scene):return MultiUserVoiceCase.model_validate_json((R/'cases/tool_augmented'/(scene+'__g-small__a-clean__c-strong.json')).read_text())
def fixture(c):
 t=MultiUserEpisodeTrace(case_id=c.case_id);t.goal_results=[GoalResult(goal_id=g.goal_id,disposition=g.expected_disposition,evidence_verified=True) for u in c.users for g in u.goals];return t
class RevisionTests(unittest.TestCase):
 def test_project_success_without_unneeded_read(self):
  c=case('education_04_project_roles');t=fixture(c);env=create_virtual_environment(c.environment);a=next(a for a in c.expected_tool_actions if a.tool_name=='record_project_role_preferences');env.invoke(a.tool_name,a.actor_id,**a.argument_subset)
  t.tool_actions=[ToolActionTrace(tool_name=a.tool_name,resource_id=a.resource_id,timestamp_ms=3000,result_available_ms=3200,sut_attributed_actor_id='live',oracle_actor_id='live',oracle_authorized=True,executed=True,arguments=a.argument_subset,request_basis_status='supported',caller_provenance_status='supported')];t.final_environment_state=env.snapshot()
  self.assertTrue(apply_score_policy(c,MultiUserEvaluator().evaluate(c,t).model_dump(mode='json'))['task_success'])
 def test_clock_conversion_requires_explicit_daypart(self):
  s='events_entertainment_05_schedule_time_fix';self.assertEqual(canonical(s,{'doors_time':'6:00 PM'})['doors_time'],'18:00');self.assertEqual(canonical(s,{'doors_time':'12:00 am'})['doors_time'],'00:00');self.assertNotEqual(canonical(s,{'doors_time':'06:00'}),canonical(s,{'doors_time':'18:00'}));self.assertEqual(canonical(s,{'unknown':'6:00 PM'})['unknown'],'6:00 PM')
 def test_evening_case_has_public_daypart_and_24h_expected(self):
  c=case('events_entertainment_05_schedule_time_fix');self.assertIn('evening',c.task_contract_revision['review']['initial_assistant_context']['content']);a=next(a for a in c.expected_tool_actions if a.tool_name=='update_event_schedule');self.assertEqual(a.argument_subset['doors_time'],'18:00');self.assertEqual(a.argument_subset['show_time'],'19:00')
 def seat(self,early=False,basis='supported',speaker='live'):
  c=case('events_entertainment_03_accessible_seat');t=fixture(c);env=create_virtual_environment(c.environment);expected=next(a for a in c.expected_tool_actions if a.tool_name=='update_attendee_seat_requirement')
  t.spoken_events=[SpokenEventTrace(event_id='withdraw',proposal_id='withdraw',speaker_id=speaker,addressee='assistant',utterance='I withdraw distance consideration for my own wheelchair-accessible seat entirely.',start_ms=1000,end_ms=2000)]
  def call(priority,ms):
   args=copy.deepcopy(expected.argument_subset);args['proximity_priority']=priority;env.invoke(expected.tool_name,'live',**args);t.tool_actions.append(ToolActionTrace(tool_name=expected.tool_name,resource_id=expected.resource_id,timestamp_ms=ms,result_available_ms=ms+200,sut_attributed_actor_id='live',oracle_actor_id='live',oracle_authorized=True,executed=True,arguments=args,request_basis_status=basis,request_basis_evidence=['withdraw'],caller_provenance_status='supported'))
  if early:call('secondary',500)
  call('none',3000);t.final_environment_state=env.snapshot();return c,t
 def test_first_update_then_revision_and_single_final_update_both_pass(self):
  for early in [False,True]:
   c,t=self.seat(early);self.assertTrue(apply_score_policy(c,MultiUserEvaluator().evaluate(c,t).model_dump(mode='json'))['task_success']);self.assertEqual(c.expected_tool_actions[-1].argument_subset['proximity_priority'],'secondary')
 def test_submitted_argument_without_supported_owner_source_cannot_rewrite_oracle(self):
  for basis,speaker in [('contradicted','live'),('insufficient_evidence','live'),('supported','primary')]:
   c,t=self.seat(basis=basis,speaker=speaker);resolved=resolve_case(c,t);a=next(a for a in resolved.expected_tool_actions if a.tool_name=='update_attendee_seat_requirement');self.assertEqual(a.argument_subset['proximity_priority'],'secondary');self.assertFalse(MultiUserEvaluator().evaluate(c,t).task_success)
 def test_none_and_secondary_not_globally_equivalent(self):
  s='events_entertainment_03_accessible_seat';self.assertNotEqual(canonical(s,{'proximity_priority':'none'}),canonical(s,{'proximity_priority':'secondary'}))
 def test_absent_return_is_na_but_missing_goal_still_fails(self):
  c=case('meeting_04_quiet_dissenter');t=MultiUserEpisodeTrace(case_id=c.case_id);t.capability_results=[CapabilityResult(assertion_id=a.assertion_id,passed=False,reason='No delivered return') for a in c.capability_assertions if a.diagnostic_only];score=MultiUserEvaluator().evaluate(c,t);self.assertTrue(score.recognition_split.pairs);self.assertTrue(all(p.spoken_anchor_passed is None and p.behavior_continuity_passed is None for p in score.recognition_split.pairs));self.assertIsNone(score.recognition_split.spoken_anchor_rate);self.assertIsNone(score.recognition_split.behavior_only_rate);self.assertFalse(score.task_success)
 def test_boundary_rules_reach_semantic_packet(self):
  c=case('public_services_01_transit_accessibility');p={};enrich_judge_packet(c,fixture(c),p);self.assertIn('tool_evaluation_revision',p);self.assertTrue(any('public participant ID' in v for v in p['judge_boundary_rules']));self.assertTrue(any('obsolete' in v for v in p['judge_boundary_rules']))
if __name__=='__main__':unittest.main()
