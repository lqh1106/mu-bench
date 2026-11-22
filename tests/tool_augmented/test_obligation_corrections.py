from pathlib import Path
import sys,json,unittest,copy
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from mubench.config import activate,profile
activate(profile('openai','tool_augmented'))
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase,MultiUserEpisodeTrace,GoalResult,SpokenEventTrace,ToolActionTrace
from experiments.tau_multiuser_voice.tool_obligation_corrections import correct_case,mark_conditional,goal_matches,date_value,MARKER,enrich
from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator
from experiments.tau_multiuser_voice.caller_protocol import apply_score_policy
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment

def load(scene):return MultiUserVoiceCase.model_validate_json((R/'cases/tool_augmented'/(scene+'__g-small__a-clean__c-strong.json')).read_text())
class CorrectionTests(unittest.TestCase):
 def conditional(self):
  c=load('emergency_incident_03_roll_call_overlap');t=MultiUserEpisodeTrace(case_id=c.case_id);t.spoken_events=[SpokenEventTrace(event_id=e.event_id,proposal_id=str(i),speaker_id=e.actor_id,addressee='assistant',utterance='No corrected description supplied.',start_ms=i*10,end_ms=i*10+5) for i,e in enumerate(c.events)]
  raw={'goal_results':[{'goal_id':'g_extra2','disposition':'ignore','applicability':'not_triggered','completion_checks':[{'passed':True}],'event_evidence':[{'source_id':'user_event_000003'}]}]};norm={'goal_results':[{'goal_id':'g_extra2','disposition':'ignore','evidence_verified':True,'evidence':[]}]};return c,t,raw,norm
 def test_verified_conditional_survives_typed_goal_conversion(self):
  c,t,raw,norm=self.conditional();self.assertTrue(mark_conditional(c,t,norm,raw));g=GoalResult.model_validate(norm['goal_results'][0]);self.assertTrue(goal_matches(c,'g_extra2',g,'fulfill'));self.assertEqual(g.disposition,'ignore')
 def test_missing_event_never_becomes_nonapplicable(self):
  c,t,raw,norm=self.conditional();t.spoken_events.pop();self.assertFalse(mark_conditional(c,t,norm,raw));self.assertFalse(goal_matches(c,'g_extra2',GoalResult.model_validate(norm['goal_results'][0]),'fulfill'))
 def test_no_marker_for_triggered_or_unverified_goal(self):
  for change in ['triggered','unverified','failed_check']:
   c,t,raw,norm=self.conditional()
   if change=='triggered':raw['goal_results'][0]['applicability']='applicable'
   if change=='unverified':norm['goal_results'][0]['evidence_verified']=False
   if change=='failed_check':raw['goal_results'][0]['completion_checks'][0]['passed']=False
   self.assertFalse(mark_conditional(c,t,norm,raw))
 def test_unconditional_goal_cannot_bypass_oracle(self):
  c,t,raw,norm=self.conditional();g=GoalResult(goal_id='g_primary',disposition='ignore',evidence_verified=True,evidence=[MARKER]);self.assertFalse(goal_matches(c,'g_primary',g,'fulfill'))
 def chores(self,record=False):
  c=load('family_coordination_03_chore_quiet_child');t=MultiUserEpisodeTrace(case_id=c.case_id);t.goal_results=[GoalResult(goal_id=g.goal_id,disposition=g.expected_disposition,evidence_verified=True) for u in c.users for g in u.goals];env=create_virtual_environment(c.environment)
  if record:
   args={'family_plan_id':'FAM-MU-003','family_member_user_id':'live','existing_chore':'dishes_daily'};env.invoke('record_existing_chore_constraint','live',**args);t.tool_actions=[ToolActionTrace(tool_name='record_existing_chore_constraint',resource_id='FAM-MU-003',timestamp_ms=1000,sut_attributed_actor_id='live',oracle_actor_id='live',oracle_authorized=True,executed=True,arguments=args,request_basis_status='supported')]
  t.final_environment_state=env.snapshot();return c,t
 def test_review_does_not_require_unrequested_write(self):
  c,t=self.chores();self.assertTrue(apply_score_policy(c,MultiUserEvaluator().evaluate(c,t).model_dump(mode='json'))['task_success'])
 def test_valid_optional_workload_note_allowed(self):
  c,t=self.chores(True);self.assertTrue(apply_score_policy(c,MultiUserEvaluator().evaluate(c,t).model_dump(mode='json'))['task_success'])
 def test_wrong_or_finalized_proposal_still_fails(self):
  c,t=self.chores();t.final_environment_state['family_plans']['FAM-MU-003']['business_state']['allocation_phase']='finalized';self.assertFalse(MultiUserEvaluator().evaluate(c,t).task_success)
 def test_dates_preserve_actual_calendar_difference(self):
  self.assertEqual(date_value('2026-1101'),date_value('2026-11-01'));self.assertEqual(date_value('20261101'),'2026-11-01');self.assertNotEqual(date_value('2026-11-01'),date_value('2026-11-02'));self.assertIsNone(date_value('11/01/26'));self.assertIsNone(date_value('2026-0230'))
 def test_boundary_rules_survive_actual_semantic_request(self):
  from experiments.tau_multiuser_voice.jt3_review import semantic_review
  from tempfile import TemporaryDirectory
  captured=[]
  class Capture:
   def post_json(self,url,key,endpoint,payload):
    captured.append(json.loads(payload['messages'][1]['content']))
    raise RuntimeError('capture-only: no API call')
  packet={'judge_boundary_rules':['existing-rule'],'required_output_schema':{}};enrich(packet)
  with TemporaryDirectory() as d:
   with self.assertRaisesRegex(RuntimeError,'capture-only'):
    semantic_review(Capture(),{'GPT_URL':'unused','GPT_KEY':'unused'},load('family_coordination_03_chore_quiet_child'),{'goals':[]},{'sources':[]},packet,Path(d),lambda *args:None)
  self.assertIn('existing-rule',captured[0]['judge_boundary_rules'])
  self.assertTrue(any('calendar' in r for r in captured[0]['judge_boundary_rules']))
 def test_date_rule_reaches_packet_without_altering_quotes(self):
  p={'evidence':{'text':'2026-1101'}};enrich(p);self.assertEqual(p['evidence']['text'],'2026-1101');self.assertTrue(any('calendar' in r for r in p['judge_boundary_rules']))
if __name__=='__main__':unittest.main()
