"""Public-contract examples and counterexamples for T-R4/JT6."""
from pathlib import Path
import sys,json,copy,os,unittest
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from mubench.config import activate
activate(R/'runtime/runners/tool_augmented_audio')
from experiments.tau_multiuser_voice.business_values import canonical,RULES,VERSION
from experiments.tau_multiuser_voice.interface_contract import normalize_fields
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase,MultiUserEpisodeTrace,ToolActionTrace,GoalResult
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator,_contains_subset,build_reference_final_state
from experiments.tau_multiuser_voice.completion_tools import comparable_state


def case(scene):return MultiUserVoiceCase.model_validate_json((R/'cases/tool_augmented'/(scene+'__g-small__a-clean__c-strong.json')).read_text())
class ValueTests(unittest.TestCase):
 def test_entree_equivalence_is_comparison_only(self):
  from experiments.tau_multiuser_voice.business_values import normalize_arguments
  scene='hospitality_03_table_side_talk'
  for value in ['vegetarian','Vegetarian Entree','vegetarian entrée']:
   self.assertEqual(canonical(scene,{'entree':value}),{'entree':'vegetarian'})
  for value in ['vegan','vegetarian pasta','vegetarian entree without nuts','chicken entree']:
   self.assertNotEqual(canonical(scene,{'entree':value}),{'entree':'vegetarian'})
  raw={'entree':'vegetarian entree','guest_user_id':'p02'}
  self.assertEqual(normalize_arguments(scene,raw),raw)
  self.assertEqual(canonical(scene,{'guest_user_id':'P02','other':'vegetarian entree'}),{'guest_user_id':'P02','other':'vegetarian entree'})
  self.assertEqual(canonical('retail_01',raw),raw)
 def test_timer_labels_only(self):
  s='shared_home_01_parallel_timers';raw={'caller_id':'P01','household_task_id':'HSHMU001','timers_by_owner':{'p01':{'label':' Oven ','minutes':30},'p02':{'label':'Stovetop','minutes':12}}}
  v=canonical(s,raw);self.assertEqual(v['timers_by_owner']['p01']['label'],'oven');self.assertEqual(v['caller_id'],'P01');self.assertEqual(v['household_task_id'],'HSHMU001');self.assertEqual(raw['timers_by_owner']['p01']['label'],' Oven ')
  self.assertNotEqual(canonical(s,{'timers':{'primary':['oven',30]}}),canonical(s,{'timers':{'primary':['oven',12]}}))
 def test_refund_alias_does_not_change_channel(self):
  s='retail_04_return_side_talk';self.assertEqual(canonical(s,{'refund_destination':'original payment method'}),canonical(s,{'refund_destination':'original_payment_method'}));self.assertNotEqual(canonical(s,{'refund_destination':'store_credit'}),canonical(s,{'refund_destination':'original_payment_method'}))
 def test_project_roles_and_ownership(self):
  s='education_04_project_roles';a={'accepted_roles_by_user':{'live':['Slides']},'rejected_roles_by_user':{'live':['Speaking']}}
  b={'accepted_roles_by_user':{'live':['slides']},'rejected_roles_by_user':{'live':['speaking']}}
  self.assertEqual(canonical(s,a),b);self.assertNotEqual(canonical(s,a),{'accepted_roles_by_user':{'primary':['slides']},'rejected_roles_by_user':{'live':['speaking']}})
 def test_numbers_identifiers_unknown_fields_unchanged(self):
  s='vehicle_01_driver_destination';a={'trip_id':'trip-mu-001','caller_id':'P02','destination':'School','unknown':'SCHOOL','amount':100}
  v=canonical(s,a);self.assertEqual(v['destination'],'school');self.assertEqual(v['trip_id'],a['trip_id']);self.assertEqual(v['caller_id'],a['caller_id']);self.assertEqual(v['unknown'],'SCHOOL');self.assertNotEqual(canonical(s,{'destination':'airport'}),canonical(s,{'destination':'school'}))
 def test_time_does_not_infer_am_pm(self):
  s='events_entertainment_05_schedule_time_fix';self.assertEqual(canonical(s,{'doors_time':'6:00'}),{'doors_time':'06:00'});self.assertNotEqual(canonical(s,{'doors_time':'6:00'}),canonical(s,{'doors_time':'18:00'}));self.assertEqual(canonical(s,{'doors_time':'6 PM'}),{'doors_time':'6 PM'})
 def test_slot_matches_actual_fixture(self):
  c=case('public_services_02_appointment_interpreter');args=next(a.argument_subset for a in c.expected_tool_actions if a.tool_name=='schedule_public_service_appointment');args=copy.deepcopy(args);args['slot']='tuesday 9:00'
  e=create_virtual_environment(c.environment);self.assertEqual(e.invoke('schedule_public_service_appointment','primary',**args)['business_state']['slot'],'Tuesday 09:00')
 def test_child_name_mapping_is_unique_and_not_actor_mapping(self):
  s='family_coordination_01_pickup_assignments';v=canonical(s,{'assignments_by_child':{'MAYA':'p02'},'unresolved_children':['LEO']});self.assertEqual(v,{'assignments_by_child':{'maya':'p02'},'unresolved_children':['leo']})
  self.assertEqual(normalize_fields(s,{'assignments_by_child':{'Maya':'p02'}}),{'assignments_by_child':{'Maya':'p02'}})
  with self.assertRaises(ValueError):canonical(s,{'assignments':{'Maya':'p01','maya':'p02'}})
 def test_backend_resolves_only_unique_public_child_spelling(self):
  c=case("family_coordination_01_pickup_assignments");a=next(a for a in c.expected_tool_actions if a.tool_name=="record_school_pickup_assignments");env=create_virtual_environment(c.environment)
  args=copy.deepcopy(a.argument_subset);args["assignments_by_child"]={"maya":"live"};args["unresolved_children"]=["LEO"]
  result=env.invoke(a.tool_name,a.actor_id,**args);self.assertEqual(result["business_state"]["assignments_by_child"],{"Maya":"live"});self.assertEqual(result["business_state"]["unresolved_children"],["Leo"])
 def test_title_words_are_not_removed(self):
  s='shared_home_02_media_backchannel';self.assertNotEqual(canonical(s,{'album':'Full Evening Jazz'}),canonical(s,{'album':'Evening Jazz'}));self.assertEqual(canonical(s,{'album':'Evening  Jazz'}),canonical(s,{'album':'evening jazz'}))
 def test_currency_and_missing_count(self):
  self.assertEqual(canonical('finance_billing_02_two_traveler_exchange',{'requests_by_user':{'p01':{'amount':500,'target_currency':'eur'}}})['requests_by_user']['p01']['target_currency'],'EUR')
  s='emergency_incident_03_roll_call_overlap';self.assertEqual(canonical(s,{'reports_by_user':{'p01':'one person unaccounted for'}})['reports_by_user']['p01'],'one_unaccounted_for');self.assertNotEqual(canonical(s,{'reports_by_user':{'p01':'all accounted for'}}),canonical(s,{'reports_by_user':{'p01':'one unaccounted for'}}))
 def test_initial_sources_survive_version_upgrade(self):
  from experiments.tau_multiuser_voice.scene_context import get_initial_context
  for p in (R/'cases/tool_augmented').glob('*.json'):
   c=MultiUserVoiceCase.model_validate_json(p.read_text());expected=c.task_contract_revision['review'].get('initial_assistant_context')
   self.assertEqual(get_initial_context(c),expected or {},p.name)
 def test_all_408_versioned_and_task_oracles_preserved(self):
  total=0
  for p in (R/'cases/tool_augmented').glob('*.json'):
   c=MultiUserVoiceCase.model_validate_json(p.read_text());r=c.task_contract_revision;self.assertIn(r['design_revision'],{'tool-scene-r4','tool-scene-r5'});self.assertEqual(r['business_value_contract'],VERSION);total+=1
  self.assertEqual(total,408);self.assertEqual(len(RULES),24)
 def test_all_24_reference_states_remain_idempotent_and_extra_effects_visible(self):
  for scene in RULES:
   c=case(scene);state=build_reference_final_state(c);view=comparable_state(c,state);self.assertEqual(view,comparable_state(c,view),scene)
   wrong=copy.deepcopy(state);wrong['unrelated_operation']={'label':'bad'};self.assertNotEqual(comparable_state(c,wrong),view,scene)
 def test_real_timer_execution_accepts_labels_but_not_wrong_duration(self):
  c=case('shared_home_01_parallel_timers');a=next(a for a in c.expected_tool_actions if a.tool_name=='upsert_household_timers');args=copy.deepcopy(a.argument_subset)
  args['timers_by_owner']['primary']['label']='Oven';args['timers_by_owner']['live']['label']='Stovetop'
  env=create_virtual_environment(c.environment);result=env.invoke(a.tool_name,a.actor_id,**args)
  self.assertEqual(result['business_state']['timers_by_owner']['primary']['label'],'oven');self.assertTrue(_contains_subset(args,a.argument_subset,c))
  args['timers_by_owner']['live']['minutes']=20;self.assertFalse(_contains_subset(args,a.argument_subset,c))
if __name__=='__main__':unittest.main()
