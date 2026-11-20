from pathlib import Path
import sys,json,copy
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from mubench.config import activate
activate(R/'runtime/runners/tool_augmented_audio')
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase,MultiUserEpisodeTrace
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.business_participant_identifiers import configure_schemas,public_result
from experiments.tau_multiuser_voice.caller_protocol import resolve_caller,apply_score_policy
from run_tau_mu_voice_realtime import normalize_tool_arguments
from experiments.tau_multiuser_voice.evaluator import MultiUserEvaluator,_contains_subset,build_reference_final_state
from experiments.tau_multiuser_voice.completion_tools import comparable_state
from experiments.tau_multiuser_voice.medication_source_contract import TEXT_RULE
rows=[]
DESC='Blake supplied this family written-list entry; it remains unverified, pending patient confirmation.'
for p in sorted((R/'cases/tool_augmented').glob('healthcare_care_02_medication_list_source__*.json')):
 c=MultiUserVoiceCase.model_validate_json(p.read_text());e=create_virtual_environment(c.environment);configure_schemas(e,c.environment.tool_names)
 schema=e.definitions['record_medication_verification'].parameters
 assert 'No fixed wording' in schema['properties']['source']['description'] and 'enum' not in schema['properties']['source']
 a=next(a for a in c.expected_tool_actions if a.tool_name=='record_medication_verification')
 raw=public_result(c,a.tool_name,copy.deepcopy(a.argument_subset));raw.update(caller_id='p01',source=DESC)
 actor,_,args,err=resolve_caller(c,raw);assert not err;args=normalize_tool_arguments(a.tool_name,args,case=c)
 before=e.snapshot()['care_plans']['CARE-MU-002']['business_state'];result=e.invoke(a.tool_name,actor,**args);b=result['business_state']
 assert b['source']==before['source']=='care_plan_list' and b['existing_medication_list']==before['existing_medication_list']
 assert b['external_medication_notes'][0]['source']==DESC
 assert e.invoke(a.tool_name,actor,**args)==result and len(e.snapshot()['audit_events'])==1
 assert _contains_subset(args,a.argument_subset,c)
 expected=next(a for a in c.state_assertions if a.assertion_id=='s_business_state');assert _contains_subset(b,expected.expected,c)
 assert comparable_state(c,e.snapshot())==comparable_state(c,build_reference_final_state(c))
 for key,val in [('documented_dose','20 mg'),('source_participant_user_id','primary'),('source_document','care_plan_list_v1'),('confirmation_status','patient_confirmed'),('source','')]:
  bad=copy.deepcopy(args);bad[key]=val;assert not _contains_subset(bad,a.argument_subset,c),(p.name,key)
 bad=e.snapshot();bad['care_plans']['CARE-MU-002']['business_state']['existing_medication_list'][0]['dose']='10 mg';assert comparable_state(c,bad)!=comparable_state(c,build_reference_final_state(c))
 bad=e.snapshot();bad['care_plans']['CARE-MU-002']['business_state']['source']='invented_list';assert comparable_state(c,bad)!=comparable_state(c,build_reference_final_state(c))
 assert not _contains_subset({'source':'wrong'},{'source':'written_list'},c)
 assert 'false source or patient-confirmation claim' in c.task_contract_revision['review']['goal_criteria']['g_live']
 rows.append({'case_id':c.case_id,'public_free_text':True,'source_preserved':True,'faithful_text_no_literal_gate':True,'idempotent_transaction':True,'wrong_structured_values_rejected':True,'wrong_original_list_and_source_rejected':True})
# Explicit old-layout compatibility is isolated to a scoring copy.
c=MultiUserVoiceCase.model_validate_json(p.read_text());c.task_contract_revision['medication_source_contract']['legacy_flat_source_compatibility']=True
legacy=copy.deepcopy(e.snapshot());legacy['care_plans']['CARE-MU-002']['business_state']['source']=DESC
assert comparable_state(c,legacy)==comparable_state(c,build_reference_final_state(c))
bad=copy.deepcopy(legacy);bad['care_plans']['CARE-MU-002']['business_state']['source']='unrelated altered source'
assert comparable_state(c,bad)!=comparable_state(c,build_reference_final_state(c))
print(json.dumps({'cases':rows,'legacy_unrelated_source_not_masked':True,'api_calls':0,'passed':True}))
