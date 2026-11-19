from pathlib import Path
import os,sys,json,copy
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P));from mubench.config import activate;activate(P/'runtime/runners/tool_augmented_text')
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.business_participant_identifiers import configure_schemas,public_result
from experiments.tau_multiuser_voice.caller_protocol import resolve_caller
from run_tau_mu_voice_realtime import normalize_tool_arguments
errors=[]
for p in (P/'cases/tool_augmented').glob('*.json'):
 c=MultiUserVoiceCase.model_validate_json(p.read_text());env=create_virtual_environment(c.environment);configure_schemas(env,c.environment.tool_names)
 for a in c.expected_tool_actions:
  if not (a.allowed and a.must_execute):continue
  raw=public_result(c,a.tool_name,copy.deepcopy(a.argument_subset));raw['caller_id']=c.public_participant_id(a.actor_id)
  actor,caller,args,error=resolve_caller(c,raw);assert not error
  args=normalize_tool_arguments(a.tool_name,args,case=c)
  try:env.invoke(a.tool_name,actor,**args)
  except Exception as e:errors.append({'case_id':c.case_id,'tool':a.tool_name,'error':str(e)})
print(json.dumps({'case_count':408,'adapter_path_errors':errors[:8],'total_errors':len(errors),'api_calls':0}));assert not errors
# Negative mutations must retain their actual values, not inherit target answers.
checks=[('meeting_04_quiet_dissenter','stance','support','votes','live','support'),('telecom_01_family_data_lines','gigabytes_by_user',{'live':7},'data_pass_gigabytes_by_user','live',7),('education_01_fraction_answers','answers_by_user',{'live':'4/5'},'answers','live','4/5')]
for base,key,value,resultmap,uid,expected in checks:
 p=next((P/'cases/tool_augmented').glob(base+'__g-small__a-clean__c-strong.json'));c=MultiUserVoiceCase.model_validate_json(p.read_text());env=create_virtual_environment(c.environment)
 a=next(a for a in c.expected_tool_actions if a.allowed and not a.tool_name.startswith('get_'));args=copy.deepcopy(a.argument_subset);args[key]=value;r=env.invoke(a.tool_name,a.actor_id,**args);assert r['business_state'][resultmap][uid]==expected,(base,r)
print('three wrong-input mutations preserve actual effects')
