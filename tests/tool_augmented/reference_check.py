"""All-knowing fixture check; never supplied to SUT and never a model result."""
from pathlib import Path
import sys,json,copy,collections
H=Path(__file__).resolve().parent;

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mubench.config import activate,profile
activate(profile('chat','tool_augmented',transport='text'))
from run_text_case import Episode
from adapters.providers.chat.adapter import TextStream
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase
from experiments.tau_multiuser_voice.virtual_tools import create_virtual_environment
from experiments.tau_multiuser_voice.evaluator import _resolve_state_path,_contains_subset,_values_equal,build_reference_final_state
from experiments.tau_multiuser_voice.evaluation_policy import business_state_projection
rows=[]
for p in sorted((ROOT/'cases/tool_augmented').glob('*.json')):
 c=MultiUserVoiceCase.model_validate(json.loads(p.read_text()));env=create_virtual_environment(c.environment);errors=[];calls=[]
 for a in c.expected_tool_actions:
  if not (a.must_execute and a.allowed):continue
  try:
   result=env.invoke(a.tool_name,a.actor_id,**copy.deepcopy(a.argument_subset));calls.append({'name':a.tool_name,'actor':a.actor_id,'arguments':a.argument_subset,'executed':True})
  except Exception as e:errors.append({'type':'reference_execution_error','tool':a.tool_name,'error':str(e)})
 for a in c.state_assertions:
  ok,value=_resolve_state_path(env.snapshot(),a.path)
  if a.operator=='eq':passed=ok and _values_equal(value,a.expected,c)
  elif a.operator=='contains':passed=ok and _contains_subset(value,a.expected,c)
  elif a.operator=='exists':passed=ok
  elif a.operator=='not_exists':passed=not ok
  else:continue
  if a.critical and not passed:errors.append({'type':'reference_final_assertion','id':a.assertion_id,'actual':value,'expected':a.expected})
 expected=build_reference_final_state(c)
 if expected is not None and business_state_projection(c,env.snapshot())!=business_state_projection(c,expected):
  errors.append({'type':'reference_business_projection_mismatch'})
 rows.append({'case_id':c.case_id,'passed':not errors,'errors':errors,'reference_calls':calls})
out={'case_count':len(rows),'passed':sum(r['passed'] for r in rows),'failed':sum(not r['passed'] for r in rows),'error_types':dict(collections.Counter(e['type'] for r in rows for e in r['errors'])),'scope':'Synthetic all-knowing fixture/terminal-state feasibility only. Does not demonstrate model-knowable sources, lawful timeline or semantic judge correctness. No API calls.','cases':rows}
(Path(__import__('os').environ['MUBENCH_TEST_OUTPUT'])/'reference_checks.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in out.items() if k!='cases'}))
