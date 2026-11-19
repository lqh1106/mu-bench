"""Finite business equivalences and independent owner-scoped operations."""
from copy import deepcopy
import os

def revised(case):return case.task_contract_revision.get('design_revision') in {'tool-scene-r2','tool-scene-r3'} or os.environ.get('MUBENCH_TOOL_CASE_MECHANISM') in {'tr2','tr3'}
def aggregate_matches(case,expected,trace,contains):
 if not revised(case) or not expected.allowed:return []
 from experiments.tau_multiuser_voice.business_effects import MAP_FIELDS
 mapkeys=set(expected.argument_subset)&MAP_FIELDS
 if not mapkeys:return []
 # A fixed meaningful time boundary is not bypassed by combining calls.
 if expected.after_event_id or expected.before_event_id:return []
 calls=[a for a in trace.tool_actions if a.executed and a.tool_name==expected.tool_name and a.resource_id==expected.resource_id and a.request_basis_status=='supported']
 merged={}
 # Existing actual owner records can already satisfy part of a compound goal.
 aliases={'votes_by_user':'votes','answers_by_user':'answers','items_by_owner':'lists','gigabytes_by_user':'data_pass_gigabytes_by_user','reports_by_user':'reports'}
 for collection in case.environment.initial_state.values():
  if not isinstance(collection,dict):continue
  for record in collection.values():
   if not isinstance(record,dict):continue
   b=record.get('business_state',{})
   for key in mapkeys:
    value=b.get(key,b.get(aliases.get(key,'')))
    if isinstance(value,dict):merged.setdefault(key,{}).update(deepcopy(value))
 for a in calls:
  for k,v in a.arguments.items():
   if k in MAP_FIELDS and isinstance(v,dict):merged.setdefault(k,{}).update(deepcopy(v))
   else:merged[k]=deepcopy(v)
 return calls if contains(merged,expected.argument_subset,case) else []

def transient_assertion(case,assertion):
 return revised(case) and assertion.path.split('.')[-1] in {'last_modified_by','last_operation','last_transaction_id'}

def project(case,state):
 if not revised(case):return state
 out=deepcopy(state);scene=case.case_id.split('__')[0]
 # Explicit redundant request fields are not independent business effects.
 aliases={
 'meeting_04_quiet_dissenter':['participant_user_id','stance'],
 'shared_home_03_grocery_threads':['live_excluded_items'],
 'education_04_project_roles':['live_accepts','live_rejects']}
 for collection in out.values():
  if not isinstance(collection,dict):continue
  for resource in collection.values():
   if not isinstance(resource,dict):continue
   b=resource.get('business_state')
   if not isinstance(b,dict):continue
   for k in aliases.get(scene,[]):b.pop(k,None)
   if 'owner' in b:
    names={u.display_name:u.user_id for u in case.users};names.update({case.public_participant_id(u.user_id):u.user_id for u in case.users})
    b['owner']=names.get(b['owner'],b['owner'])
   if scene=='telecom_03_router_restart_pushback':b.pop('reason',None)
   if scene=='airline_01_split_rebooking':
    moved=b.get('rebook_user_ids',[]);keep=b.get('keep_original_user_ids',[])
    if not set(moved)&set(keep):b['keep_original_user_ids']=sorted(set(resource.get('participant_user_ids',[]))-set(moved))
   if scene=='shared_home_03_grocery_threads':
    b['excluded_items_by_owner']={k:v for k,v in b.get('excluded_items_by_owner',{}).items() if v}
 return out


def v3(case):
 return case.task_contract_revision.get('design_revision')=='tool-scene-r3'

def canonical_values(case,value,path=()):
 """Normalize only declared meaning-bearing business fields."""
 if not v3(case):return deepcopy(value)
 if isinstance(value,dict):return {k:canonical_values(case,v,path+(k,)) for k,v in value.items()}
 if isinstance(value,list):return [canonical_values(case,v,path+('[]',)) for v in value]
 scene=case.case_id.split('__')[0]
 if isinstance(value,str) and scene=='education_01_fraction_answers' and any(k in path for k in ('answers_by_user','answers')):
  from fractions import Fraction
  import re
  text=value.strip().casefold().replace('-',' ')
  words={'one half':'1/2','a half':'1/2','half':'1/2','two thirds':'2/3','three quarters':'3/4','three fourths':'3/4'}
  text=words.get(text,text)
  if re.fullmatch(r'[+-]?\d+\s*/\s*\d+|[+-]?\d+(?:\.\d+)?',text):
   try:
    q=Fraction(text.replace(' ',''));return str(q.numerator)+'/'+str(q.denominator)
   except (ValueError,ZeroDivisionError):return value
 if isinstance(value,str) and path and path[-1] in {'wheelchair_access','step_free_entry'}:
  return {'required':'mandatory','mandatory':'mandatory','optional':'optional'}.get(value.strip().casefold(),value)
 return deepcopy(value)

_previous_project=project
def project(case,state):
 out=_previous_project(case,state)
 if not v3(case):return out
 out.pop('audit_events',None)
 for collection in out.values():
  if not isinstance(collection,dict):continue
  for record in collection.values():
   if not isinstance(record,dict):continue
   for field in ('last_modified_by','last_operation','last_transaction_id'):record.pop(field,None)
 return canonical_values(case,out)


def result_fact_paths(value,prefix=''):
 paths=[]
 if isinstance(value,dict):
  for key,item in value.items():
   name=(prefix+'.'+key) if prefix else key
   paths.append(name);paths.extend(result_fact_paths(item,name))
 return sorted(set(paths))

def annotate_tool_facts(case,trace,tool_log):
 """Bind actual completed results to the evaluator's compact source footprint."""
 if not v3(case):return
 import hashlib,json
 for index,action in enumerate(trace.tool_actions):
  if index>=len(tool_log):raise ValueError('actual tool result log missing for dependency evidence')
  row=tool_log[index]
  if row.get('tool_name')!=action.tool_name or row.get('resource_id')!=action.resource_id:raise ValueError('tool result evidence does not match traced call')
  action.returned_fact_paths=[];action.returned_result_sha256=None
  result=row.get('model_visible_result',row.get('result'))
  if action.executed and row.get('executed') and isinstance(result,dict) and not result.get('error'):
   action.returned_fact_paths=result_fact_paths(result)
   action.returned_result_sha256=hashlib.sha256(json.dumps(result,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
