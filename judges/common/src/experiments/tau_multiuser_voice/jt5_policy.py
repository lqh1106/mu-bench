"""JT5 policy: case-insensitive citations and explicit third-generation acceptance."""
import os,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[5]
def enabled():return os.environ.get('MUBENCH_TOOL_JUDGE_VERSION') in {'jt5','jt6','jt7'}
def activate(path):
 if path is None:return
 policy=ROOT/'runtime/jt7_policy.json'
 if not policy.exists():policy=ROOT/'runtime/jt6_policy.json'
 if not policy.exists():policy=ROOT/'runtime/jt5_policy.json'
 if not policy.exists():return
 data=json.loads(policy.read_text());current=Path(path).resolve()
 if any(current==Path(folder).resolve() for folder in data.get('preserve_jt4_evidence_dirs',[])):return
 if os.environ.get('MUBENCH_TOOL_JUDGE_VERSION') in {'jt4','jt5','jt6','jt7'}:
  os.environ['MUBENCH_TOOL_JUDGE_VERSION']=data.get('default_version','jt5')
  p=Path(path);p.mkdir(parents=True,exist_ok=True)
  (p/'judge_execution_protocol.json').write_text(json.dumps({'judge_version':os.environ.get('MUBENCH_TOOL_JUDGE_VERSION','jt5'),'at':time.time(),'case_insensitive_quotes':True,'third_generation_direct_acceptance':True,'scoring_model':'gpt-6-luna','business_basis_unknown_policy':data.get('business_basis_unknown_policy')},indent=2))
def contains(quote,text):
 return quote.casefold() in text.casefold() if enabled() else quote in text
def accept_calls(request,payload,path,prefix,stage,error):
 rows=payload.get('calls') if isinstance(payload,dict) else None
 if not isinstance(rows,list) or {r.get('call_index') for r in rows}!=set(range(len(request['calls']))) or len(rows)!=len(request['calls']):raise ValueError('JT5 third generation has no usable complete calls schema')
 if any(r.get('status') not in {'supported','contradicted','insufficient_evidence'} for r in rows):raise ValueError('JT5 third generation has unusable status')
 if path is not None:
  p=Path(path);p.mkdir(parents=True,exist_ok=True)
  (p/(prefix+'_'+stage+'_jt5_third_accepted.json')).write_text(json.dumps({'judge_version':os.environ.get('MUBENCH_TOOL_JUDGE_VERSION','jt5'),'generation':3,'accepted_directly':True,'validation_passed':False,'validation_error':str(error),'verdicts_unchanged':True,'response':payload},ensure_ascii=False,indent=2))
 return sorted(rows,key=lambda r:r['call_index'])

def bind_case_only(request,response):
 """Restore raw case for already imported JT4 validators; never rewrite words."""
 if not enabled():return response
 from copy import deepcopy
 out=deepcopy(response);events={e['event_id']:e['text'] for e in request.get('delivered_events',[])}
 for row in out.get('calls',[]):
  texts=[events[e] for e in row.get('event_ids',[]) if e in events]
  if row.get('uses_initial_source'):texts.append(request.get('actual_initial_source',''))
  rebound=[]
  for q in row.get('quotes',[]):
   replacement=q
   if isinstance(q,str) and not any(q in t for t in texts):
    for t in texts:
     i=t.casefold().find(q.casefold())
     if i>=0 and t[i:i+len(q)].casefold()==q.casefold():replacement=t[i:i+len(q)];break
   rebound.append(replacement)
  row['quotes']=rebound
 return out
