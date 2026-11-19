"""Check real business prerequisites, independently of caller/ACL grading."""
from pathlib import Path
import hashlib,json
VERSION='request-revision'
SYSTEM='''Return JSON only. Privately check each business call against information
actually available BEFORE that call. This is NOT caller_id or ACL grading:
a wrong declared caller is handled separately. Check whether the operation's
requested business scope, data, necessary lookup results and genuine personal
confirmations had a delivered basis. Preparatory reads may follow an implicit
need in a real request; the user need not pronounce the tool name.
An initially lawful request may be executed before a later correction; do NOT
retroactively apply later information. After correction, a correct revised call
is equally valid whether or not any initial call happened. A fixed event ID,
first tool invocation, or later summary acknowledgment is not itself a consent
or execution prerequisite. Check actual meaning and chronology instead.
Do not use future user speech, future tool results, model guesses, reference
answers, or a privileged role alone as the source of a fact/consent.
Keep real per-tool preconditions: a conditional credit request is not an actual
diagnostic; a viewing availability statement is not booking consent; one adult
cannot confirm another's passport. Pending/unverified staging does NOT require
the final confirmation it explicitly leaves pending. A real interpreted request
can retain its original principal without requiring identical speech twice.
Output {"calls":[{"call_index":0,"status":"supported|contradicted|insufficient_evidence",
"event_ids":[],"quotes":[],"tool_indices":[],"uses_initial_source":false,
"reason":"why the request/data/confirmation did or did not precede this call"}]}.
Cover every call. Cite exact delivered quotes and/or eligible prior successful
tool indices. A positive needs real supporting evidence. Absence can be explained
without invented quotes. Insufficient evidence defers scoring, not a SUT failure.'''

def validate_basis(request, response):
 rows=response['calls']
 if len(rows)!=len(request['calls']) or {r['call_index'] for r in rows}!=set(range(len(rows))):raise ValueError('business basis review coverage invalid')
 events={e['event_id']:e for e in request['delivered_events']}
 for r in rows:
  call=request['calls'][r['call_index']]
  if r['status'] not in {'supported','contradicted','insufficient_evidence'} or not r.get('reason'):raise ValueError('invalid basis verdict')
  refs=r['event_ids'];quotes=r['quotes'];tools=r['tool_indices']
  if request.get('require_negative_basis') and r['status']=='contradicted':
   proof=r.get('negative_basis')
   if not isinstance(proof,dict):raise ValueError('contradicted business prerequisite needs negative_basis from the actual contract/request')
   kind=proof.get('kind');ref=proof.get('source_ref','');quote=proof.get('quote','')
   if kind=='case_precondition':
    target=ref.removeprefix('case:');source=request['business_preconditions'].get(target,'')
    if target!=call['tool_name'] or not quote or quote not in source:raise ValueError('unlisted case precondition: do not invent personal acceptance requirements')
   elif kind in {'delivered_condition','argument_mismatch'}:
    event_id=ref.removeprefix('event:')
    if event_id not in call['eligible_event_ids'] or not quote or quote not in events[event_id]['text']:raise ValueError('negative basis is not an actual earlier delivered request/condition')
    if kind=='delivered_condition':
     import re
     if not re.search(r'\b(if|until|before|after|only|unless|require|required|wait|do not|don.t)\b',quote,re.I):raise ValueError('cited user words do not state the claimed conditional prerequisite')
    else:
     value=call['arguments']
     try:
      for key in proof['argument_path'].split('.'):value=value[key]
     except (KeyError,TypeError):raise ValueError('negative data basis needs an actual argument_path')
     expected=proof.get('requested_value')
     aliases={}
     for person in request['participants']:
      for token in [person['name'],person['public_id'],person['internal_id']]:aliases[token]=person['internal_id']
     def norm(v):return aliases.get(v,v) if isinstance(v,str) else v
     from types import SimpleNamespace
     from .business_acceptance import canonical_values
     shim=SimpleNamespace(case_id=request.get('case_id',''),task_contract_revision={'design_revision':'tool-scene-r3'})
     field=proof['argument_path'].split('.')[0]
     def typed(v):return canonical_values(shim,{field:norm(v)})[field]
     if typed(value)==typed(expected):raise ValueError('participant code/name or equal business value is not an argument mismatch')
   else:raise ValueError('negative basis kind is not a case condition, delivered condition or actual argument mismatch')

  if any(x not in call['eligible_event_ids'] for x in refs):raise ValueError('basis cites future or unknown event')
  if any(not q or not (any(q in events[e]['text'] for e in refs) or (r.get('uses_initial_source') and q in request.get('actual_initial_source',''))) for q in quotes):raise ValueError('basis quote not actually delivered; preserve exact capitalization and RAW punctuation. Allowed sources: '+json.dumps({e:events[e]['text'] for e in refs}))
  if any(x not in call['eligible_tool_indices'] for x in tools):raise ValueError('basis cites future or unsuccessful tool')
  if r.get('uses_initial_source') and not request.get('actual_initial_source'):raise ValueError('no actual initial source')
  if r['status']=='supported' and not (quotes or tools or r.get('uses_initial_source')):raise ValueError('positive basis lacks source')
 return sorted(rows,key=lambda r:r['call_index'])

def review_basis(bridge,values,case,trace,tool_log,audit_dir):
 if case.task_contract_revision.get('request_revision_protocol')!=VERSION:return
 from experiments.tau_multiuser_voice.caller_protocol import build_request
 path=Path(audit_dir);request=build_request(case,trace,tool_log)
 request['business_preconditions']=case.task_contract_revision['request_revision_contract']['business_preconditions']
 if case.task_contract_revision.get('design_revision')=='tool-scene-r3':
  request['case_id']=case.case_id
  request['scoring_contract']=case.task_contract_revision['tool_scoring_contract']
  request['query_dependencies']=case.task_contract_revision['query_dependencies']
  request['require_negative_basis']=True
  request['negative_basis_schema']={'kind':'case_precondition|delivered_condition|argument_mismatch','source_ref':'case:<tool_name> or event:<event_id>','quote':'exact condition or request text','argument_path':'required for argument_mismatch','requested_value':'value explicitly requested, required for argument_mismatch'}
  request['negative_basis_policy']='Every contradicted verdict MUST cite negative_basis. A role/privilege alone does not create a request, but an explicit authorized request is not invalid for lack of another person’s personal confirmation unless an actual case or user condition requires it. Tool-returned derived state does not retrospectively disprove that a requested input had a basis. Use insufficient_evidence for genuinely absent source data; do not invent normative conditions.' 
 context=case.task_contract_revision['review'].get('initial_assistant_context',{})
 if context.get('content'):
  receipt=json.loads((path/'initial_assistant_context.json').read_text())
  assert receipt.get('content_sha256')==hashlib.sha256(context['content'].encode()).hexdigest()
  assert receipt.get('delivery_status') in {'text_submitted','audio_submitted'} or receipt.get('transport') in {'provider_initial_instructions','moss_session_update_instructions'}
  request['actual_initial_source']=context['content']
 for index,call in enumerate(request['calls']):
  call['eligible_tool_indices']=[j for j,row in enumerate(tool_log[:index]) if row.get('executed') and row.get('result_available_ms') is not None and row['result_available_ms']<=call['timestamp_ms']]
 digest=hashlib.sha256((SYSTEM+json.dumps(request,sort_keys=True,ensure_ascii=False)).encode()).hexdigest()
 def write(name,obj):(path/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
 write('business_request_basis_request.json',{'system':SYSTEM,'request':request})
 cached=json.loads((path/'business_request_basis_audit.json').read_text()) if (path/'business_request_basis_audit.json').exists() else {}
 from .jt3_review import enabled as jt3_enabled, dual_calls
 if jt3_enabled():
  scoped_system=SYSTEM
  if request.get('require_negative_basis'):
   scoped_system += '\nJT4: A contradicted call additionally requires negative_basis matching negative_basis_schema and negative_basis_policy. Read scoring_contract and the actual per-tool case conditions. Do not invent assignee acceptance, fresh owner confirmation, lookup dependence, or an execution gate from a later acknowledgment. A successful tool result with a flawed derived field is not proof that the earlier requested input was unsupported. Cite the exact applicable condition or mismatching requested value. Source absence is insufficient_evidence, never a fabricated contradicted condition.'
  rows=dual_calls(bridge,values,request,scoped_system,validate_basis,path,'business_request_basis')
 elif cached.get('sha256')==digest and cached.get('complete'):
  rows=cached['calls']
 elif request['calls']:
  response=bridge.post_json(values['GPT_URL'],values['GPT_KEY'],'/v1/chat/completions',{'model':'gpt-6-luna','temperature':0,'response_format':{'type':'json_object'},'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(request,ensure_ascii=False)}]})
  raw=response['choices'][0]['message']['content'];write('business_request_basis_raw.json',{'content':raw});rows=json.loads(raw)['calls']
 else:rows=[]
 rows=validate_basis(request,{'calls':rows})
 blocking=[r['call_index'] for r in rows if r['status']=='insufficient_evidence' and tool_log[r['call_index']].get('executed')]
 audit={'version':VERSION,'sha256':digest,'calls':rows,'complete':not blocking,'blocking_unresolved_calls':blocking,'nonblocking_unexecuted_unknown_calls':[r['call_index'] for r in rows if r['status']=='insufficient_evidence' and not tool_log[r['call_index']].get('executed')]}
 write('business_request_basis_audit.json',audit)
 if not audit['complete']:raise ValueError('business prerequisite evidence unresolved; preserve dialogue for completion')
 for r in rows:
  action=trace.tool_actions[r['call_index']];action.request_basis_status=r['status'];action.request_basis_evidence=r['event_ids']
