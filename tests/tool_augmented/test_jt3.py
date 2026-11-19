import unittest,sys,os,tempfile,json
from types import SimpleNamespace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from mubench.config import activate
activate(ROOT/'runtime/runners/tool_augmented_text')
os.environ['MUBENCH_TOOL_JUDGE_VERSION']='jt3'
from experiments.tau_multiuser_voice import jt3_review as j, evaluation_policy as e
class Bridge:
 def __init__(self, rows):self.rows=iter(rows);self.requests=[]
 def post_json(self,*a):
  self.requests.append(a[-1]);return {'choices':[{'message':{'content':json.dumps(next(self.rows))}}]}
class Tests(unittest.TestCase):
 def validate(self,req,res):
  if res.get('calls',[{}])[0].get('quote')=='bad':raise ValueError('not raw quotation')
  return res['calls']
 def test_blind_and_arb(self):
  b=Bridge([{'calls':[{'call_index':0,'status':'supported'}]},{'calls':[{'call_index':0,'status':'contradicted'}]},{'calls':[{'call_index':0,'status':'contradicted'}]}])
  rows=j.dual_calls(b,{}, {},'',self.validate,None,'test') if False else j.dual_calls(b,{'GPT_URL':'x','GPT_KEY':'x'},{},'',self.validate,None,'test')
  self.assertEqual(rows[0]['status'],'contradicted');self.assertNotIn('anonymous_reviews',b.requests[1]['messages'][1]['content']);self.assertIn('anonymous_reviews',b.requests[2]['messages'][1]['content'])
 def test_repair_is_not_arbitration(self):
  b=Bridge([{'calls':[{'call_index':0,'status':'supported','quote':'bad'}]},{'calls':[{'call_index':0,'status':'supported'}]},{'calls':[{'call_index':0,'status':'supported'}]}]);j.dual_calls(b,{'GPT_URL':'x','GPT_KEY':'x'},{},'',self.validate,None,'test');self.assertIn('repair_only',b.requests[1]['messages'][2]['content'])
 def test_cache(self):
  with tempfile.TemporaryDirectory() as t:
   b=Bridge([{'calls':[]},{'calls':[]}]);j.dual_calls(b,{'GPT_URL':'x','GPT_KEY':'x'},{},'',lambda a,b:b['calls'],t,'test');j.dual_calls(b,{'GPT_URL':'x','GPT_KEY':'x'},{},'',lambda a,b:b['calls'],t,'test');self.assertEqual(len(b.requests),2)
 def temporal(self,passed,purpose,time):
  original=e.policy_for;e.policy_for=lambda c:{'pacing_queries':[{'tool_name':'get_x'}]}
  try:return e.cited_result_temporal_check(SimpleNamespace(task_contract_revision={}),{'passed':passed,'assistant_evidence':[{'source_id':'a'}],'tool_evidence':[{'source_id':'t','purpose':purpose}]},{'a':{'source_id':'a','kind':'assistant_text','start_ms':10,'end_ms':15},'t':{'source_id':'t','kind':'executed_tool_result','tool_name':'get_x','result_available_ms':time,'result_availability_basis':'explicit_text_tool_delivery'}})
  finally:e.policy_for=original
 def test_prior_future_rejected(self):self.assertTrue(self.temporal(True,'prior_basis',20)[0])
 def test_counterevidence_future_permitted(self):self.assertFalse(self.temporal(False,'counterevidence',20)[0])
 def test_later_corroboration_not_prior_basis(self):self.assertFalse(self.temporal(True,'final_state_corroboration',20)[0])
 def test_acknowledged_equal_valid(self):self.assertFalse(self.temporal(True,'prior_basis',10)[0])
class MoreTests(unittest.TestCase):
 def test_unique_source_id_repair(self):
  p={'passed':True,'assistant_evidence':[{'source_id':'wrong','quote':'Actual statement.'}]}
  j.bind_raw_format_quotes(p,{'sources':[{'source_id':'a','kind':'assistant_text','text':'Actual statement.'}]})
  self.assertEqual(p['assistant_evidence'][0]['source_id'],'a')
 def test_ambiguous_source_not_chosen(self):
  p={'passed':True,'assistant_evidence':[{'source_id':'wrong','quote':'Yes.'}]}
  j.bind_raw_format_quotes(p,{'sources':[{'source_id':'a','kind':'assistant_text','text':'Yes.'},{'source_id':'b','kind':'assistant_text','text':'Yes.'}]})
  self.assertEqual(p['assistant_evidence'][0]['source_id'],'wrong')
 def test_primary_resume_no_api(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);packet={'judge_version':'jt3','tool_evidence_roles':j.RULES}
   (root/'semantic_judge_request.json').write_text(json.dumps(packet))
   (root/'jt3_primary_raw_1.json').write_text(json.dumps({'content':'{"saved_first_verdict": true}'}))
   class NoAPI:
    def post_json(self,*a):raise AssertionError('Completed primary was regenerated')
   with patch('experiments.tau_multiuser_voice.judge_evidence.validate_and_bind_judge_payload',lambda c,k,v,cat:(v,{'evaluation_valid':True})), patch('experiments.tau_multiuser_voice.full_review.full_review',lambda b,v,c,k,p,cat,r,d,w:p):
    final=j.semantic_review(NoAPI(),{},None,{'goals':[]},{'sources':[]},packet,root,lambda *a:None)
   self.assertTrue(final['saved_first_verdict'])
class RelevanceTests(unittest.TestCase):
 def test_unknown_denied_attempt_does_not_block_business_score(self):
  self.check(False)
 def test_unknown_executed_prerequisite_still_blocks(self):
  with self.assertRaises(ValueError):self.check(True)
 def check(self,executed):
  from unittest.mock import patch
  from experiments.tau_multiuser_voice.business_request_basis import review_basis,VERSION
  case=SimpleNamespace(task_contract_revision={'request_revision_protocol':VERSION,'request_revision_contract':{'business_preconditions':{}},'review':{}})
  trace=SimpleNamespace(tool_actions=[SimpleNamespace()]);request={'calls':[{'timestamp_ms':100}],'delivered_events':[]}
  rows=[{'call_index':0,'status':'insufficient_evidence','reason':'request relevance unclear','event_ids':[],'quotes':[],'tool_indices':[],'uses_initial_source':False}]
  with tempfile.TemporaryDirectory() as t,patch('experiments.tau_multiuser_voice.caller_protocol.build_request',return_value=request),patch('experiments.tau_multiuser_voice.jt3_review.dual_calls',return_value=rows):
   review_basis(None,None,case,trace,[{'executed':executed}],t)
   a=json.loads((Path(t)/'business_request_basis_audit.json').read_text())
   self.assertTrue(a['complete']);self.assertEqual(a['nonblocking_unexecuted_unknown_calls'],[0])
class CriterionIdentityTests(unittest.TestCase):
 def run_binding(self,index):
  payload={'goal_results':[{'goal_id':'g','completion_checks':[{'criterion':'review what needs reconsidering','passed':False,'reason':'unchanged','assistant_evidence':[]}]}]}
  original=json.loads(json.dumps(payload))
  b=Bridge([{'bindings':[{'goal_id':'g','check_index':0,'criterion_index':index}]}])
  result=j.repair_criterion_identity(b,{'GPT_URL':'x','GPT_KEY':'x'},payload,{'goals':[{'goal_id':'g','completion_criteria':['review what needs reconsideration']} ]},None,'test')
  self.assertEqual(payload,original)
  self.assertEqual(result['goal_results'][0]['completion_checks'][0],dict(original['goal_results'][0]['completion_checks'][0],criterion='review what needs reconsideration'))
 def test_labels_only(self):self.run_binding(0)
 def test_unknown_binding_not_forced(self):
  with self.assertRaises(ValueError):self.run_binding(None)
 def test_out_of_range_rejected(self):
  with self.assertRaises(ValueError):self.run_binding(2)
if __name__=='__main__':unittest.main()


