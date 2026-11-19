import unittest,sys,importlib.util
from pathlib import Path
from copy import deepcopy
p=Path(__file__).resolve().parents[1]/'runtime/ablations/ablation_core.py'
spec=importlib.util.spec_from_file_location('retry_core',p);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
VALID={'goal':{'value':'continue','evidence':'Please continue.'},'operation':'request','arguments':[],'constraints':[]}
OK={'faithful':True,'complete':True,'no_solution_or_extra_facts':True,'reason':'faithful'}
BAD={'faithful':True,'complete':False,'no_solution_or_extra_facts':True,'reason':'specific missing constraint explanation'}
class Backend:
 def __init__(self,seq):self.seq=list(seq);self.calls=[]
 def generate_json(self,messages):
  self.calls.append(deepcopy(messages));x=self.seq.pop(0)
  if isinstance(x,Exception):raise x
  return deepcopy(x)
class RetryTests(unittest.TestCase):
 def test_second_generation_sees_full_review_feedback(self):
  b=Backend([VALID,BAD,VALID,OK]);a=[]
  core.compile_semantics(b,'Please continue.',{},a)
  self.assertIn('specific missing constraint explanation',str(b.calls[2]))
  self.assertIn('previous_candidate',str(b.calls[2]));self.assertFalse(a[-1]['forced_accept'])
 def test_third_candidate_is_directly_adopted(self):
  b=Backend([VALID,BAD,VALID,BAD,VALID]);a=[]
  self.assertEqual(core.compile_semantics(b,'Please continue.',{},a),VALID)
  self.assertEqual(len(b.calls),5);self.assertTrue(a[-1]['forced_accept']);self.assertIsNone(a[-1]['independent_review_passed'])
 def test_parse_errors_feedback_and_third_adoption(self):
  bad=dict(VALID,operation='invalid');b=Backend([bad,bad,bad]);a=[]
  self.assertEqual(core.compile_semantics(b,'Please continue.',{},a),bad)
  self.assertIn('invalid semantic operation',str(b.calls[1]));self.assertFalse(a[-1]['validation_passed']);self.assertTrue(a[-1]['forced_accept'])
 def test_invalid_json_final_relays_actual_utterance(self):
  b=Backend([ValueError('invalid JSON')]*3);a=[]
  r=core.compile_semantics(b,'Please continue.',{},a)
  self.assertEqual(r['goal']['value'],'Please continue.');self.assertTrue(a[-1]['verbatim_fallback'])
 def test_first_accept_and_service_error(self):
  b=Backend([VALID,OK]);a=[];core.compile_semantics(b,'Please continue.',{},a);self.assertEqual(len(b.calls),2)
  with self.assertRaises(RuntimeError):core.compile_semantics(Backend([RuntimeError('HTTP402')]),'Please continue.',{},[])
if __name__=='__main__':unittest.main()
