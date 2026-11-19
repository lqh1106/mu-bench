from pathlib import Path
import os,json,sys,subprocess,tempfile,unittest,shutil
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mubench.config import activate,profile
activate(profile('chat','tool_augmented',transport='text'))
from adapters.providers.chat.adapter import TextStream
class Chunks:
 def __init__(self,rows):self.rows=rows
 def stream(self,messages,tools):yield from self.rows
class ReleaseTests(unittest.TestCase):
 def test_discarded_output_never_becomes_history(self):
  history=[];calls=[]
  s=TextStream(Chunks([({'content':'ABCDE'},None),({'content':'future'},'stop')]),history,[])
  r=s.response(lambda ch,p:len(p)==3,calls.extend)
  self.assertEqual(history,[{'role':'assistant','content':'ABC'}]);self.assertEqual(r['discarded_buffer_suffix'],'DE');self.assertFalse(calls)
 def test_partial_tool_is_not_executed(self):
  history=[];calls=[]
  s=TextStream(Chunks([({'tool_calls':[{'index':0,'id':'c1','function':{'name':'write','arguments':'{"x":'}}],'content':'ABCDE'},None)]),history,[])
  s.response(lambda c,p:len(p)==2,calls.extend)
  self.assertFalse(calls);self.assertNotIn('tool_calls',history[0])
 def test_completed_tool_survives_later_interruption(self):
  history=[];calls=[]
  s=TextStream(Chunks([({'tool_calls':[{'index':0,'id':'c1','function':{'name':'write','arguments':'{"x":'}}]},None),({'tool_calls':[{'index':0,'function':{'arguments':'1}'}}]},'tool_calls')]),history,[])
  s.response(lambda c,p:False,calls.extend)
  self.assertEqual(json.loads(calls[0]['function']['arguments']),{'x':1})
 def test_exactly_one_claim_per_case(self):
  import fcntl
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'lock'
   with p.open('a') as a,p.open('a') as b:
    fcntl.flock(a,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with self.assertRaises(BlockingIOError):fcntl.flock(b,fcntl.LOCK_EX|fcntl.LOCK_NB)
 def test_profile_selection(self):
  from mubench.config import profile,environment
  p=profile('moss','tool_free','serial_identity')
  e=environment(p,'tool_free','serial_identity')
  self.assertEqual(e['MUBENCH_ABLATION_ARM'],'serial_identity');self.assertEqual(e['MUBENCH_JUDGE_PROTOCOL'],'ablation_serial_identity_review')
  self.assertEqual(profile('native','tool_augmented',transport='text').name,'tool_augmented_text')
 def test_credential_values_are_not_bundled(self):
  import re
  for p in ROOT.rglob('*'):
   if not p.is_file() or p.suffix not in ['.py','.json','.md']:continue
   if '.git' in p.parts or 'outputs' in p.parts:continue
   s=p.read_text()
   self.assertIsNone(re.search(r'\b(?:sk-|xai-)[A-Za-z0-9_-]{24,}',s),str(p))
if __name__=='__main__':unittest.main()
