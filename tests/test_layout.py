from pathlib import Path
import json,unittest,subprocess,sys,os
ROOT=Path(__file__).resolve().parents[1]
class LayoutTests(unittest.TestCase):
 def test_one_shared_ablation_catalog(self):
  self.assertEqual(sorted(p.name for p in (ROOT/'cases/ablations').iterdir()),['serial','serial_identity','serial_identity_semantics'])
  paths=list((ROOT/'cases/ablations').rglob('case.json'));self.assertEqual(len(paths),1176)
  for p in paths:self.assertNotIn('scoring_sha256',json.loads(p.with_name('manifest.json').read_text()))
 def test_all_provider_adapters_are_visible(self):
  for provider in ['openai','grok','gemini','moss','chat']:
   self.assertTrue((ROOT/'adapters/providers'/provider).is_dir())
 def test_audio_and_text_evidence_profiles(self):
  for transport,expected in [('audio','assistant_speech'),('text','assistant_text')]:
   code="from mubench.config import activate,profile;activate(profile('openai','tool_augmented',transport='"+transport+"'));from experiments.tau_multiuser_voice.judge_evidence import build_evidence_catalog;print(build_evidence_catalog([{'role':'assistant','text':'hello','start_ms':0,'end_ms':1,'committed':True}])['sources'][0]['kind'])"
   env=dict(os.environ,PYTHONPATH=str(ROOT))
   p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
   self.assertEqual(p.stdout.strip(),expected)
 def test_compiled_case_independent_of_provider(self):
  for provider in ['native','moss']:
   code="from pathlib import Path;from types import SimpleNamespace;from mubench.config import activate,profile;activate(profile('"+provider+"','tool_free','serial'));from experiments.tau_multiuser_voice.cases import CaseCatalog;from ablation_runtime import prepare;case=CaseCatalog.get('airline_04_gate_announcement_collision__g-small__a-clean__c-strong');args=SimpleNamespace(ablation_arm='serial',openweight_endpoint=Path('endpoint.json'),output_dir=Path('unused_output'));changed,experiment=prepare(case,args);print(changed.case_id)"
   env=dict(os.environ,PYTHONPATH=str(ROOT),MUBENCH_PROVIDER=provider,MUBENCH_ABLATION_ARM='serial')
   p=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
   self.assertEqual(p.stdout.strip(),'airline_04_gate_announcement_collision__g-small__a-clean__c-strong')
if __name__=='__main__':unittest.main()
