"""Compile model-independent ablation cases and separate runtime fingerprints."""
from pathlib import Path
import hashlib,json,sys,os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mubench.config import activate,profile
os.environ['MUBENCH_ABLATION_ARM']='serial';activate(profile('native','tool_free','serial'))
from ablation_core import Experiment,digest,write
from experiments.tau_multiuser_voice.models import MultiUserVoiceCase
count=0
for arm in ('serial','serial_identity','serial_identity_semantics'):
 for path in sorted((ROOT/'cases/tool_free').glob('*.json')):
  case=MultiUserVoiceCase.model_validate(json.loads(path.read_text()));out=ROOT/'cases/ablations'/arm/case.case_id
  experiment=Experiment(arm,case.model_dump(mode='json'),out)
  experiment.manifest.update(task_success_policy='separate-authorization')
  write(out/'case.json',experiment.transformed);write(out/'manifest.json',experiment.manifest);count+=1
shared=['runtime/core','runtime/ablations','runtime/protocols/ablations','judges/common','judges/tool_free','judges/ablations','mubench/config.py']
for provider,additional in [('minicpm',['adapters/native/minicpm/ablation.py','adapters/native/minicpm/src']),('moss',['adapters/providers/moss'])]:
 files={}
 for name in shared+additional:
  p=ROOT/name
  for f in ([p] if p.is_file() else p.rglob('*')):
   if f.is_file() and f.suffix in ('.py','.json') and '__pycache__' not in f.parts:
    files[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
 manifest={'files':files,'scoring_sha256':digest(files),'task_success_policy':'separate-authorization','provider':provider,'shared_cases':'cases/ablations'}
 write(ROOT/'runtime/fingerprints'/f'{provider}.json',manifest)
print(json.dumps({'shared_cases':count,'runtime_fingerprints':['minicpm','moss']}))
