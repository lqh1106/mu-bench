from pathlib import Path
import os,subprocess,sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mubench.config import environment,profile
rows=[]
for provider,subset,arm,entry in [('native','tool_free',None,'run_mu_logical_voice.py'),('moss','tool_free',None,'run_mu_logical_voice.py'),('openai','tool_augmented',None,'run_mu_logical_voice.py'),('seeduplex','tool_free',None,'run_mu_logical_voice.py'),('seeduplex','tool_augmented',None,'run_mu_logical_voice.py'),('native','tool_free','serial','run_audio_ablation.py'),('moss','tool_free','serial','run_audio_ablation.py')]:
 rt=profile(provider,subset,arm);args=[sys.executable,str(rt/entry),'--help']
 if arm:args.extend(['--ablation-arm',arm])
 p=subprocess.run(args,cwd=ROOT,env=environment(rt,subset,arm,provider),capture_output=True,text=True)
 if p.returncode:raise RuntimeError(str(rt.relative_to(ROOT))+': '+p.stderr[-3000:])
 rows.append(str(rt.relative_to(ROOT)))
print(json.dumps({'entrypoint_imports_passed':rows},indent=2))
