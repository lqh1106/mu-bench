"""Run the matching model's stateful service implementation."""
from pathlib import Path
import argparse,sys,os,json,subprocess
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--config',required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--token-file',type=Path,required=True);p.add_argument('--endpoint-file',type=Path,required=True);p.add_argument('--host',default='127.0.0.1');p.add_argument('--advertise-host',default='127.0.0.1');p.add_argument('--port',type=int,default=19000);p.add_argument('--max-sessions',type=int,default=1);a=p.parse_args()
a.output_root=a.output_root.resolve();a.output_root.mkdir(parents=True,exist_ok=True)
src=ROOT/'adapters/native'/a.model/'src'
if not src.exists():src=ROOT/'runtime/core/src'
os.environ['PYTHONPATH']=os.pathsep.join([str(ROOT/'compat'),str(ROOT/'runtime/core/src'),str(ROOT/'judges/common/src'),str(ROOT)])+os.pathsep+os.environ.get('PYTHONPATH','')
args=[sys.executable,str(ROOT/'runtime/core/hardened_openweight_service.py'),'--legacy-src',str(src),'--native-config',str(Path(a.config).resolve()),'--allowed-root',str(a.output_root),'--output-root',str(a.output_root),'--api-key-file',str(a.token_file.resolve()),'--host',a.host,'--port',str(a.port),'--max-sessions',str(a.max_sessions),'--timeout-seconds','6000']
# Session logs and audio must lie beneath allowed-root on a shared filesystem.
cfg=json.loads(Path(a.config).read_text());source_path=cfg.get('backend_args',{}).get('repo_path')
if source_path:
 roots=[source_path,str(Path(source_path)/'moshi'),str(Path(source_path)/'third_party/Step-Audio2')]
 os.environ['PYTHONPATH']=os.pathsep.join(roots)+os.pathsep+os.environ.get('PYTHONPATH','')
sample={'minicpm':1000,'raon':80,'personaplex':80,'moshi':80,'freeze_omni':160,'flm_audio':80,'voila':40,'lychee':400}
# Actual quantum is checked against the backend, so discover it from its native
# session class in the documented service endpoint rather than guessing remotely.
q=sample[a.model]
a.endpoint_file.parent.mkdir(parents=True,exist_ok=True)
a.endpoint_file.write_text(json.dumps({'schema_version':'mu-bench-worker-endpoint-v2','protocol':'mu-single-session-logical-v14.0','model':a.model,'native_quantum_ms':q,'base_url':f'http://{a.advertise_host}:{a.port}','api_key_file':str(a.token_file.resolve()),'timeout_seconds':6000},indent=2)+'\n')
os.execv(sys.executable,args)
