"""Exercise the real HTTP/session transport with an explicitly non-model fixture."""
from pathlib import Path
import json,os,socket,subprocess,sys,tempfile,time,unittest,urllib.request,secrets
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mubench.config import activate,profile
activate(profile('native','tool_free'))
from experiments.mu_bench.logical_time.openweight.client import SessionClient
class NativeTransportTests(unittest.TestCase):
 def test_state_pause_resume_and_finish(self):
  with tempfile.TemporaryDirectory() as temporary:
   d=Path(temporary);token=secrets.token_urlsafe(32);(d/'key').write_text(token);(d/'config.json').write_text(json.dumps({'model':'test_fake','native_quantum_ms':80}))
   with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
   env=dict(os.environ,NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
   args=[sys.executable,str(ROOT/'runtime/core/hardened_openweight_service.py'),'--legacy-src',str(ROOT/'runtime/core/src'),'--native-config',str(d/'config.json'),'--allowed-root',str(d),'--output-root',str(d/'sessions'),'--api-key-file',str(d/'key'),'--host','127.0.0.1','--port',str(port),'--max-sessions','1','--timeout-seconds','30','--allow-test-backend']
   with (d/'server.log').open('w') as log:
    process=subprocess.Popen(args,stdout=log,stderr=log,env=env)
    try:
     url=f'http://127.0.0.1:{port}';deadline=time.monotonic()+20
     opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
     while True:
      if process.poll() is not None:raise AssertionError((d/'server.log').read_text())
      try:
       with opener.open(urllib.request.Request(url+'/health',headers={'Authorization':'Bearer '+token}),timeout=1) as response:json.load(response)
       break
      except Exception:
       if time.monotonic()>deadline:raise
       time.sleep(.1)
     client=SessionClient(url,token,timeout_seconds=5);initial=client.start(seed=7,max_media_ms=800)
     self.assertEqual(initial['logical_ms'],0)
     tick=client.step_pcm(bytes(1280),0);self.assertEqual(tick['logical_ms'],80);self.assertEqual(len(tick['assistant_pcm16']),1280)
     client.pause();self.assertEqual(client.status()['logical_ms'],80);time.sleep(.05);self.assertEqual(client.status()['logical_ms'],80)
     client.resume();next_tick=client.step_pcm(bytes(1280),80);self.assertEqual(next_tick['logical_ms'],160)
     client.finish()
    finally:
     process.terminate()
     try:process.wait(timeout=15)
     except subprocess.TimeoutExpired:process.kill();process.wait()
if __name__=='__main__':unittest.main()
