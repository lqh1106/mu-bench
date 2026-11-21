"""Transport checks: text bypass, ASR preservation, and unchanged defaults."""
from pathlib import Path
import importlib.util,json,os,sys,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('route_test_bridge',ROOT/'runtime/core/run_tau_voice_realtime_debug_20260909.py');bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
class Response:
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def read(self):return b'{"ok":true}'
class RoutingTests(unittest.TestCase):
 def send(self,payload,route='http://text.test/v1'):
  with patch.dict(os.environ,{'MUBENCH_TEXT_BASE_URL':route,'MUBENCH_TEXT_MODEL':'gpt-6-luna'}),patch.object(bridge.urllib.request,'urlopen',return_value=Response()) as send:
   bridge.post_json('http://speech.test/v1','test-token','/v1/chat/completions',payload)
   req=send.call_args.args[0]
   self.assertEqual(req.get_header('Authorization'),'Bearer test-token')
   return req.full_url,json.loads(req.data)
 def test_text_direct_same_payload_and_key(self):
  p={'model':'gpt-6-luna','messages':[{'role':'user','content':'Judge this evidence'}]};u,got=self.send(p);self.assertEqual(u,'http://text.test/v1/chat/completions');self.assertEqual(got,p)
 def test_asr_remains_on_speech_endpoint(self):
  u,_=self.send({'model':'gpt-audio-1.5','messages':[{'role':'user','content':[{'type':'input_audio','input_audio':{'data':'fixture','format':'wav'}}]}]});self.assertEqual(u,'http://speech.test/v1/chat/completions')
 def test_audio_payload_is_never_rerouted(self):
  u,_=self.send({'model':'gpt-6-luna','messages':[{'role':'user','content':[{'type':'input_audio','input_audio':{'data':'fixture'}}]}]});self.assertEqual(u,'http://speech.test/v1/chat/completions')
 def test_unrelated_text_model_is_not_redirected(self):
  u,_=self.send({'model':'qwen-text-sut','messages':[]});self.assertEqual(u,'http://speech.test/v1/chat/completions')
 def test_without_option_route_is_unchanged(self):
  u,_=self.send({'model':'gpt-6-luna','messages':[]},'');self.assertEqual(u,'http://speech.test/v1/chat/completions')
if __name__=='__main__':unittest.main()
