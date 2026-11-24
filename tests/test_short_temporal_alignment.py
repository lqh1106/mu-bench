"""Boundary checks for subsecond temporal ASR evidence; no provider calls."""
from pathlib import Path
import hashlib,json,tempfile,unittest,wave,sys
from types import SimpleNamespace
from unittest.mock import patch
R=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(R))
from mubench.config import activate
activate(R/'runtime/runners/tool_augmented_audio')
from experiments.tau_multiuser_voice.tool_temporal_alignment import prepare_turns
from experiments.tau_multiuser_voice.evaluation_policy import TemporalAlignmentPending
from experiments.tau_multiuser_voice.judge_evidence import build_evidence_catalog
from experiments.mu_bench.logical_time import audible_asr_v3 as asr

class ShortAlignmentTests(unittest.TestCase):
 def fixture(self,root,cut_ms):
  (root/'audio').mkdir();f=root/'audio/both.wav'
  with wave.open(str(f),'wb') as w:w.setnchannels(2);w.setsampwidth(2);w.setframerate(8000);w.writeframes(b'\0\0'*2*8000*3)
  turns=[{'source_id':'original','asr_derived':True,'start_ms':0,'end_ms':2500,'text':'original interval-wide transcript','committed':True}]
  tools=[{'executed':True,'result_available_ms':cut_ms}]
  return f,turns,tools
 def run_clip(self,cut_ms,mode='unknown'):
  tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);root=Path(tmp.name);wavefile,turns,tools=self.fixture(root,cut_ms);before=hashlib.sha256(wavefile.read_bytes()).hexdigest()
  def decode(response,start,part,*args):
   duration=response['assistant_intervals'][0]['end_ms']
   if duration<=1100:
    if mode=='network':raise RuntimeError('HTTP 502')
    if mode=='known':return {'assistant_text':'Okay.'}
    audit=part/'audible_asr_v3/invocation_fixture/audit.json';audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps({'error':'asr_all_segments_unintelligible','source_wav_sha256':'wrong' if mode=='wrong_hash' else hashlib.sha256(Path(response['assistant_wav']).read_bytes()).hexdigest(),'segments':[{'status':'unintelligible','transcript':''}]}))
    raise asr.AudibleASRError('asr_all_segments_unintelligible')
   return {'assistant_text':'The returned state is confirmed.'}
  with patch.object(asr,'transcribe_response',side_effect=decode):
   result=prepare_turns(None,{},SimpleNamespace(task_start_ms=0),turns,tools,root)
  self.assertEqual(before,hashlib.sha256(wavefile.read_bytes()).hexdigest())
  return result,json.loads((root/'temporal_alignment_receipt.json').read_text())
 def test_200ms_unknown_continues_without_inventing_text(self):
  result,receipt=self.run_clip(200)
  self.assertEqual(result[0]['asr_status'],'unintelligible');self.assertEqual(result[0]['text'],'');self.assertTrue(result[0]['transcript_content_unknown']);self.assertIn('confirmed',result[1]['text'])
  self.assertTrue(receipt['segments'][0]['subsecond_unintelligible_accepted'])
  catalog=build_evidence_catalog(result,[])
  self.assertEqual(len(catalog['sources']),1);self.assertNotIn('original interval',catalog['sources'][0]['text'])
 def test_999ms_unknown_is_allowed(self):
  self.assertTrue(self.run_clip(999)[1]['segments'][0]['subsecond_unintelligible_accepted'])
 def test_exactly_one_second_is_not_waived(self):
  with self.assertRaises(TemporalAlignmentPending):self.run_clip(1000)
 def test_above_one_second_is_not_waived(self):
  with self.assertRaises(TemporalAlignmentPending):self.run_clip(1001)
 def test_short_network_error_is_not_unintelligibility(self):
  with self.assertRaises(TemporalAlignmentPending):self.run_clip(200,'network')
 def test_wrong_audio_audit_is_not_accepted(self):
  with self.assertRaises(TemporalAlignmentPending):self.run_clip(200,'wrong_hash')
 def test_valid_short_transcript_is_retained(self):
  result,receipt=self.run_clip(200,'known');self.assertEqual(result[0]['text'],'Okay.');self.assertFalse(receipt['segments'][0]['subsecond_unintelligible_accepted'])
if __name__=='__main__':unittest.main()
