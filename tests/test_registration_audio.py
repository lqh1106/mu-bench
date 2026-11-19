import sys, unittest
from unittest.mock import patch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mubench.config import activate
activate(Path(__file__).resolve().parents[1]/'runtime/runners/tool_free')
from experiments.tau_multiuser_voice.cases import CaseCatalog
from experiments.tau_multiuser_voice.enrollment import InSessionVoiceEnrollment
from experiments.tau_multiuser_voice.spoken_protocol import with_spoken_session_rules,receipt
from experiments.tau_multiuser_voice.registration_audio import spoken_codes,segments,MAX_WORDS,MAX_CHARACTERS
class FakeTTS:
    sample_rate=16000
    def synthesize(self,text,voice):return b'\x01\x00'*1600
class RegistrationTests(unittest.TestCase):
    def test_bounded_complete_chunks(self):
        text='One short sentence. '+('A long sentence with several words, '*20)+'My participant code is p01.'
        pieces=segments(text)
        self.assertEqual(' '.join(pieces).split(),text.split())
        self.assertTrue(all(len(p.split())<=MAX_WORDS and len(p)<=MAX_CHARACTERS for p in pieces))
        self.assertEqual(pieces[0],'One short sentence.')
        self.assertEqual(spoken_codes('p01 p02'),'p zero one p zero two')
    def test_protocol_does_not_use_participant_slot(self):
        case=with_spoken_session_rules(CaseCatalog.get('airline_04_gate_announcement_collision__g-small__a-clean__c-strong'))
        with patch('experiments.tau_multiuser_voice.scene_audio.render_source',return_value=(b'\x01\x00'*1600,{'audio_segments':[]})):
            en=InSessionVoiceEnrollment(case=case,tts=FakeTTS(),separate_session_protocol=True)
        r=receipt(en)
        self.assertEqual(len(en.traces),len([u for u in case.users if u.voice.enrollment_status=='enrolled']))
        self.assertTrue(all('report_speaker_attribution once' not in t.utterance for t in en.traces))
        self.assertGreater(en.traces[0].start_ms,r['session_end_ms'])
        self.assertTrue(all(case.public_participant_id(t.user_id) in [case.public_participant_id(u.user_id) for u in case.users] for t in en.traces))
        self.assertNotIn(r['speaker_id'],[b['user_id'] for b in en.prompt_bindings()])
        self.assertEqual(en.queues[r['speaker_id']].items[0].audio_id,'session-protocol')
if __name__=='__main__':unittest.main()
