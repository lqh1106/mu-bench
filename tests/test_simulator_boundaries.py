from pathlib import Path
import ast,json,unittest
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
from mubench.config import activate,profile
activate(profile('native','tool_free'))
from experiments.tau_multiuser_voice.cases import CaseCatalog
from experiments.tau_multiuser_voice.simulation_contract import writer_prompt,compile_event_contract,KNOWN_SPEECH_ACTS,SimulatorInvalid
from experiments.mu_bench.logical_time.online_users import OnlineClosurePlanner
from experiments.tau_multiuser_voice.models import WorldObservation,ObservationKind

class SimulatorBoundaries(unittest.TestCase):
 def setUp(self):
  self.case=CaseCatalog.get('airline_04_gate_announcement_collision__g-small__a-clean__c-strong')
 def test_writer_receives_fixed_recipient_and_speech_act(self):
  event=self.case.events[0]
  p=json.loads(writer_prompt({'event_contract':compile_event_contract(self.case,event)},alignment_ms=200))
  self.assertEqual(p['output_schema']['properties']['speech_act']['const'],event.speech_act)
  self.assertEqual(p['output_schema']['properties']['addressee']['const'],event.addressee)
 def test_closure_receives_exact_legal_enums_and_namespace(self):
  class Backend:
   def generate_json(s,messages):
    s.messages=messages;return {'action':'end','actor_id':None,'reason':'exchange finished'}
  backend=Backend();planner=OnlineClosurePlanner(backend)
  verdict=SimpleNamespace(accepted=True,model_dump=lambda **kw:{'accepted':True})
  with patch('experiments.mu_bench.logical_time.online_users.critique_closure',return_value=verdict):
   planner.decide_online(case=self.case,observation=WorldObservation(sim_time_ms=0,kind=ObservationKind.CLOCK),spoken_events=[],round_index=0,played_timeline=[])
  contract=json.loads(backend.messages[-1]['content'])['output_contract']
  self.assertEqual(set(contract['allowed_speech_acts']),set(KNOWN_SPEECH_ACTS))
  self.assertEqual(set(contract['allowed_addressees']),{u.user_id for u in self.case.users}|{'assistant','group',None})
 def test_exhaustion_scores_prefix_other_faults_propagate(self):
  tree=ast.parse((ROOT/'runtime/runners/tool_augmented_text/run_text_case.py').read_text())
  cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Episode')
  node=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run')
  writes=[];scope={'write':lambda p,o:writes.append(o)}
  exec(compile(ast.Module(body=[node],type_ignores=[]),'<actual-text-run>','exec'),scope)
  for reason in ['proposal_contract_exhausted','closure_contract_exhausted']:
   def fail():raise SimulatorInvalid(reason,case_id=self.case.case_id,event_id='e_test')
   obj=SimpleNamespace(interact=fail,save=lambda:None,evaluate=lambda:'scored',out=Path('unused'))
   self.assertEqual(scope['run'](obj),'scored');self.assertFalse(writes[-1]['unaccepted_candidate_delivered'])
  def bug():raise RuntimeError('other technical fault')
  obj.interact=bug
  with self.assertRaises(RuntimeError):scope['run'](obj)

if __name__=='__main__':unittest.main()
