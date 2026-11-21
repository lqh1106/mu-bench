"""Unknown business basis follows contradicted handling, with raw review retained."""
from pathlib import Path
import sys,copy,json,unittest
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from mubench.config import activate
activate(R/'runtime/runners/tool_augmented_audio')
from experiments.tau_multiuser_voice.models import MultiUserEpisodeTrace,ToolActionTrace
from experiments.tau_multiuser_voice.business_request_basis import route_basis_reviews,ROUTING_POLICY


def trace(executed=True):
 t=MultiUserEpisodeTrace(case_id='fixture')
 for i in range(3):t.tool_actions.append(ToolActionTrace(tool_name='fixture',resource_id='fixture',timestamp_ms=i,oracle_actor_id=None,sut_attributed_actor_id='primary',oracle_authorized=True,executed=executed))
 return t
class RoutingTests(unittest.TestCase):
 def rows(self):return [dict(call_index=i,status=s,event_ids=['e'+str(i)],quotes=['quote'],tool_indices=[],reason='original '+s) for i,s in enumerate(['supported','contradicted','insufficient_evidence'])]
 def test_raw_decisions_unchanged_effective_unknown_is_contradicted(self):
  rows=self.rows();original=copy.deepcopy(rows);t=trace();a=route_basis_reviews(rows,t,'digest')
  self.assertEqual(rows,original);self.assertTrue(a['complete']);self.assertEqual(a['blocking_unresolved_calls'],[]);self.assertEqual(a['insufficient_evidence_routed_as_contradicted'],[2])
  self.assertEqual([r['status'] for r in a['calls']],['supported','contradicted','insufficient_evidence']);self.assertEqual([r['effective_status'] for r in a['calls']],['supported','contradicted','contradicted'])
  self.assertNotIn('negative_basis',a['calls'][2]);self.assertEqual(a['calls'][2]['reason'],'original insufficient_evidence')
 def test_serialized_trace_retains_status_reason_and_sources(self):
  t=trace();route_basis_reviews(self.rows(),t,'digest');again=MultiUserEpisodeTrace.model_validate_json(t.model_dump_json());a=again.tool_actions[2]
  self.assertEqual(a.request_basis_status,'contradicted');self.assertEqual(a.request_basis_original_status,'insufficient_evidence');self.assertEqual(a.request_basis_review_reason,'original insufficient_evidence');self.assertEqual(a.request_basis_evidence,['e2']);self.assertEqual(a.request_basis_routing_policy,ROUTING_POLICY)
 def test_unexecuted_unknown_does_not_invent_execution(self):
  t=trace(False);a=route_basis_reviews(self.rows(),t,'digest');self.assertEqual(a['nonblocking_unexecuted_unknown_calls'],[2]);self.assertTrue(all(not x.executed for x in t.tool_actions));self.assertTrue(all(x.oracle_authorized for x in t.tool_actions))
 def test_downstream_supported_filter_cannot_count_unknown(self):
  t=trace();route_basis_reviews(self.rows(),t,'digest');self.assertEqual([i for i,a in enumerate(t.tool_actions) if a.request_basis_status=='supported'],[0]);self.assertEqual(t.tool_actions[1].request_basis_status,t.tool_actions[2].request_basis_status)
if __name__=='__main__':unittest.main()
