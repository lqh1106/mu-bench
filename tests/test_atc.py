from pathlib import Path
from unittest.mock import patch
import hashlib,json,sys,tempfile,types,unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mubench.metrics import from_rice,write_atc

class ATCTests(unittest.TestCase):
    def test_axis_mapping_and_equal_weight(self):
        r=from_rice(dict(routing_score=.25,continuity_score=1,isolation_score=.5,equity_score=1))
        self.assertEqual((r['attribution_score'],r['threading_score'],r['coordination_score']),(.25,1.,.75))
        self.assertEqual(r['coordination_denominator'],2)
    def test_absent_equity_uses_isolation(self):
        for extra in ({},{'equity_score':None}):
            r=from_rice(dict(routing_score=1/3,continuity_score=2/3,isolation_score=1,**extra))
            self.assertEqual(r['coordination_score'],1)
            self.assertEqual(r['coordination_components_used'],['I'])
            self.assertFalse(r['equity_applicable'])
            self.assertEqual(r['coordination_denominator'],1)
    def test_zero_equity_is_applicable(self):
        r=from_rice(dict(isolation_score=1,equity_score=0))
        self.assertEqual(r['coordination_score'],.5)
        self.assertTrue(r['equity_applicable'])
    def test_missing_isolation_does_not_become_equity(self):
        r=from_rice(dict(equity_score=1))
        self.assertIsNone(r['coordination_score'])
    def test_invalid_units_rejected(self):
        for x in (float('nan'),float('inf'),-1,2,True,'1'):
            with self.assertRaises(ValueError):from_rice(dict(routing_score=x))
    def test_source_is_unchanged_and_provenance_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            d=Path(td);rice={'routing_score':.2,'continuity_score':.3,'isolation_score':.4,'equity_score':None,'measurement_track':'text-logical','voice_comparable':False,'units_used':{'I':['privacy_success']}}
            p=d/'rice_score.json';p.write_text(json.dumps(rice));before=p.read_bytes()
            r=write_atc(d,rice)
            self.assertEqual(p.read_bytes(),before)
            self.assertEqual(r['source_rice_sha256'],hashlib.sha256(before).hexdigest())
            self.assertEqual(r['source_units_used'],rice['units_used'])
            self.assertFalse(r['voice_comparable'])
    def test_finalize_publishes_atc_without_rejudging(self):
        from mubench.finalize import finalize
        rice={'routing_score':.25,'continuity_score':1,'isolation_score':.5,'equity_score':1}
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);case=root/'case';case.mkdir();(case/'score.json').write_text('{"task_success": true}')
            (case/'full_review_audit.json').write_text(json.dumps({'completed':True,'item_ids':['fixture:all'],'covered_item_ids':['fixture:all']}))
            score_before=(case/'score.json').read_bytes();formal=root/'formal'
            module=types.ModuleType('rice_compute');module.compute_rice=lambda score,overlay:dict(rice)
            with patch('mubench.config.activate'),patch.dict(sys.modules,{'rice_compute':module}):
                result=finalize(case,root,root/'private.env','test',formal,'text')
            self.assertEqual((case/'score.json').read_bytes(),score_before)
            self.assertEqual(json.loads((formal/'atc_score.json').read_text())['coordination_score'],.75)
            self.assertEqual(result['atc_sha256'],hashlib.sha256((formal/'atc_score.json').read_bytes()).hexdigest())
            self.assertTrue((formal/'rice_score.json').exists())

if __name__=='__main__':unittest.main()
