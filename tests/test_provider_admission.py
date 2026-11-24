from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase, mock
import json
import tempfile
from mubench import cli
from mubench.provider_errors import admission_block_reason, attempt_admission_block

class AdmissionTests(TestCase):
    def test_bare_handshake_402(self):
        for text in ('server rejected WebSocket connection: HTTP 402',
                     'HTTP Error 402: Payment Required', 'http/1.1 402 Payment Required'):
            self.assertEqual(admission_block_reason(text), 'http_402')
    def test_existing_quota_markers(self):
        self.assertEqual(admission_block_reason('personal-team-blocked:spending-limit'), 'spending-limit')
        self.assertEqual(admission_block_reason('payment_required'), 'payment_required')
    def test_other_errors_do_not_stop_batch(self):
        for text in ('HTTP 502', 'received 1011 internal error', '402 audio samples', 'Not connected to API'):
            self.assertIsNone(admission_block_reason(text))
    def test_underlying_failure_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=root/'evidence';result.mkdir()
            (result/'runner_error.json').write_text(json.dumps({'error':'HTTP 402'}))
            self.assertEqual(attempt_admission_block(root,result,'No score'), 'http_402')
    def test_log_tail_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'case.log').write_text('WebSocket connection rejected: HTTP 402')
            self.assertEqual(attempt_admission_block(root,root/'absent','No score'), 'http_402')
    def test_blocked_batch_retains_unstarted_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cases=[Path('case_'+str(i)+'.json') for i in range(4)]
            cfg={'sut':{'provider':'openai','model':'unit-model'},'output_root':str(root)}
            args=SimpleNamespace(config='unused',subset='tool_free',case=None,workers=1,ablation=None)
            with mock.patch.object(cli,'load',return_value=cfg),mock.patch.object(cli,'selected',return_value=cases),mock.patch.object(cli,'run_one',return_value={'case_id':'case_0','status':'technical_error','admission_block_reason':'http_402'}) as run:
                self.assertEqual(cli.run(args),1)
                self.assertEqual(run.call_count,1)
            rows=json.loads((root/'latest_run_summary.json').read_text())
            self.assertEqual([r['status'] for r in rows],['technical_error','not_started','not_started','not_started'])
            status=json.loads((root/'unit-model/tool_free/baseline/admission_status.json').read_text())
            self.assertEqual(status['pending_case_ids'],['case_1','case_2','case_3'])
            self.assertFalse((root/'unit-model/tool_free/baseline/runs').exists())
    def test_non_billing_error_keeps_next_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cases=[Path('case_0.json'),Path('case_1.json')]
            cfg={'sut':{'provider':'openai','model':'unit-model'},'output_root':str(root)}
            args=SimpleNamespace(config='unused',subset='tool_free',case=None,workers=1,ablation=None)
            with mock.patch.object(cli,'load',return_value=cfg),mock.patch.object(cli,'selected',return_value=cases),mock.patch.object(cli,'run_one',side_effect=[{'case_id':'case_0','status':'technical_error'},{'case_id':'case_1','status':'complete','task_success':True}]) as run:
                self.assertEqual(cli.run(args),1);self.assertEqual(run.call_count,2)
