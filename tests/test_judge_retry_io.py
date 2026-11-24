from pathlib import Path
from unittest import TestCase, mock
import json
import tempfile
from mubench.judge_retry_io import prepare_retry, promote_retry
from mubench.finalize import finalize

class RetryIOTests(TestCase):
    def test_retry_preserves_real_receipts_and_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);retry=root/'semantic_judge_retry_2'
            (root/'initial_assistant_context.json').write_text('{"required":true}')
            (root/'semantic_judge_response.json').write_text('{"old":"invalid citation"}')
            (root/'score.json').write_text('{"task_success":false}')
            score=(root/'score.json').read_bytes();receipt=(root/'initial_assistant_context.json').read_bytes()
            prepare_retry(root,retry)
            self.assertEqual((retry/'initial_assistant_context.json').read_bytes(),receipt)
            (retry/'full_review_audit.json').write_text(json.dumps({'completed':True,'item_ids':['goal:x'],'covered_item_ids':['goal:x']}))
            (retry/'semantic_judge_response.json').write_text('{"new":"adopted"}')
            promote_retry(root,retry,2)
            self.assertEqual((root/'score.json').read_bytes(),score)
            self.assertEqual((root/'initial_assistant_context.json').read_bytes(),receipt)
            self.assertEqual(json.loads((root/'semantic_judge_response.json').read_text()),{'new':'adopted'})
            self.assertEqual(json.loads((root/'_judge_retry_history/before_attempt_2/semantic_judge_response.json').read_text()),{'old':'invalid citation'})
    def test_incomplete_retry_cannot_be_adopted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);retry=root/'retry';retry.mkdir()
            (retry/'full_review_audit.json').write_text(json.dumps({'completed':True,'item_ids':['x'],'covered_item_ids':[]}))
            with self.assertRaises(ValueError):promote_retry(root,retry,2)
            self.assertFalse((root/'full_review_audit.json').exists())
    def test_incomplete_audit_cannot_be_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'score.json').write_text('{"task_success":false}')
            with mock.patch('mubench.config.activate'), self.assertRaisesRegex(ValueError,'complete.*review'):
                finalize(root,None,None,'unit-model',root/'formal','text')
            self.assertFalse((root/'formal').exists())
