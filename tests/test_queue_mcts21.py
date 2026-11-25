import json
import os
from pathlib import Path
import tempfile
import unittest


@unittest.skipUnless(os.name=='posix', 'Server queue uses POSIX process locking')
class QueueCompletionTests(unittest.TestCase):
    def test_both_methods_must_finish_normally(self):
        from server.queue_mcts21 import normal_complete
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);jobs=[('causal','case')]
            record=dict(completed=True,cases=[dict(domain='causal',case_id='case',stop_reason='all_recovered')])
            for m in ('mcts','two-step'):
                (root/m).mkdir();(root/m/'causal-case.json').write_text(json.dumps(record))
            self.assertTrue(normal_complete(root,jobs))
            record['cases'][0]['stop_reason']='request_errors'
            (root/'two-step/causal-case.json').write_text(json.dumps(record))
            self.assertFalse(normal_complete(root,jobs))

    def test_missing_or_partial_question_is_not_completion(self):
        from server.queue_mcts21 import normal_complete
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);jobs=[('causal','case')]
            self.assertFalse(normal_complete(root,jobs))
            for m in ('mcts','two-step'):
                (root/m).mkdir();(root/m/'causal-case.json').write_text(json.dumps(dict(
                    completed=False,cases=[],interrupted_case={'case_id':'case'})))
            self.assertFalse(normal_complete(root,jobs))
