import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from server.run_dynamic_validation import append_extra_suites, ordered_waiting


class ExtraSuiteTests(unittest.TestCase):
    def test_skip_inflight_family_and_keep_cross_difficulty_ids_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'manifest.json'
            jobs=[dict(id=d+'/causal/n1_002',dataset='causal',case_id='n1_002',
                       family_sha256=d,references=4,difficulty=d,dataset_root=d)
                  for d in ['easy','medium']]
            path.write_text(json.dumps(dict(jobs=jobs)))
            cfg=dict(models=['sft','dpo'],additional_suites=[dict(name='mixed',manifest=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),models=['sft','dpo'])])
            tasks=[dict(id='sft/old',model='sft',family_sha256='medium',status='running')]
            self.assertEqual(append_extra_suites(tasks,cfg),3)
            self.assertEqual(append_extra_suites(tasks,cfg),0)
            self.assertEqual(len({t['id'] for t in tasks}),4)
            dpo=[t for t in tasks if t['model']=='dpo']
            self.assertEqual(len({t['result_key'] for t in dpo}),2)

    def test_quick_tasks_dispatch_before_added_suite(self):
        cfg=dict(models=['sft','dpo'])
        tasks=[dict(model='sft',status='pending',dataset='3d',case_id='new',references=3,queue_priority=1),
               dict(model='dpo',status='pending',dataset='3d',case_id='quick',references=9)]
        self.assertEqual(ordered_waiting(tasks,cfg)[0]['case_id'],'quick')


if __name__=='__main__':unittest.main()
