from pathlib import Path
import json
import tempfile
import unittest

from server.run_mixed_mcts_queue import case_ids, writers, Job, baseline_ready, reference_counts


class MixedQueueTests(unittest.TestCase):
    def test_resume21_gate_waits_for_every_normal_22_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'a.json',Path(directory)/'b.json'
            def write(p,done,reason='model_call_cap'):
                p.write_text(json.dumps(dict(completed=done,cases=[dict(stop_reason=reason)] if done else [])))
            write(a,True);write(b,False)
            jobs=[Job('2.2','evaluation22','3d',p.stem,str(p)) for p in (a,b)]
            self.assertFalse(baseline_ready(jobs))
            write(b,True,'request_errors')
            self.assertFalse(baseline_ready(jobs))
            write(b,True)
            self.assertTrue(baseline_ready(jobs))
            self.assertFalse(baseline_ready([]))

    def test_22_budget_does_not_depend_on_unfinished_21_results(self):
        counts=reference_counts(Path(__file__).resolve().parents[1]/'datasets/medium_sampled')
        self.assertEqual(len(counts),27)
        self.assertEqual(2*(counts['causal','n1_002']+5),106)
        self.assertEqual(2*(counts['causal','n2_006']+5),58)
        self.assertEqual(2*(counts['3d','g3_h3_b2_002']+5),28)

    def test_dataset_jobs_include_nested_and_flat_domains(self):
        jobs=case_ids(Path(__file__).resolve().parents[1]/'datasets/medium_sampled')
        self.assertEqual(len(jobs),27)
        self.assertEqual(len(set(jobs)),27)
        self.assertIn(('causal','n1_003'),jobs)
        self.assertIn(('3d','g3_h3_b2_002'),jobs)

    def test_adoption_requires_exact_module_and_output_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for pid,args in [(1,['python','-m','experiments.test_easy','--output','/target.json']),
                             (2,['python','-m','experiments.test_easy','--output','/other.json']),
                             (3,['bash','echo experiments.test_easy --output /target.json'])]:
                p=root/str(pid);p.mkdir();(p/'cmdline').write_bytes(b'\0'.join(x.encode() for x in args))
            self.assertEqual(writers('/target.json',root),[1])
