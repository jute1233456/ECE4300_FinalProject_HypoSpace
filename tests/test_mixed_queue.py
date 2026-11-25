from pathlib import Path
import tempfile
import unittest

from server.run_mixed_mcts_queue import case_ids, writers


class MixedQueueTests(unittest.TestCase):
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
