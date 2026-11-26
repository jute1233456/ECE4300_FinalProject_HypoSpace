import unittest
from server.run_finetune_quick6 import case_result


class ResultTests(unittest.TestCase):
    def test_partial_duplicate_hits_are_not_double_counted(self):
        data = dict(completed=False, cases=[], interrupted_case=dict(records=[
            dict(reference_hit=True, hypothesis_key="a"),
            dict(reference_hit=True, hypothesis_key="a"),
            dict(reference_hit=True, hypothesis_key="b", protocol_error="bad"),
            dict(reference_hit=False, hypothesis_key="c")]))
        result = case_result(data, 4)
        self.assertEqual(result["recovered"], 2)
        self.assertEqual(result["coverage"], .5)
        self.assertEqual(result["protocol_errors"], 1)
        self.assertFalse(result["completed"])

    def test_request_failure_is_not_successful_completion(self):
        data = dict(completed=True, cases=[dict(stop_reason="request_errors", records=[],
                    metrics=dict(recovery=.5))])
        result = case_result(data, 4)
        self.assertFalse(result["completed"])
        self.assertEqual(result["recovered"], 2)
