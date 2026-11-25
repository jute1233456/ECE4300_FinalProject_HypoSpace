import json
from pathlib import Path
import tempfile
import unittest
from server.compare_tree_tools import compare, metrics


class TreeComparisonTests(unittest.TestCase):
    def test_tool_and_reflection_costs_are_counted_without_double_counting(self):
        records = [dict(valid=True, hypothesis_key=['edge'], reference_hit=True, total_tokens=40,
                        api_requests=[{},{}], tool_trace=[{}], reflection_response={'total_tokens':10}),
                   dict(valid=True, hypothesis_key=['edge'], reference_hit=True, total_tokens=30,
                        api_requests=[{},{}],tool_trace=[{}])]
        case = dict(records=records,metrics={'recovery':0.5})
        report = metrics([({},case)])
        self.assertEqual(report['api_calls'],5)
        self.assertEqual(report['total_tokens'],80)
        self.assertEqual(report['distinct_reference_hits'],1)
        self.assertEqual(report['valid_duplicate_fraction'],0.5)

    def test_only_normal_completed_questions_with_matching_config_are_compared(self):
        with tempfile.TemporaryDirectory() as folder:
            old,new = Path(folder)/'old',Path(folder)/'new'
            for root in (old,new):
                (root/'mcts').mkdir(parents=True)
                data = dict(completed=True,requested_model='model',max_tokens=16384,
                    tree_config={'policy':'llm','child_limit':7},cases=[dict(domain='causal',case_id='case',
                    reference_count=1,submission_cap=6,model_call_cap=12,prompt='same',records=[],
                    metrics={'recovery':0})])
                if root==new:
                    data['tree_config']['tree_tools']={'version':1}
                (root/'mcts/case.json').write_text(json.dumps(data))
            self.assertEqual(len(compare(old,new)['groups']['mcts']['matched_cases']),1)
            data['cases'][0]['model_call_cap']=10
            (new/'mcts/case.json').write_text(json.dumps(data))
            report=compare(old,new)['groups']['mcts']
            self.assertEqual(len(report['matched_cases']),0)
            self.assertEqual(len(report['excluded_config_mismatch']),1)
