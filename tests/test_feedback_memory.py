import unittest
from agents.feedback_memory import FeedbackMemoryAgent
from experiments.candidate_feedback import describe_candidate
from experiments.direct_evaluation import ROOT,load_domain


class FeedbackMemoryTests(unittest.TestCase):
    def test_invalid_history_is_labeled_and_explanation_is_not_copied(self):
        agent=FeedbackMemoryAgent(None)
        agent.observe('Expression: NOR(x,y)\nA misleading explanation.',('NOR',),False,
                      candidate='Expression: NOR(x,y)',issues=['Outputs 1, expected 0.'])
        prompt=agent.next_prompt('Question')
        self.assertIn('INVALID',prompt)
        self.assertIn('Outputs 1, expected 0.',prompt)
        self.assertNotIn('misleading explanation',prompt)
        self.assertEqual(agent.valid_candidates,{})
        self.assertEqual(FeedbackMemoryAgent(None).next_prompt('Question'),'Question')

    def test_duplicate_valid_answer_uses_one_memory_entry(self):
        agent=FeedbackMemoryAgent(None)
        agent.observe('Expression: x','x',True)
        _,info=agent.observe('Expression: x','x',True)
        self.assertTrue(info['duplicate_valid'])
        self.assertEqual(len(agent.valid_candidates),1)
        self.assertEqual(agent.valid_candidates['x']['count'],2)

    def test_boolean_feedback_reports_actual_truth_table_violation(self):
        benchmark,prepare,score=load_domain('boolean',ROOT/'datasets/balanced/boolean.json')
        case=next(c for c in benchmark.all_observation_sets if c['observation_set_id']=='n1_inputs_00_outputs_0')
        observations,_=prepare(case)
        key,valid=score('Expression: NOR(x,y)',observations)
        _,issues=describe_candidate('boolean',benchmark,observations,'Expression: NOR(x,y)',key,valid)
        self.assertFalse(valid)
        self.assertTrue(any('outputs 1' in issue and 'output is 0' in issue for issue in issues))

    def test_parse_failure_is_not_reported_as_a_computed_zero(self):
        benchmark,prepare,score=load_domain('boolean',ROOT/'datasets/balanced/boolean.json')
        case=next(c for c in benchmark.all_observation_sets if c['observation_set_id']=='n1_inputs_00_outputs_1')
        observations,_=prepare(case)
        text='Expression: NOT (x NOR y)'
        key,valid=score(text,observations)
        _,issues=describe_candidate('boolean',benchmark,observations,text,key,valid)
        self.assertFalse(valid)
        self.assertTrue(any('syntax' in issue for issue in issues))
        self.assertFalse(any('outputs 0' in issue for issue in issues))
