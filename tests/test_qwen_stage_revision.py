import unittest
from unittest.mock import patch
from llm.qwen_stage_revision import QwenStageRevisionClient
from llm.qwen_stage_protocol import isolate_stage,QwenStageExampleClient


class QwenRevisionTests(unittest.TestCase):
    def test_review_is_also_isolated_and_has_its_own_example(self):
        client=object.__new__(QwenStageExampleClient)
        with patch('llm.qwen_stage_protocol.LocalThinkingClient.generate',return_value='review') as call:
            self.assertEqual(client.review('Task: output Structure:',max_tokens=32768),'review')
            self.assertIn('CURRENT STAGE: REFLECTION',call.call_args.args[1])
            self.assertIn('Review: The last attempt',call.call_args.kwargs['system'])

    def test_isolation_preserves_problem_and_reasserts_stage_after_reference(self):
        prompt,system=isolate_stage('selection','Task: output Expression: x','Output Branch: <number>',True)
        self.assertIn('Task: output Expression: x',prompt)
        self.assertLess(prompt.index('END REFERENCE DATA'),prompt.rindex('Output Branch:'))
        self.assertIn('Branch: 0',system)
        self.assertIn('not a recommendation',system)

    def test_examples_reach_actual_wire_request_without_losing_rationale(self):
        client=object.__new__(QwenStageExampleClient);client.stage_decisions=[]
        with patch('llm.qwen_stage_revision.SplitTreeToolClient._stage',return_value='Branch: 0\nReason: test') as call:
            self.assertEqual(client._stage('selection','Task: Expression: x','unused',32768),'Branch: 0')
            self.assertIn('FORMAT EXAMPLE',call.call_args.args[2])
            self.assertIn('REFERENCE DATA',call.call_args.args[1])

    def test_rationale_is_preserved_but_control_parser_gets_first_line(self):
        client=object.__new__(QwenStageRevisionClient);client.stage_decisions=[]
        with patch('llm.qwen_stage_revision.SplitTreeToolClient._stage',return_value='Branch: 0\nReason: legal alternative'):
            self.assertEqual(client._stage('selection','task','old',32768),'Branch: 0')
        with patch('llm.qwen_stage_revision.SplitTreeToolClient._stage',return_value='Plan: fix x\nChanged condition: x') as call:
            self.assertEqual(client._stage('expansion','task','old',32768),'Plan: fix x')
            self.assertIn('Reason: legal alternative',call.call_args.args[1])

    def test_decisions_reset_between_rounds(self):
        client=object.__new__(QwenStageRevisionClient);client.stage_decisions=['stale']
        with patch('llm.qwen_stage_revision.SplitTreeToolClient.generate',return_value='done'):
            client.generate('task',max_tokens=32768)
        self.assertEqual(client.stage_decisions,[])
