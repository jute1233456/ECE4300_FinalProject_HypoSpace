import unittest
from unittest.mock import patch
from llm.qwen_stage_revision import QwenStageRevisionClient


class QwenRevisionTests(unittest.TestCase):
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
