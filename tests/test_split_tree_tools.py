import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agents.reasoning_tree import ReasoningTreeAgent
from llm.base import LLMResponse
from llm.split_tree_tools import SplitTreeToolClient


class StageTests(unittest.TestCase):
    def run_client(self, outputs):
        client = object.__new__(SplitTreeToolClient)
        client.model = 'fake'
        client.tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text))
        agent = ReasoningTreeAgent(client, 'llm', 7, -1, unbounded_tree=True, mcts_version='2.2')
        prompt = agent.next_prompt('Original task')
        def read(**kwargs):
            client.requests.append(dict(response=dict(usage=dict(prompt_tokens=2, completion_tokens=3,
                total_tokens=5), choices=[dict(finish_reason='stop', message={})])))
            return [dict(function_call=dict(name='read_tree', arguments='{}'))]
        client.tool_model = SimpleNamespace(chat=read)
        def generate(_, prompt, **kwargs):
            text = next(outputs)
            return LLMResponse(text, 'fake', 5, 7, 12, 'stop', reasoning_text='thought')
        with tempfile.TemporaryDirectory() as directory:
            client.bind_tree(agent, Path(directory)/'tree.json', 'case')
            with patch('llm.split_tree_tools.LocalThinkingClient.generate', generate):
                response = client.generate(prompt, max_tokens=32768)
        return agent, response

    def test_stage_pipeline_preserves_model_decisions_and_accounting(self):
        agent, response = self.run_client(iter(['Branch: 0', 'Plan: test constraint', 'Graph: A->B']))
        self.assertEqual(response.text, 'Branch: 0\nPlan: test constraint\nGraph: A->B')
        self.assertEqual([r['stage'] for r in response.api_requests],
                         ['read_tree','selection','expansion','candidate'])
        self.assertEqual(response.total_tokens, 41)
        self.assertEqual(response.tool_trace[0]['name'], 'read_tree')
        self.assertIsNone(response.protocol_error)
        agent.observe(response.text, 'key', True)
        self.assertEqual(len(agent.root.children), 1)

    def test_invalid_selection_does_not_fabricate_candidate(self):
        _, response = self.run_client(iter(['Branch: 9']))
        self.assertTrue(response.final_answer_missing)
        self.assertIn('selection', response.protocol_error)
        self.assertEqual(len(response.api_requests), 2)

    def test_exhaustion_requires_argument_and_is_recorded(self):
        with self.assertRaises(ValueError):
            SplitTreeToolClient.parse_plan('EXHAUSTED')
        _, response = self.run_client(iter(['Branch: 0', 'EXHAUSTED\nProof: All legal possibilities are covered.']))
        self.assertEqual(response.text, 'Branch: 0\nEXHAUSTED')
        self.assertEqual(len(response.api_requests), 3)
        self.assertIn('Proof:', response.api_requests[-1]['stage_response']['text'])
