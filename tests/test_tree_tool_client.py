from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from agents.reasoning_tree import ReasoningTreeAgent
from llm.tree_tools import TreeToolClient


class TreeToolClientTests(unittest.TestCase):
    def test_missing_branch_is_reasked_and_extra_request_is_charged(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = ReasoningTreeAgent(None, 'llm', 7, -1)
            prompt = agent.next_prompt('Question')
            client = self.client(agent, Path(folder)/'tree.json', [
                [dict(role='assistant', function_call={'name':'read_tree','arguments':'{}'},content='')],
                [dict(role='assistant',content='Expression: x AND y')],
                [dict(role='assistant',content='Branch: 0\nPlan: Try conjunction\nExpression: x AND y')]])
            response = client.generate(prompt)
            self.assertEqual(response.format_repairs,1)
            self.assertEqual(len(response.api_requests),3)
            self.assertEqual(response.total_tokens,45)
            self.assertTrue(response.text.startswith('Branch: 0'))

    def client(self, agent, path, responses):
        client = object.__new__(TreeToolClient)
        client.model = 'test'
        client.tokenizer = Mock()
        client.tokenizer.encode.return_value = [1]
        client.bind_tree(agent, path, 'case')
        remaining = iter(responses)

        def chat(**kwargs):
            content = next(remaining)
            client.requests.append({'response': {'choices': [{'finish_reason': 'stop'}],
                'usage': dict(prompt_tokens=10, completion_tokens=5, total_tokens=15)}})
            return content
        client.tool_model = Mock()
        client.tool_model.chat.side_effect = chat
        return client

    def test_real_disk_read_is_returned_before_final_and_all_calls_are_counted(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = ReasoningTreeAgent(None, 'llm', 7, -1)
            prompt = agent.next_prompt('Question')
            client = self.client(agent, Path(folder)/'tree.json', [
                [dict(role='assistant', function_call={'name':'read_tree', 'arguments':'{}'}, content='')],
                [dict(role='assistant', content='<think>Graph: WRONG</think>\nBranch: 0\nPlan: Try an edge\nGraph: B->A')]])
            response = client.generate(prompt)
            self.assertEqual(response.total_tokens, 30)
            self.assertEqual(len(response.api_requests), 2)
            self.assertEqual(response.tool_trace[0]['result']['nodes'][0]['id'], 'root')
            self.assertNotIn('WRONG', response.text)
            self.assertEqual(response.reasoning_text, 'Graph: WRONG')
            self.assertIsNone(response.protocol_error)
            sent = client.tool_model.chat.call_args.kwargs['messages']
            self.assertTrue(any(m['role']=='function' for m in sent))

    def test_skipped_reads_exhaust_budget_without_accepting_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = ReasoningTreeAgent(None, 'llm', 7, -1)
            prompt = agent.next_prompt('Question')
            client = self.client(agent, Path(folder)/'tree.json', [
                [dict(role='assistant', content='Branch: 0\nGraph: B->A')] for _ in range(5)])
            response = client.generate(prompt)
            self.assertEqual(response.text, '')
            self.assertTrue(response.final_answer_missing)
            self.assertEqual(len(response.api_requests), 5)
            self.assertIsNotNone(response.protocol_error)
