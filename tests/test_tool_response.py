import unittest
from agents.reasoning_tree import ReasoningTreeAgent
from llm.tool_response import tree_candidate


class ToolResponseTests(unittest.TestCase):
    def test_reasoning_cannot_enter_branch_selection_or_memory(self):
        agent = ReasoningTreeAgent(None, 'llm', 7, -1)
        agent.next_prompt('Question')
        raw = '<think>Branch: 9\nGraph: Wrong\n</think>\nBranch: 0\nPlan: New direction\nGraph: B->A'
        reasoning, final = tree_candidate(raw, len(agent.options))
        _, trace = agent.observe(final, 'edge', True)
        self.assertIn('Wrong', reasoning)
        self.assertEqual(trace['tree_event'], 'expanded')
        self.assertEqual(agent.root.visits, 1)
        self.assertEqual(agent.history['edge'], 'Graph: B->A')
        self.assertNotIn('Wrong', agent.history['edge'])

    def test_incomplete_thinking_and_unlisted_branch_are_rejected(self):
        for text in ('<think>Branch: 0\nGraph: A->B',
                     'Branch: 3\nGraph: A->B', 'Graph: A->B'):
            with self.assertRaises(ValueError):
                tree_candidate(text, 1)
