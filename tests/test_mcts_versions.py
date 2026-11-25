import unittest
from agents.mcts_versions import infer_version
from agents.reasoning_tree import ReasoningTreeAgent


class MCTSVersionTests(unittest.TestCase):
    def test_old_protocol_and_strict_policy_are_separate(self):
        old = ReasoningTreeAgent(None,'llm',7,-1,mcts_version='2.0')
        new = ReasoningTreeAgent(None,'llm',7,-1,mcts_version='2.1')
        for agent in (old,new):
            agent.history['key']='Graph: A->B'
        self.assertIn('Return a structurally different hypothesis',old.next_prompt('Question'))
        prompt = new.next_prompt('Question')
        self.assertNotIn('Return a structurally different hypothesis',prompt)
        self.assertIn('duplicates are allowed',prompt)
        self.assertIn('sound theoretical argument',prompt)
        self.assertNotIn('If this direction cannot be expanded, write EXHAUSTED',prompt)
        self.assertIn('If this direction cannot be expanded, write EXHAUSTED',old.protocol())

    def test_historical_protocol_version_is_not_public_method_version(self):
        base = {'method':'mcts-tree','tree_config':{'policy':'llm'}}
        self.assertEqual(infer_version(base),'1.0')
        base['tree_config']['tree_tools']={'version':2}
        self.assertEqual(infer_version(base),'2.0')
        base['mcts_version']='2.1'
        self.assertEqual(infer_version(base),'2.1')
        self.assertIsNone(infer_version({'method':'direct'}))
