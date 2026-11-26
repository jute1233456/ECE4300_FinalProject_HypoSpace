import unittest
from server.audit_ds_training_data import identity,truncated,variant


class DSDataAuditTests(unittest.TestCase):
    def test_operator_order_is_not_a_new_problem_but_operator_set_is(self):
        row={'observations':[{'inputs':{'x':0,'y':1},'output':1,'string':'cosmetic'}]}
        a={'variables':['x','y'],'operators':['AND','OR'],'max_depth':2}
        b={**a,'operators':['OR','AND']}
        self.assertEqual(identity('boolean',row,a),identity('boolean',row,b))
        self.assertNotEqual(identity('boolean',row,a),identity('boolean',row,{**a,'operators':['AND','OR','NOR']}))

    def test_reflection_and_hidden_stage_truncations_are_not_ignored(self):
        self.assertTrue(truncated({'reflection_response':{'finish_reason':'length'}}))
        self.assertTrue(truncated({'api_requests':[{'response':{'choices':[{'finish_reason':'length'}]}}]}))
        self.assertFalse(truncated({'finish_reason':'stop','reflection_response':{'finish_reason':'stop'}}))

    def test_revision_partition_and_unrelated_methods_stay_separate(self):
        self.assertEqual(variant({'mcts_version':'2.2','tree_config':{'stage22_revision':True}}),'2.2_revision')
        self.assertEqual(variant({'mcts_version':'2.2','tree_config':{'partition_reflection':True}}),'2.2_partition')
        self.assertEqual(variant({'mcts_version':'2.1'}),'other_DS')
