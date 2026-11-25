import unittest
from experiments.generate_medium_training import family,graph_universe


class MediumTrainingGenerationTests(unittest.TestCase):
    def test_causal_node_renaming_is_same_family(self):
        a={'observations':[{'perturbed_node':'A','effects':{'A':0,'B':1}}]}
        b={'observations':[{'perturbed_node':'B','effects':{'A':1,'B':0}}]}
        self.assertEqual(family(a,'causal',{'nodes':['A','B']}),family(b,'causal',{'nodes':['A','B']}))

    def test_boolean_variable_swap_is_same_family(self):
        meta={'variables':['x','y'],'operators':['AND','OR'],'max_depth':2}
        a={'observations':[{'inputs':{'x':0,'y':1},'output':0}]}
        b={'observations':[{'inputs':{'x':1,'y':0},'output':0}]}
        self.assertEqual(family(a,'boolean',meta),family(b,'boolean',meta))

    def test_voxel_translation_and_rotation_are_same_family(self):
        a={'observation':'110000000'}
        b={'observation':'000010010'}
        self.assertEqual(family(a,'3d',{'grid_size':3,'max_height':3}),
                         family(b,'3d',{'grid_size':3,'max_height':3}))
        self.assertEqual(len(graph_universe(list('ABCD'))),543)
