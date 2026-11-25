import json
from pathlib import Path
import tempfile
import unittest

from agents.tree_file_tools import TreeFileTools


class TreeFileToolsTests(unittest.TestCase):
    def test_reads_descendants_closed_nodes_and_current_disk_version(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'tree.json'
            nodes = [dict(id='root', parent_id=None, children=['root.0']),
                     dict(id='root.0', parent_id='root', children=['root.0.0']),
                     dict(id='root.0.0', parent_id='root.0', children=[], closed=True)]
            path.write_text(json.dumps({'case': {'nodes': nodes}, 'other': {'secret': 123}}))
            tools = TreeFileTools(path, 'case')
            result = json.loads(tools.execute('read_subtree', {'node_id': 'root.0'}))
            self.assertEqual([n['id'] for n in result['nodes']], ['root.0', 'root.0.0'])
            self.assertEqual(result['ancestors'][0]['id'], 'root')
            self.assertTrue(result['nodes'][-1]['closed'])
            self.assertNotIn('secret', tools.execute('read_tree', {}))
            digest = tools.trace[-1]['file_sha256']
            nodes[-1]['closed'] = False
            path.write_text(json.dumps({'case': {'nodes': nodes}}))
            self.assertFalse(json.loads(tools.execute('read_node', {'node_id': 'root.0.0'}))['nodes'][0]['closed'])
            self.assertNotEqual(digest, tools.trace[-1]['file_sha256'])

    def test_cannot_choose_a_file_or_unknown_tool(self):
        tools = TreeFileTools('unused.json', 'case')
        for name, arguments in [('read_tree', {'path': '/etc/passwd'}),
                                ('run_shell', {}), ('read_node', {}),
                                ('read_tree', [])]:
            with self.assertRaises(ValueError):
                tools.execute(name, arguments)
