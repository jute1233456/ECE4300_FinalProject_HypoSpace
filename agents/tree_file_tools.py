"""Read-only, per-question tree tools. No reference answers or arbitrary paths."""
from hashlib import sha256
import json
from pathlib import Path


FUNCTIONS = [
    {'name': 'read_tree', 'description': 'Read the complete current search tree from its local file, including closed nodes and all statistics.',
     'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}},
    {'name': 'read_node', 'description': 'Read one node and its immediate children from the local tree file.',
     'parameters': {'type': 'object', 'properties': {'node_id': {'type': 'string'}}, 'required': ['node_id'], 'additionalProperties': False}},
    {'name': 'read_subtree', 'description': 'Read a node, all its descendants, and its ancestors from the local tree file.',
     'parameters': {'type': 'object', 'properties': {'node_id': {'type': 'string'}}, 'required': ['node_id'], 'additionalProperties': False}},
]


class TreeFileTools:
    def __init__(self, path, case_id):
        self.path = Path(path).resolve()
        self.case_id = case_id
        self.trace = []

    def execute(self, name, arguments):
        if name not in {f['name'] for f in FUNCTIONS}:
            raise ValueError('Unknown tree tool')
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        expected = set() if name == 'read_tree' else {'node_id'}
        if not isinstance(args, dict) or set(args) != expected:
            raise ValueError('Unexpected tool arguments')
        # Re-read on every call: tool results cannot silently reuse a stale in-memory tree.
        raw = self.path.read_bytes()
        tree = json.loads(raw)[self.case_id]
        nodes = {n['id']: n for n in tree['nodes']}
        if name == 'read_tree':
            result = tree
        else:
            node_id = args['node_id']
            if not isinstance(node_id, str) or node_id not in nodes:
                raise ValueError('Unknown node_id')
            wanted = {node_id, *nodes[node_id]['children']}
            ancestors = []
            if name == 'read_subtree':
                pending = list(wanted)
                while pending:
                    current = pending.pop()
                    for child in nodes[current]['children']:
                        if child not in wanted:
                            wanted.add(child)
                            pending.append(child)
                parent = nodes[node_id]['parent_id']
                while parent is not None:
                    ancestors.append(nodes[parent])
                    parent = nodes[parent]['parent_id']
            result = dict(nodes=[n for n in tree['nodes'] if n['id'] in wanted], ancestors=ancestors)
        digest = sha256(raw).hexdigest()
        result = dict(case_id=self.case_id, file_sha256=digest, **result)
        self.trace.append(dict(name=name, arguments=args, file=str(self.path),
                               file_sha256=digest, result=result))
        return json.dumps(result, ensure_ascii=False, separators=(',', ':'))
