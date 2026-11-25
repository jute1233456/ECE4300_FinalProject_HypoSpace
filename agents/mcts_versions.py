"""Public method versions, independent of model and internal tool protocol versions."""
VERSIONS = {
    '2.2': 'Stage-MCTS: independent review, read-tree, selection, expansion and candidate requests.',
    '1.0': 'LLM selects branches from text prompts; no file-reading tools.',
    '2.0': 'File-reading tools expose the complete search tree; original search prompts.',
    '2.1': 'Version 2.0 plus duplicate submission fallback and theory-based exhaustion rules.',
}

COMPLETION_POLICY = (
    'Prefer a structurally new valid candidate, but duplicates are allowed. '
    'Once you have a complete candidate you believe satisfies the observations, submit it '
    'even if it appeared before. Do not continue reasoning solely to avoid repetition. '
    'The evaluator checks validity and duplication. '
    'If you cannot find a new Plan, omit Plan and submit your available candidate; '
    'do not invent a step solely to satisfy the format. '
    'Output EXHAUSTED only when a sound theoretical argument from the problem rules and '
    'selected branch constraints establishes that ALL legal possibilities within that branch '
    'have been ruled out or already covered. This conclusion applies only to that branch. '
    'Failure to find a new candidate, repeated answers, uncertainty, reaching a child limit, '
    'or insufficient tokens or search budget does NOT establish exhaustion. '
    'Without such an argument, do not output EXHAUSTED; submit an available candidate, even if repeated.'
)


def infer_version(result):
    """Read historical metadata without rewriting results or confusing protocol version=2."""
    if result.get('mcts_version'):
        return result['mcts_version']
    tree = result.get('tree_config', {})
    if result.get('method') != 'mcts-tree' or tree.get('policy') != 'llm':
        return None
    return '2.0' if tree.get('tree_tools') else '1.0'
