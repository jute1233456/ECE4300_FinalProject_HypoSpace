"""Constraint-only feedback; never consult reference answers."""
from .boolean_scoring import expand_nor


def describe_candidate(domain, benchmark, observations, text, key, valid):
    issues=[]
    parsed=benchmark.parse_llm_response(text)
    if parsed is None:
        import re
        brief=next((line.strip()[:200] for line in text.splitlines()
                    if re.match(r'\s*(Graph|Expression|Structure):',line,re.I)), 'No parseable candidate')
        return brief, ['Cannot parse candidate. Check required syntax, allowed names and structural constraints.']
    if domain=='boolean':
        candidate='Expression: '+parsed
        try:
            from boolean_dataset import BooleanExpression
            expression=BooleanExpression(expand_nor(parsed),benchmark.variables,benchmark.operators)
            # The upstream evaluate() returns 0 for missing truth-table entries,
            # including parse failures. That default must never become feedback.
            if expression.sympy_expr is None or any(str(s) not in benchmark.variables
                                                    for s in expression.sympy_expr.free_symbols):
                raise ValueError('Cannot evaluate an unparsed expression or unknown variables')
            for obs in observations:
                actual=int(expression.evaluate(obs.inputs))
                if actual!=obs.output:
                    issues.append(f'Inputs {obs.inputs}: expression outputs {actual}, observed output is {obs.output}.')
        except Exception:
            issues.append('Invalid expression syntax; use infix AND/OR, unary NOT, and NOR(a,b), with allowed variables only.')
        if not valid and not issues:
            issues.append('Expression violates the allowed structural space: check operators, variables, nesting depth and constants.')
    elif domain=='causal':
        import networkx as nx
        candidate='Graph: '+(', '.join(f'{a}->{b}' for a,b in sorted(parsed.edges)) or 'No edges')
        graph=parsed.to_networkx()
        for obs in observations:
            source=obs['perturbed_node']
            descendants=nx.descendants(graph,source)
            for node,expected in obs['effects'].items():
                actual=int(node in descendants and node!=source)
                if actual!=expected:
                    issues.append(f'Perturb({source}): {node} predicted {actual}, observed {expected}.')
    else:
        import numpy as np
        candidate='Structure:\n'+parsed.normalize().to_string()
        expected=benchmark._parse_observation(observations)
        actual=parsed.get_top_view()
        if actual.shape!=expected.shape:
            issues.append(f'Grid shape {actual.shape} differs from observed shape {expected.shape}.')
        else:
            for row,col in np.argwhere(actual!=expected):
                issues.append(f'Top-view row {row+1}, column {col+1}: predicted {actual[row,col]}, observed {expected[row,col]}.')
        if len(parsed.layers)>benchmark.metadata['max_height']:
            issues.append('Structure exceeds the maximum height.')
        for z in range(1,len(parsed.layers)):
            for row,col in np.argwhere((parsed.layers[z]==1)&(parsed.layers[z-1]!=1)):
                issues.append(f'Unsupported block at layer {z+1}, row {row+1}, column {col+1}; the cell directly below is empty.')
    if not valid and not issues:
        issues.append('Candidate violates the stated constraints; recheck each observation.')
    return candidate,issues[:3]
