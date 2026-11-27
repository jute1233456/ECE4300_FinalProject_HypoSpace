"""Build preferences from a frozen raw collection and Codex's independent review.

No CC datasets, judgments, or manifests are inputs. Never launches training.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


# Explicit, independently authored corrections, not generated model responses.
SUPPLEMENTS = {
    19: ('A', 'All eight supplied structures are valid, with five repeated outcomes. Repeating a supported structure is a novelty failure, not a physical-support failure. Adjacency of tall columns is not required.',
         'Enumerate unvisited ordered height assignments over the fixed footprint. Keep permutations of the same height multiset distinct unless their complete spatial assignments are identical.',
         'A invents mandatory adjacency and a fixed tall-column count, absent from the rules.'),
    45: ('A', 'The supplied structures are physically valid; seven of eight outcomes are repeats. Changing the bottom footprint cannot solve this novelty problem because the footprint is forced by the observation and support.',
         'Keep the footprint fixed and enumerate ordered column heights, comparing the complete spatial height assignment against prior attempts before constructing layers.',
         'A permits extra bottom blocks despite observed top-view zeros.'),
    53: ('A', 'All eight outcomes are valid and novel. They already demonstrate nonuniform supported heights, so there is no evidence that support, uniform height, or adjacency is a failed constraint.',
         'Track the ordered heights at the four fixed positions and enumerate unvisited assignments. Do not rotate the footprint or reject different assignments merely because their sorted height multisets match.',
         'A leaves an observed occupied column empty at ground level but occupied above, violating support.'),
    81: ('E', 'The history must be evaluated under structural uniqueness rather than whole-function equivalence. A NOR with an AND operand and a NOR operand has a different truth table from a NOT applied to a NOR; their validity at the single observed row does not make them identical.',
         'Compare normalized operator trees using only the listed equivalences, then explore shallow unvisited shapes with a separate observed-row truth check.',
         'E conflates XOR-shaped NOR(AND,NOR) with OR-shaped NOT(NOR).'),
    83: ('D', 'Literal disjunctions and nested NOR forms must not be merged solely because they agree at the observed row. No additional unobserved-row output is required.',
         'Check the specified observation, gate depth, and structural signature independently; broaden to a shallow operator shape whose normalized tree is absent from the supplied attempts.',
         'D equates expressions whose full truth tables differ.'),
    88: ('A', 'The rejected flattened disjunction does not establish an extra rule banning all tautologies. The history explicitly contains an accepted complementary-literal disjunction, so semantic-constant pruning would remove valid search directions.',
         'Check parser acceptance and depth separately from structural novelty. Keep only the stated normalization rules and explore a different shallow root shape without imposing a new truth-table deduplication rule.',
         'A invents semantic-tautology/truth-table pruning unsupported by the stated equivalences.'),
    97: ('C', 'The supplied history contains several rejected flat disjunctions and an accepted NOR with AND and NOR operands. Resubmitting that accepted operand pair would repeat a known expression rather than expand coverage.',
         'Separate the rejected-syntax investigation from novelty. Explore another shallow operator shape, checking both the observed row and exact normalized-tree novelty before submission.',
         'C proposes the first accepted NOR/AND/NOR expression again and misstates its observed-row evaluation.'),
    101: ('B', 'The latest conjunction repeats an earlier conjunction with redundant parentheses. The history also already includes a disjunction whose operands are the literal y and a primitive NOR over the two variables.',
          'Canonicalize redundant parentheses and the permitted associative, commutative, and idempotent forms before choosing an unvisited shallow tree; avoid resubmitting either known shape.',
          'B explicitly proposes the already accepted y OR NOR(x,y).'),
    102: ('A', 'The rejected NOR with an OR operand containing a negated literal has a three-gate path. Primitive NOR is allowed but still must satisfy the observed row; a NOR over the two bare variables is zero at the required input.',
          'Count primitive NOR as one gate and any nested negation as another. Explore a different depth-compliant shape, then check the observation and its normalized signature against the supplied history.',
          'A calls bare NOR(x,y) a novel valid candidate although it fails the observation.'),
    103: ('B', 'Recent history contains accepted nested-NOR shapes and a rejected NOR with a deeper negated operand. Replacing these with a bare two-variable NOR would violate the required output at the observed input.',
          'Retain the observation check when changing root families, count primitive NOR as one gate, and compare the complete normalized tree against known expressions before accepting a new direction.',
          'B proposes bare NOR(x,y), false at the required observed row.'),
    134: ('E', 'The complex NOT-over-AND attempt exceeds the two-level gate-depth limit, while the latest NOT-over-NOR is valid. An AND containing NOT y can never output one at the required row where y is one.',
          'Use observed-row truth propagation to reject impossible operand restrictions first, then explore shallow trees with at most one further gate below the root and verify structural novelty.',
          'E requires NOT y as an AND operand at the observation y=1, making the required output impossible.'),
    141: ('B', 'The two rejected AND-root forms match the observed rows but contain a NOT below another second-level gate. Their rejection does not establish a semantic-tautology ban. The OR-with-AND sibling is already accepted.',
          'Explore another unvisited shallow shape, treating a primitive NOR as one gate, and test all three specified rows before checking structural novelty. Leave the unspecified row unconstrained.',
          'B proposes NOR(AND(x,y),OR(x,y)), which fails the required observation pattern.'),
}

# These otherwise useful originals submit a complete new candidate, contain an
# uncertain extra rule, or have a misleading explanation. Keep originals in the
# audit, but author a reflection-only positive for the current protocol.
REWRITE = {6, 8, 12, 69, 70, 73, 76, 77, 80, 85, 87, 90, 92, 94, 95, 96, 99,
           107, 114, 130, 131, 132, 138, 139, 142, 149, 179, 197}


def history(state):
    content = state['messages'][-1]['content']
    data = json.JSONDecoder().raw_decode(content[content.index('{'):])[0]
    context = data['problem_and_search_context']
    marker = 'Observed outcomes: '
    return json.JSONDecoder().raw_decode(context.split(marker, 1)[1])[0] if marker in context else []


def constraint(state):
    if state['domain'] == '3d':
        return ('Layer 1 must equal the observed footprint. Every occupied column is a continuous stack of height one to three; zero footprint cells stay empty in every layer. Compare ordered spatial heights, not sorted height multisets, and omit empty top layers.')
    if state['domain'] == 'boolean':
        allowed = ('AND, OR, and NOT' if 'extended_' in state['case_id'] else 'AND, OR, NOT, and primitive NOR')
        return (f'Use only x, y and {allowed}, with no literal constants and at most two gates on any root-to-leaf path. Match only the specified observations. Normalize only the stated structural equivalences, without absorption or De Morgan rewriting.')
    return ('Preserve every observed descendant set, acyclicity, and the ten-edge cap. A zero perturbation row constrains outgoing reachability, not incoming edges. Redundant transitive shortcuts are allowed; uniqueness compares exact edge sets.')


def direction(state):
    return {
        '3d': 'Enumerate an unvisited ordered height assignment on the fixed footprint, derive layers by height thresholds, and check its complete spatial signature against all supplied attempts before submission.',
        'boolean': 'Explore a different shallow operator-tree shape. Check the observed truth values, longest gate path, and normalized-tree novelty separately before constructing the next candidate; do not infer exhaustion from a valid answer.',
        'causal': 'Keep the forced reachability backbone and enumerate an unvisited optional-shortcut subset, using an exact edge-set novelty check and recomputing descendant closure and edge count before submission.',
    }[state['domain']]


def authored(state, diagnosis=None, next_direction=None):
    outcomes = history(state)
    if diagnosis is None:
        if not outcomes:
            diagnosis = 'The supplied attempt history is empty, so there is no evidenced failed branch or repeated answer to diagnose.'
        else:
            valid = sum(bool(x['valid']) for x in outcomes)
            repeated = sum(x.get('reward') == -1 for x in outcomes)
            diagnosis = (f'Among the {len(outcomes)} supplied outcomes, {valid} are valid and {repeated} have repeated-answer reward -1. Distinguish a repeated valid answer from a constraint failure; neither an accepted answer nor this recent history proves exhaustion.')
    return f'Review: {diagnosis}\n\nConstraint check: {constraint(state)}\n\nNext direction: {next_direction or direction(state)}'


def format_only(text, state):
    """Label edits explicitly; never overwrite the original response evidence."""
    text = re.sub(r'\*\*([^*]+:)\*\*', r'\1', text)
    text = re.sub(r'\bNext (?:actionable adjustment|adjustment|constraint-free direction):', 'Next direction:', text, flags=re.I)
    text = re.sub(r'\bConstraint to (?:check|verify|recheck)(?: next| explicitly)?:', 'Constraint check:', text, flags=re.I)
    if not text.startswith('Review:'):
        text = 'Review: ' + text
    if 'Constraint check:' not in text:
        text += '\n\nConstraint check: ' + constraint(state)
    if 'Next direction:' not in text:
        text += '\n\nNext direction: ' + direction(state)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    args = ap.parse_args()
    root = args.root.resolve()
    out = root / 'dataset'
    if out.exists():
        raise FileExistsError('Use a fresh output directory; do not overwrite released datasets')
    out.mkdir()
    snapshot = json.loads((root/'snapshot.json').read_text())
    states = read(root/'states.jsonl')
    review_states = read(root/'review_states.jsonl')
    reviews = read(root/'independent_reviews.jsonl')
    assert len(reviews) == len(review_states) == 206
    assert sorted(r['index'] for r in reviews) == list(range(len(review_states)))
    state_by_id = {s['state_id']: s for s in states}
    assert len(state_by_id) == len(states) == snapshot['states']
    responses = {}
    excluded_responses = []
    for item in snapshot['sources']:
        path = root/'responses'/item['name']
        assert sha(path) == item['sha256'], path.name
        row = json.loads(path.read_text())
        responses[path.name] = row
        result = row.get('response') or {}
        reason = []
        if row.get('response_status') != 'received': reason.append('not_received')
        if result.get('finish_reason') != 'stop': reason.append('not_stop')
        if not result.get('final_text', '').strip(): reason.append('empty_final')
        if reason: excluded_responses.append({'filename': path.name, 'state_id': row['state_id'], 'reasons': reason})
    assert len(responses) == snapshot['response_files']
    rows, raw_rows, manifests, skipped, manual = [], [], [], [], []
    used_prompts = set()
    for review in sorted(reviews, key=lambda x:x['index']):
        index = review['index']; state = review_states[index]
        assert state['state_id'] == review['state_id']
        original = state_by_id[state['state_id']]
        assert state['messages'] == original['messages']
        assert digest(state['messages']) == state['input_sha256'] == original['input_sha256']
        candidates = {c['label']: c for c in state['candidates']}
        assert set(review['reviewed_labels']) == set(candidates)
        for c in candidates.values():
            if c['source'] == 'qwen':
                raw = responses[c['filename']]
                assert raw['state_id'] == state['state_id'] and raw['input_sha256'] == state['input_sha256']
                assert raw['response_status'] == 'received' and raw['response']['finish_reason'] == 'stop'
                assert raw['response']['final_text'] == c['text']
                assert raw['checkpoint_sha256'] == c['checkpoint_sha256']
            else:
                assert c['source'] == 'ds'
                assert original['ds_answers'][c['index']]['text'] == c['text']
        positive = candidates.get(review['chosen_label'])
        if index in SUPPLEMENTS:
            bad, diagnosis, adjustment, evidence = SUPPLEMENTS[index]
            negative = candidates[bad]
            chosen_text = authored(state, diagnosis, adjustment)
            origin = 'codex_authored_missing_positive'
            reason = evidence
        elif positive:
            negative = candidates[review['rejected_label']]
            chosen_text = authored(state) if index in REWRITE else format_only(positive['text'], state)
            origin = ('codex_authored_protocol_replacement' if index in REWRITE else
                      positive['source'] if chosen_text == positive['text'] else 'codex_format_edit_'+positive['source'])
            reason = review['reason']
        else:
            skipped.append(dict(review, category='no_high_confidence_opposing_pair'))
            continue
        if state['input_sha256'] in used_prompts:
            skipped.append(dict(review, category='duplicate_exact_input'))
            continue
        used_prompts.add(state['input_sha256'])
        assert chosen_text != negative['text'] and chosen_text.strip() and negative['text'].strip()
        assert not re.search(r'</?think\b', chosen_text + negative['text'], re.I)
        assert all(h in chosen_text for h in ('Review:', 'Constraint check:', 'Next direction:'))
        pair = dict(prompt=state['messages'], chosen=[dict(role='assistant', content=chosen_text)],
                    rejected=[dict(role='assistant', content=negative['text'])])
        rows.append(pair)
        if positive:
            raw_rows.append(dict(prompt=state['messages'], chosen=[dict(role='assistant', content=positive['text'])],
                                 rejected=[dict(role='assistant', content=negative['text'])]))
        provenance = {k:v for k,v in negative.items() if k!='text'}
        good_provenance = {k:v for k,v in positive.items() if k!='text'} if positive else None
        manifests.append(dict(pair_index=len(rows)-1, review_index=index, state_id=state['state_id'],
                              case_id=state['case_id'], family_id=state['family_id'], domain=state['domain'],
                              input_sha256=state['input_sha256'], chosen_sha256=digest(chosen_text),
                              rejected_sha256=digest(negative['text']), chosen_origin=origin,
                              chosen_original=good_provenance, rejected_original=provenance,
                              chosen_confidence='high', rejected_confidence='high', reason=reason,
                              ds_original_wire_input_available=False if positive and positive['source']=='ds' else None,
                              cc_labels_used=False))
        if origin.startswith('codex_'):
            manual.append(dict(state_id=state['state_id'], review_index=index, origin=origin,
                               original_text=positive['text'] if positive else None, text=chosen_text,
                               authorization='User permits manually authored helpful positives when needed; stage-format edits and replacements are explicit.'))
    skipped.extend(dict(s, category='insufficient_distinct_complete_answers') for s in read(root/'ineligible.jsonl'))
    assert len(rows) + len(skipped) == len(states)
    write(out/'dpo.jsonl', rows)
    write(out/'raw_selected_pairs.audit_only.jsonl', raw_rows)
    write(out/'pair_manifest.jsonl', manifests)
    write(out/'skipped.jsonl', skipped)
    write(out/'authored_and_edited_positives.jsonl', manual)
    write(out/'excluded_responses.jsonl', excluded_responses)
    shutil.copy2(root/'independent_reviews.jsonl', out/'independent_reviews.jsonl')
    shutil.copy2(root/'snapshot.json', out/'snapshot.json')
    manifest = dict(name='independent-reflection-dpo-v1', snapshot_utc=datetime.fromtimestamp(snapshot['snapshot_utc'], timezone.utc).isoformat(),
                    raw_states=len(states), raw_qwen_response_files=len(responses), independently_reviewed_states=len(reviews),
                    reviewed_distinct_answers=sum(len(s['candidates']) for s in review_states),
                    pairs=len(rows), skipped_states=len(skipped),
                    by_domain=dict(Counter(m['domain'] for m in manifests)),
                    chosen_origins=dict(Counter(m['chosen_origin'] for m in manifests)),
                    rejected_sources=dict(Counter(m['rejected_original']['source'] for m in manifests)),
                    manual_missing_positive=len(SUPPLEMENTS), protocol_replacements=len(REWRITE),
                    excluded_response_files=len(excluded_responses), cc_labels_used=False,
                    split_created=False, training_started=False, reference_solutions_in_completions=False,
                    source_collection=snapshot['source'], input_files=['states.jsonl','review_states.jsonl','independent_reviews.jsonl','snapshot.json','responses/*.json'],
                    notes=['Only reflection stage, not selection/expansion or end-to-end reward evidence.',
                           'DS responses are historical answers applied to reconstructed equivalent states; original DS wire messages were not preserved.',
                           'Preference labels are independent Codex judgments, not externally verified quality scores.',
                           'No live collection data created after snapshot is silently included.',
                           'Raw selected pairs are audit only; use dpo.jsonl, with explicit authored/format-edited positive provenance.'])
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'checks.json').write_text(json.dumps(dict(raw_hashes_verified=True,all_states_accounted=True,all_reviewable_states_read=True,
        exact_prompt_match=True,verbatim_rejected=True,original_positive_preserved_in_audit=True,chosen_edits_explicit=True,
        one_pair_per_unique_input=True,no_thinking_in_completions=True,no_split=True,no_cc_judgments=True,
        tokenizer_validation='pending'),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
