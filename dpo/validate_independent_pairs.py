"""CPU-only validation using the same chat/mask boundary as server.train_dpo."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from transformers import AutoTokenizer


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset', type=Path)
    ap.add_argument('--model', required=True)
    ap.add_argument('--max-length', type=int, default=8192)
    ap.add_argument('--quick3', type=Path)
    ap.add_argument('--code-root', type=Path)
    args = ap.parse_args()
    root = args.dataset
    rows = lines(root/'dpo.jsonl')
    manifests = lines(root/'pair_manifest.jsonl')
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    lengths = []
    side_counts = Counter()
    for i, (row, metadata) in enumerate(zip(rows, manifests, strict=True)):
        assert metadata['pair_index'] == i
        assert set(row) == {'prompt', 'chosen', 'rejected'}
        prefix = tok.apply_chat_template(row['prompt'], tokenize=True,
                                          add_generation_prompt=True, enable_thinking=False)
        record = {'pair_index': i, 'state_id': metadata['state_id'], 'prompt_tokens': len(prefix)}
        for side in ('chosen','rejected'):
            assert len(row[side]) == 1 and row[side][0]['role'] == 'assistant'
            full = tok.apply_chat_template(row['prompt']+row[side], tokenize=True, enable_thinking=False)
            assert full[:len(prefix)] == prefix and len(full)>len(prefix), (i, side, 'loss mask mismatch')
            assert len(full) <= args.max_length, (i, side, len(full), 'no truncation permitted')
            record[side+'_tokens'] = len(full)-len(prefix)
            record[side+'_sequence_tokens'] = len(full)
            side_counts[side] += 1
        lengths.append(record)
    checks = json.loads((root/'checks.json').read_text())
    checks.update(tokenizer_validation='passed', pairs=len(rows), checked_sequences=dict(side_counts),
                  model=args.model, enable_thinking=False, max_allowed_length=args.max_length,
                  max_prompt_tokens=max(x['prompt_tokens'] for x in lengths),
                  max_sequence_tokens=max(x[s+'_sequence_tokens'] for x in lengths for s in ('chosen','rejected')),
                  assistant_mask_matches_training_code=True, truncated_sequences=0)
    if args.quick3:
        assert args.code_root, '--code-root is required to use the actual benchmark family canonicalization'
        sys.path.insert(0, str(args.code_root))
        from server.prepare_quick_validation import family_key
        quick = json.loads(args.quick3.read_text())
        families = {j['family_sha256'] for j in quick['jobs']}
        # Collection IDs hash canonical JSON; quick3 hashes repr(tuple). Never
        # compare these different hash encodings directly.
        train_families = set()
        for case_id in {m['case_id'] for m in manifests}:
            question = json.loads((root.parent/'questions'/f'{case_id}.json').read_text())
            key = family_key(question['question'], question['domain'], question['metadata'])
            train_families.add(hashlib.sha256(repr(key).encode()).hexdigest())
        overlap = sorted(train_families & families)
        cases = {(j['dataset'],j['case_id']) for j in quick['jobs']}
        exact = sorted({(m['domain'],m['case_id']) for m in manifests} & cases)
        assert not overlap and not exact, (overlap, exact)
        checks['quick3_exact_case_overlap'] = exact
        checks['quick3_family_id_overlap'] = overlap
        checks['quick3_family_check'] = 'same family_key; SHA256(repr(tuple)) on both sides'
    (root/'lengths.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in lengths))
    (root/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    files = sorted(p for p in root.iterdir() if p.is_file() and p.name!='SHA256SUMS')
    (root/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in files))
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
