"""Create an immutable length-filtered view; preserve every retained source line verbatim."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def build(source, lengths, output, expected_sha, cap=14336):
    raw=Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('Source checksum differs from audited dataset')
    report=json.loads(Path(lengths).read_text(encoding='utf-8'))
    if report['template_prefix_mismatches'] or report['skipped_missing_roles']:
        raise ValueError('Length audit has template failures')
    measured={row['sample_id']:row for row in report['details']}
    kept=[];dropped=[];kept_rows=[];excluded=[];ids=set()
    for line in raw.splitlines(keepends=True):
        if not line.strip():continue
        row=json.loads(line);identity=row['sample_id']
        if identity in ids or identity not in measured:
            raise ValueError('Duplicate or unmeasured sample')
        ids.add(identity);detail=measured[identity]
        if (row['case_id'],row['stage'])!=(detail['case_id'],detail['stage']):
            raise ValueError('Length audit does not match sample metadata')
        if detail['full_tokens']<=cap:
            kept.append(line);kept_rows.append(row)
        else:
            dropped.append(line);excluded.append(detail)
    if ids!=set(measured):raise ValueError('Length audit has extra samples')
    output=Path(output)
    if output.exists():raise ValueError('Refusing to overwrite a frozen dataset')
    output.mkdir(parents=True)
    data=b''.join(kept);(output/'train.jsonl').write_bytes(data)
    (output/'excluded.long.jsonl').write_bytes(b''.join(dropped))
    manifest=dict(version='original-format-pruned-sft-v1',source=str(source),source_sha256=expected_sha,
        length_audit_sha256=hashlib.sha256(Path(lengths).read_bytes()).hexdigest(),max_length=cap,
        source_samples=len(ids),train_samples=len(kept),excluded_samples=len(dropped),
        train_sha256=hashlib.sha256(data).hexdigest(),
        by_stage=dict(Counter(x['stage'] for x in kept_rows)),
        by_case=dict(Counter(x['case_id'] for x in kept_rows)),
        excluded_by_case=dict(Counter(x['case_id'] for x in excluded)),
        max_retained_tokens=max(measured[x['sample_id']]['full_tokens'] for x in kept_rows),
        excluded=excluded,notes='Retained rows byte-identical; no summarization, compression, truncation or whole-round deletion.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    longest=max(kept_rows,key=lambda x:measured[x['sample_id']]['full_tokens'])
    line=next(line for line in kept if json.loads(line)['sample_id']==longest['sample_id'])
    (output/'longest.jsonl').write_bytes(line)
    return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--lengths',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--max-length',type=int,default=14336)
    a=p.parse_args();m=build(a.source,a.lengths,a.output,a.expected_sha,a.max_length)
    print(json.dumps({k:v for k,v in m.items() if k!='excluded'},indent=2))


if __name__=='__main__':main()
