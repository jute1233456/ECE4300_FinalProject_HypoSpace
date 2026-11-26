"""Serialize inputs, associated DS answers and all target-model attempts. No data decisions."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import tarfile

from dpo.common import atomic,read_jsonl,write_jsonl


def export_records(root,archive=False):
    states=read_jsonl(root/'states.jsonl');by_state=defaultdict(list);errors=0
    if any('split' in s for s in states):raise ValueError('Partitioned v1 inputs are obsolete; prepare collection-v2 inputs')
    for path in sorted((root/'responses').glob('*.json')):
        response=json.loads(path.read_text(encoding='utf-8'))
        if response['response_status']!='received':errors+=1
        by_state[response['state_id']].append(response)
    rows=[dict(**state,target_responses=sorted(by_state[state['state_id']],key=lambda r:(r['sample_index'],r['attempt'],r['finished_utc']))) for state in states]
    write_jsonl(root/'records.jsonl',rows)
    counts=dict(inputs=len(rows),ds_answers=sum(len(r.get('ds_answers',[])) for r in rows),target_attempts=sum(len(r['target_responses']) for r in rows))
    atomic(root/'export_summary.json',dict(format='fixed input plus DS history plus all target model attempts',counts=counts,
        error_attempts=errors,notes=['No scores, chosen/rejected labels or preference pairs have been created.',
        'No train/validation split is performed. No attempt is dropped based on its content or finish reason.',
        'Historical DS input alignment limitations are recorded, not resolved by the collector.',
        'Read the matching questions/ file for task rules and references. All attempts remain under responses/.']))
    if archive:
        temp=root/'collected-responses.tar.gz.tmp'
        with tarfile.open(temp,'w:gz') as tar:
            for name in ['manifest.json','generation_recipe.json','states.jsonl','contexts','questions','responses','records.jsonl','export_summary.json']:
                if (root/name).exists():tar.add(root/name,arcname=name)
        temp.replace(root/'collected-responses.tar.gz')
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',action='store_true');a=p.parse_args()
    print(json.dumps(export_records(a.root,a.archive),indent=2))
