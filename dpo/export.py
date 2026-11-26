"""Export raw, UNREVIEWED outputs only. Never create scores or preference pairs."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import tarfile

from dpo.common import atomic,read_jsonl,write_jsonl


def export_review(root,archive=False):
    states=read_jsonl(root/'states.jsonl');by_state=defaultdict(dict);errors=0
    for path in sorted((root/'responses').glob('*.json')):
        response=json.loads(path.read_text(encoding='utf-8'))
        if response['response_status']!='received':errors+=1;continue
        existing=by_state[response['state_id']].get(response['sample_id'])
        if not existing or response['finished_utc']>existing['finished_utc']:by_state[response['state_id']][response['sample_id']]=response
    folder=root/'review';folder.mkdir(exist_ok=True);counts={}
    for split in ['train','validation']:
        rows=[]
        for state in states:
            outputs=list(by_state[state['state_id']].values())
            if state['split']==split and outputs:rows.append(dict(**state,outputs=sorted(outputs,key=lambda x:x['sample_index']),review_status='unreviewed'))
        write_jsonl(folder/(split+'.unscored.jsonl'),rows)
        counts[split]=dict(states=len(rows),responses=sum(len(r['outputs']) for r in rows))
    atomic(folder/'manifest.json',dict(format='fixed prompt plus multiple raw stage responses',review_status='unreviewed',counts=counts,
        error_attempts=errors,notes=['No scores, chosen/rejected labels or preference pairs have been created.',
        'Each state is fixed across samples; train/validation are disjoint by question family.',
        'Holdout questions were seen in SFT; this is only a DPO-stage holdout.',
        'Read the matching questions/ file for task rules and references. All attempts remain under responses/.']))
    if archive:
        temp=root/'cc-review.tar.gz.tmp'
        with tarfile.open(temp,'w:gz') as tar:
            for name in ['manifest.json','generation_recipe.json','states.jsonl','contexts','questions','responses','review']:
                if (root/name).exists():tar.add(root/name,arcname=name)
        temp.replace(root/'cc-review.tar.gz')
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--archive',action='store_true');a=p.parse_args()
    print(json.dumps(export_review(a.root,a.archive),indent=2))
