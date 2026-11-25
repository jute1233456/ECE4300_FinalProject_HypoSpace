"""Keep verified interaction trajectories; these are not yet tokenizer-ready SFT tensors."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);args=p.parse_args()
    raw=[];new=[];duplicates=[]
    for method in ('mcts','two-step'):
        for path in sorted((args.run_dir/method).glob('*.json')):
            if path.name=='status.json' or '.tree.' in path.name or '.failed-' in path.name:continue
            d=json.loads(path.read_text())
            if not isinstance(d,dict) or not d.get('completed'):continue
            assert d.get('mcts_version')=='2.1'
            for case in d['cases']:
                seen=set();candidates=[]
                for i,r in enumerate(case['records'],1):
                    key=json.dumps(r.get('hypothesis_key'),sort_keys=True)
                    duplicate=r.get('valid') and key in seen
                    if r.get('valid'):seen.add(key)
                    sample=dict(method=method,version='2.1',domain=case['domain'],case_id=case['case_id'],
                        round=i,source=str(path),duplicate=bool(duplicate),record=r,
                        split='train',requires_tokenizer_and_loss_mask_validation=True)
                    raw.append(sample)
                    if r.get('valid') and r.get('reference_hit') and r.get('tool_trace') and r.get('text') and not r.get('protocol_error'):
                        candidates.append(sample)
                if candidates:
                    count=min(10,len(candidates))
                    chosen=[candidates[i*(len(candidates)-1)//max(1,count-1)] for i in range(count)]
                    for s in chosen:(duplicates if s['duplicate'] else new).append(s)
    selected=new+duplicates[:len(new)//4]
    dest=args.run_dir/'training_exports';dest.mkdir(exist_ok=True)
    for name,rows in [('all_trajectories',raw),('selected_trajectories',selected)]:
        with (dest/f'{name}.jsonl').open('w') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    (dest/'manifest.json').write_text(json.dumps(dict(all=len(raw),selected=len(selected),
        new=len(new),duplicates=min(len(duplicates),len(new)//4),
        note='Raw verified interactions only. Do not train directly without validating tokenizer template, reasoning isolation and assistant-only loss masks.'),indent=2))


if __name__=='__main__':main()
