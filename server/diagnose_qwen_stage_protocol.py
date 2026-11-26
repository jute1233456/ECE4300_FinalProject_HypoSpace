"""Replay captured failed selections, comparing exact old input with isolated stages."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import time

from llm.local_thinking import LocalThinkingClient
from llm.qwen_stage_protocol import isolate_stage
from llm.split_tree_tools import SplitTreeToolClient


def main():
    # Resolve Transformers' lazy import before creating concurrent clients.
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser();parser.add_argument('--variants',default='original,isolated')
    args=parser.parse_args();variants=args.variants.split(',')
    assert set(variants)<= {'original','isolated','examples'}
    root=Path('/opt/data/private/ECE4300 Final Project')
    out=root/'logs/qwen-protocol-diagnosis-20261126';out.mkdir(parents=True,exist_ok=True)
    source=root/'results/qwen-stage22-revision19-20261126'
    cases=[]
    for file in sorted(source.glob('boolean-*.json')):
        if '.tree.' in file.name:continue
        d=json.loads(file.read_text());c=(d.get('cases')or[d.get('interrupted_case',{})])[0]
        for r in c.get('records',[]):
            if (r.get('protocol_error')or'').startswith('selection:'):
                entry=next(a for a in r['api_requests'] if a.get('stage')=='selection')
                cases.append(dict(case=c['case_id'],request=entry['request']));break
        if len(cases)==3:break
    assert len(cases)==3
    (out/'inputs.json').write_text(json.dumps(cases,indent=2))
    def trial(case,variant,repeat):
        target=out/f"{case['case']}-{variant}-{repeat}.json"
        if target.exists():
            saved=json.loads(target.read_text())
            return {k:v for k,v in saved.items() if k!='response'}
        prompt=case['request']['prompt'];system=case['request']['system']
        if variant!='original':prompt,system=isolate_stage('selection',prompt,system,examples=variant=='examples')
        client=LocalThinkingClient(tokenizer_path=root/'models/Qwen3-8B',api_key='local',
            model='Qwen3-8B',base_url='http://127.0.0.1:8004/v1',timeout=3600,
            generation_options=dict(temperature=0.7,top_p=0.8,top_k=20,seed=4300+repeat,
                chat_template_kwargs={'enable_thinking':True}))
        started=time.monotonic();response=client.generate(prompt,system=system,max_tokens=32768)
        try:branch=SplitTreeToolClient.parse_branch(response.text.strip().splitlines()[0],1);valid=True
        except (ValueError,IndexError):branch=None;valid=False
        row=dict(case=case['case'],variant=variant,repeat=repeat,valid_branch=valid,branch=branch,
                 seconds=time.monotonic()-started,response=asdict(response))
        target.write_text(json.dumps(row,indent=2))
        return {k:v for k,v in row.items() if k!='response'}
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(trial,c,v,n) for c in cases for v in variants for n in range(2)]):
            row=f.result();rows.append(row);print(row,flush=True)
            (out/('summary.json' if variants==['original','isolated'] else 'summary-'+ '-'.join(variants)+'.json')).write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
