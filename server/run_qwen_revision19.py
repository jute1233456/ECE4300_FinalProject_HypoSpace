"""Exactly nineteen cases, native thinking, revised Stage-MCTS, self-hosted pool only."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from hashlib import sha256
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-name',default='qwen-stage22-revision19-20261126')
    parser.add_argument('--protocol-label',default='stage22_revision')
    options=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',options.run_name):parser.error('Invalid run name')
    release=Path(__file__).resolve().parents[1]
    root=release.parents[1]
    output=root/'results'/options.run_name
    output.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(output/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pin=json.loads((release/'release_manifest.json').read_text())
    for name,digest in pin['sha256'].items():
        assert sha256((release/name).read_bytes()).hexdigest()==digest,name
    data=json.loads((release/'datasets/sampled.json').read_text())
    groups=[[dict(dataset='boolean_extended' if x['case_id'].startswith('extended_') else ds,
                  case_id=x['case_id'],references=x['references']) for x in data[ds]]
            for ds in ('causal','3d','boolean')]
    from itertools import zip_longest
    jobs=[j for row in zip_longest(*groups) for j in row if j]
    assert len(jobs)==19
    manifest=dict(model='Qwen3-8B',mcts_version='2.2',stage22_revision=True,
        protocol_revision=options.protocol_label,
        provider='self-hosted',base_url='http://127.0.0.1:8004/v1',workers=9,
        max_tokens=32768,thinking=True,jobs=jobs,source_release=str(release),
        total=19,total_references=sum(j['references'] for j in jobs))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    state=dict(pid=os.getpid(),status='running',completed=0,total=19,workers=9,
               protocol_revision=options.protocol_label,
               provider='self-hosted',model='Qwen3-8B',statuses=[])
    def save():
        p=output/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(output/'state.json')
    save()
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
    env.pop('ARK_API_KEY',None)
    def run(job):
        target=output/f"{job['dataset']}-{job['case_id']}.json"
        args=[sys.executable,'-u','-m','experiments.test_hard_balanced',
            '--model','Qwen3-8B','--enable-thinking','--base-url',manifest['base_url'],
            '--max-tokens','32768','--dataset',job['dataset'],'--case-id',job['case_id'],
            '--method','mcts-tree','--tree-policy','llm','--duplicate-reward','-1',
            '--exploration-list','off','--two-step-reflection','--unbounded-tree',
            '--tree-tools','--mcts-version','2.2','--stage22-revision',
            '--pause-file',str(output/'PAUSE')]
        if target.exists():
            d=json.loads(target.read_text())
            if d.get('completed'):return dict(**job,status='completed')
            args+=['--resume',str(target)]
        else:args+=['--output',str(target)]
        with target.with_suffix('.log').open('a') as log:
            code=subprocess.run(args,cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:return dict(**job,status='process_error',exit_code=code)
        d=json.loads(target.read_text());c=(d.get('cases')or[{}])[0]
        return dict(**job,status='completed' if d.get('completed') and c.get('stop_reason')!='request_errors' else 'needs_inspection',
                    coverage=c.get('metrics',{}).get('recovery'))
    with ThreadPoolExecutor(max_workers=9) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            try:result=f.result()
            except Exception as error:result=dict(status='exception',exception=type(error).__name__)
            state['statuses'].append(result);state['completed']=sum(x['status']=='completed' for x in state['statuses'])
            save();print(result,flush=True)
    state['status']='complete' if state['completed']==19 else 'needs_inspection';save()


if __name__=='__main__':main()
