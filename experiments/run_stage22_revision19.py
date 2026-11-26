"""Run exactly the nineteen sampled cases on a frozen Stage-2.2 revision."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Thread

from llm.ark_coding_plan import ArkCodingPlanClient
from .rate_limit import recover_quota


ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=9)
    args=parser.parse_args()
    if not 1<=args.workers<=19:parser.error('Use 1 to 19 workers')
    client=ArkCodingPlanClient()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=ROOT/f'results/stage22-revision19-{stamp}'
    release=ROOT/f'local-backups/stage22-revision19-{stamp}'
    output.mkdir(parents=True);release.mkdir(parents=True)
    for folder in ('agents','llm','experiments','datasets','HypoSpace'):
        shutil.copytree(ROOT/folder,release/folder,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
    (release/'config').mkdir()
    # Existing API key stays in the process environment; the snapshot contains no credential.
    (release/'config/llm.json').write_text(json.dumps(dict(ark_coding_plan=dict(
        model=client.model,timeout=client.timeout))),encoding='utf-8')
    pins={str(p.relative_to(release)):sha256(p.read_bytes()).hexdigest()
          for p in release.rglob('*') if p.is_file() and p.suffix in ('.py','.json')}
    sampled=json.loads((release/'datasets/sampled.json').read_text(encoding='utf-8'))
    groups=[]
    for domain in ('causal','3d','boolean'):
        groups.append([dict(dataset='boolean_extended' if row['case_id'].startswith('extended_') else domain,
                            case_id=row['case_id'],references=row['references']) for row in sampled[domain]])
    from itertools import zip_longest
    jobs=[job for row in zip_longest(*groups) for job in row if job]
    assert len(jobs)==19 and len({(j['dataset'],j['case_id']) for j in jobs})==19
    manifest=dict(mcts_version='2.2',revision='stage22_revision',model=client.model,
        provider='Ark Coding Plan',workers=args.workers,jobs=jobs,total=19,
        total_references=sum(j['references'] for j in jobs),source_release=str(release),
        source_sha256=pins,started_utc=stamp,generation_limit='provider-token-limit',
        thinking='provider native reasoning',training=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    state=dict(pid=os.getpid(),status='running',completed=0,total=19,workers=args.workers,
               model=client.model,output=str(output),source_release=str(release),statuses=[])
    def save():
        p=output/'state.tmp';p.write_text(json.dumps(state,indent=2),encoding='utf-8');p.replace(output/'state.json')
    save()
    Thread(target=recover_quota,args=(output/'PAUSE',),daemon=True).start()
    env=dict(os.environ,ARK_API_KEY=client._api_key,PYTHONUTF8='1')
    def run(job):
        target=output/f"{job['dataset']}-{job['case_id']}.json"
        command=[sys.executable,'-u','-m','experiments.test_hard_balanced',
            '--model',client.model,'--dataset',job['dataset'],'--case-id',job['case_id'],
            '--provider-token-limit','--method','mcts-tree','--tree-policy','llm',
            '--duplicate-reward','-1','--exploration-list','off','--two-step-reflection',
            '--unbounded-tree','--tree-tools','--mcts-version','2.2','--stage22-revision',
            '--pause-file',str(output/'PAUSE'),'--output',str(target)]
        with target.with_suffix('.log').open('a',encoding='utf-8') as log:
            code=subprocess.run(command,cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if code:return dict(**job,status='process_error',exit_code=code)
        d=json.loads(target.read_text(encoding='utf-8'));c=(d.get('cases')or[{}])[0]
        return dict(**job,status='completed' if d.get('completed') and c.get('stop_reason')!='request_errors'
                    else 'needs_inspection',coverage=c.get('metrics',{}).get('recovery'))
    print('REVISION19_STARTED',str(output),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            try:result=future.result()
            except Exception as error:result=dict(status='exception',exception=type(error).__name__)
            state['statuses'].append(result)
            state['completed']=sum(s['status']=='completed' for s in state['statuses'])
            save();print(result,flush=True)
    state['status']='complete' if state['completed']==19 else 'needs_inspection';save()


if __name__=='__main__':main()
