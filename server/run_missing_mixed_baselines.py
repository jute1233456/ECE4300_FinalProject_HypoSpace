"""Fill only missing direct/memory baselines, behind the existing validation workers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from server.poolctl import call
from server.run_finetune_quick6 import atomic, case_result
from server.run_dynamic_validation import process_matches


def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=args.config.resolve().parent
    lock=(root/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=Path(cfg['manifest']);suite=json.loads(manifest.read_text());state_path=root/'tasks.json'
    if state_path.exists():tasks=json.loads(state_path.read_text())
    else:
        tasks=[dict(id=job['id']+'/'+method,job=job,method=method,status='pending',attempts=0)
               for job in suite['jobs'] for method in ['direct','memory'] if job['baselines'][method]['status']=='pending']
    def target(task):
        return root/'results'/task['method']/(task['job']['dataset']+'-'+task['job']['case_id']+'.json')
    handles={}
    for task in tasks:
        if task['status']=='running' and not process_matches(task):task['status']='pending'
    while True:
        for task in tasks:
            p=target(task)
            if p.exists():
                result=case_result(json.loads(p.read_text()),task['job']['references'])
                if result['completed']:task.update(status='completed',result=result);continue
            if task['status']=='running':
                handle=handles.get(task['id']);ended=handle.poll() is not None if handle else not process_matches(task)
                if ended:task['status']='pending' if task['attempts']<2 else 'needs_inspection'
        live=sum(t['status']=='running' for t in tasks)
        primary_path=Path(cfg['primary_state']);primary=json.loads(primary_path.read_text())
        # Reserve a slot for every current main-queue worker, including workers
        # between stages. Do not take capacity when its state is stale.
        stale=time.time()-primary_path.stat().st_mtime>120
        health=call(cfg['gateway']+'/health')
        available=max(0,health['capacity']['baseline']-len(primary.get('running',[]))-live) if not stale else 0
        for task in tasks:
            if available<=0:break
            if task['status']!='pending':continue
            job=task['job'];p=target(task);p.parent.mkdir(parents=True,exist_ok=True)
            command=[cfg['python'],'-u','-m','experiments.test_easy','--model','Qwen3-8B',
                '--enable-thinking','--base-url',cfg['gateway']+'/v1','--max-tokens','32768',
                '--dataset-root',str(manifest.parent/job['dataset_root']),'--dataset',job['dataset'],
                '--case-id',job['case_id'],'--method',task['method'],'--pause-file',str(root/'PAUSE'),
                '--resume' if p.exists() else '--output',str(p)]
            with p.with_suffix('.log').open('a') as log:
                worker=subprocess.Popen(command,cwd=cfg['eval_code'],stdout=log,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True,
                    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false'))
            # Popen exec can momentarily expose an empty argv; never record it.
            for _ in range(100):
                raw=(Path('/proc')/str(worker.pid)/'cmdline').read_bytes()
                if raw:break
                time.sleep(.01)
            if not raw:raise RuntimeError('Missing worker identity')
            task.update(status='running',pid=worker.pid,cmd_hash=hashlib.sha256(raw).hexdigest(),attempts=task['attempts']+1)
            handles[task['id']]=worker;available-=1
        atomic(state_path,tasks)
        done=sum(t['status']=='completed' for t in tasks)
        terminal=all(t['status'] in ['completed','needs_inspection'] for t in tasks)
        atomic(root/'state.json',dict(pid=os.getpid(),phase=('completed' if done==len(tasks) else 'needs_inspection') if terminal else 'running',
            completed=done,total=len(tasks),running=[t['id'] for t in tasks if t['status']=='running'],
            main_workers=len(primary.get('running',[])),main_state_stale=stale,max_tokens=32768))
        if terminal:break
        time.sleep(5)


if __name__=='__main__':main()
