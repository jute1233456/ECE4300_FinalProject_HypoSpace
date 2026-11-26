"""Persistent quick6 queue; concurrency follows live pool capacity, GPU services are external."""
import argparse
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import time

from server.poolctl import call, change_role
from server.run_finetune_quick6 import atomic, case_result


def process_matches(task):
    try:
        raw=(Path('/proc')/str(task['pid'])/'cmdline').read_bytes()
        return bool(raw) and hashlib.sha256(raw).hexdigest()==task['cmd_hash']
    except (FileNotFoundError,KeyError):return False


def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=args.config.resolve().parent
    lock=(root/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tasks_path=root/'tasks.json';quick=json.loads(Path(cfg['quick_manifest']).read_text())
    code=Path(cfg['eval_code'])
    for name,digest in json.loads((code/'release_manifest.json').read_text())['sha256'].items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest,name
    if tasks_path.exists():tasks=json.loads(tasks_path.read_text())
    else:
        tasks=[dict(id=m+'/'+j['dataset']+'/'+j['case_id'],model=m,**j,status='pending',attempts=0)
            for j in sorted(quick['jobs'],key=lambda x:(x['references'],x['dataset'],x['case_id'])) for m in cfg['models']]
        atomic(tasks_path,tasks)
    n_cases=len(quick['jobs'])
    handles={};token=Path(cfg['token_file']).read_text().strip()
    def target(task):return root/'results'/task['model']/(task['dataset']+'-'+task['case_id']+'.json')
    def inspect(task):
        p=target(task)
        if not p.exists():return None
        return case_result(json.loads(p.read_text()),task['references'])
    def snapshot():
        atomic(tasks_path,tasks)
        groups={}
        for name in cfg['models']:
            selected=[t for t in tasks if t['model']==name]
            done=[t for t in selected if t['status']=='completed']
            groups[name]=dict(completed=len(done),total=n_cases,final=len(done)==n_cases,
                found=sum(t['result']['recovered'] for t in done),references_completed=sum(t['references'] for t in done),
                macro_coverage=sum(t['result']['coverage'] for t in done)/n_cases if len(done)==n_cases else None,
                macro_coverage_completed=sum(t['result']['coverage'] for t in done)/len(done) if done else None,
                protocol_errors=sum(t.get('result',{}).get('protocol_errors',0) for t in selected))
        atomic(root/'summary.json',groups)
        atomic(root/'state.json',dict(pid=os.getpid(),phase='validating',completed=sum(t['status']=='completed' for t in tasks),
            total=len(tasks),running=[dict(id=t['id'],pid=t.get('pid')) for t in tasks if t['status']=='running'],
            parked=[t['id'] for t in tasks if t['status']=='parked'],
            scheduling='model-major' if cfg.get('serial_models') else 'interleaved',
            capacity=health.get('capacity',{}),worker_limit=health.get('validation_workers')))
    for task in tasks:
        result=inspect(task)
        if result and result['completed']:
            task.update(status='completed',result=result)
        elif task['status'] in ('running','parked') and not process_matches(task):task['status']='pending'
    while True:
        health=call(cfg['gateway']+'/health')
        legacy=Path(cfg['legacy_state']);remaining=0
        if legacy.exists():
            legacy_state=json.loads(legacy.read_text())
            remaining=max(0,legacy_state.get('total',19)-legacy_state.get('completed',0))
        if legacy.exists() and json.loads(legacy.read_text()).get('status')=='complete':
            for entry in health['backends']:
                if entry['role']=='baseline' and not entry.get('owner'):
                    change_role(cfg['gateway'],token,entry['id'],'shared')
            health=call(cfg['gateway']+'/health')
        for task in tasks:
            if task['status'] not in ('running','parked'):continue
            handle=handles.get(task['id'])
            ended=handle.poll() is not None if handle else not process_matches(task)
            if not ended:continue
            result=inspect(task)
            if result and result['completed']:task.update(status='completed',result=result)
            elif task['attempts']<2 and (not target(task).exists() or not json.loads(target(task).read_text()).get('completed')):
                task.update(status='pending',result=result)
            else:task.update(status='needs_inspection',result=result)
        model_order=cfg.get('model_order',cfg['models'])
        current_model=next((m for m in model_order if any(t['model']==m and t['status'] in ('pending','running','parked') for t in tasks)),None)
        # A one-time cooperative handover preserves live Python state, including
        # a reflection already generated before the search phase. Never stop a
        # worker until it has explicitly reached its existing API pause point.
        handover_path=root/'handover.json'
        if handover_path.exists():
            handover=json.loads(handover_path.read_text())
            for task in tasks:
                if task['status']!='running':continue
                log=target(task).with_suffix('.log')
                offset=handover['offsets'].get(task['id'])
                if offset is None:raise RuntimeError('Untracked live worker during handover')
                with log.open('rb') as source:
                    source.seek(offset);paused=b'Paused. Remove ' in source.read()
                if paused and process_matches(task):
                    os.kill(task['pid'],signal.SIGSTOP);task['status']='parked'
            if not any(t['status']=='running' for t in tasks):
                (root/'PAUSE').unlink(missing_ok=True)
                handover_path.rename(root/'handover.completed.json')
            else:
                snapshot();time.sleep(2);continue
        if cfg.get('serial_models',False):
            for task in tasks:
                if task['status']=='parked' and task['model']==current_model:
                    os.kill(task['pid'],signal.SIGCONT);task['status']='running'
        active=sum(t['status']=='running' for t in tasks)
        dedicated=sum(e['role']=='baseline' and e['observed'] is not None for e in health['backends'])
        reserve=max(0,min(remaining,cfg.get('baseline_reserve_max',3))-dedicated)
        limit=max(0,min(health['capacity']['validation']-reserve,health.get('validation_workers',9)))
        limit=min(limit,cfg.get('max_workers',9))
        for task in tasks:
            if active>=limit:break
            if task['status']!='pending':continue
            if cfg.get('serial_models',False) and task['model']!=current_model:continue
            p=target(task);p.parent.mkdir(parents=True,exist_ok=True)
            model='Qwen3-8B' if task['model']=='base' else task['model']
            command=[cfg['python'],'-u','-m','experiments.test_hard_balanced','--model',model,'--enable-thinking',
                '--base-url',cfg['gateway']+'/validation/v1','--max-tokens','32768',
                '--dataset-root',str(code/'datasets/medium_sampled'),'--dataset',task['dataset'],'--case-id',task['case_id'],
                '--method','mcts-tree','--tree-policy','llm','--duplicate-reward','-1','--exploration-list','off',
                '--two-step-reflection','--unbounded-tree','--tree-tools','--mcts-version','2.2','--stage22-revision',
                '--pause-file',str(root/'PAUSE'),'--resume' if p.exists() else '--output',str(p)]
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            with p.with_suffix('.log').open('a') as log:
                handle=subprocess.Popen(command,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
            task.update(status='running',pid=handle.pid,attempts=task['attempts']+1,
                cmd_hash=hashlib.sha256((Path('/proc')/str(handle.pid)/'cmdline').read_bytes()).hexdigest())
            handles[task['id']]=handle;active+=1
        snapshot()
        if all(t['status'] in ('completed','needs_inspection') for t in tasks):
            state=json.loads((root/'state.json').read_text());state['phase']='completed' if all(t['status']=='completed' for t in tasks) else 'needs_inspection'
            atomic(root/'state.json',state);break
        time.sleep(5)


if __name__=='__main__':main()
