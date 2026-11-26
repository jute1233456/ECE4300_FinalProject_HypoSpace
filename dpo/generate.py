"""Collect unscored stage responses using only spare, explicitly reserved GPU slots."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time
from urllib.error import HTTPError
import uuid

from dpo.common import atomic,digest,parse_response,read_jsonl
from server.poolctl import call,change_role
from server.pool_metrics import engine_jobs


def priority_load(entries):
    running=0;waiting=0
    for entry in entries:
        path=Path(entry['state']);state=json.loads(path.read_text())
        terminal=state.get('phase') in ('completed','needs_inspection')
        if not terminal and time.time()-path.stat().st_mtime>120:raise RuntimeError('Priority queue heartbeat is stale')
        running+=len(state.get('running',[]))
        tasks=json.loads(Path(entry['tasks']).read_text());enabled=None
        if entry.get('config'):enabled=json.loads(Path(entry['config']).read_text()).get('enabled_models')
        waiting+=sum(t['status']=='pending' and (enabled is None or t.get('model') in enabled) for t in tasks)
        waiting+=sum(v not in ('failed','ready') for v in state.get('future_models',{}).values())
    return running,waiting


def spare_cards(health,model,owner,running,active,max_workers):
    ours=sum(e.get('owner')==owner and e['role']=='reserved' for e in health['backends'])
    slots=max(0,min(max_workers-active,health['capacity']['validation']+ours-running-active))
    idle=[e for e in health['backends'] if e['role'] in ('shared','validation') and not e.get('owner')
          and model in e['models'] and e['inflight']==0 and e.get('observed')==0]
    return sorted(idle,key=lambda e:e['id'])[:slots]


def recover_orphan_leases(cfg,health,active_ids,token):
    """After a controller restart, reclaim only this run's proven-idle leases."""
    import urllib.request
    for entry in health['backends']:
        if entry.get('owner')!=cfg['owner'] or entry['id'] in active_ids:continue
        try:
            req=urllib.request.Request(entry['url']+'/metrics',headers={'Authorization':'Bearer '+token})
            with urllib.request.urlopen(req,timeout=5) as f:idle=engine_jobs(f.read().decode())==0
            if idle:
                lease=Path(cfg['output'])/'leases'/(entry['id']+'.json')
                role=json.loads(lease.read_text())['role'] if lease.exists() else 'validation'
                change_role(cfg['gateway'],token,entry['id'],role,None)
        except Exception:continue


def release_when_idle(cfg,entry,token):
    import urllib.request
    while True:
        try:
            req=urllib.request.Request(entry['url']+'/metrics',headers={'Authorization':'Bearer '+token})
            with urllib.request.urlopen(req,timeout=10) as reply:idle=engine_jobs(reply.read().decode())==0
            if idle:break
        except Exception:pass
        time.sleep(2)
    current=next(e for e in call(cfg['gateway']+'/health')['backends'] if e['id']==entry['id'])
    if current.get('owner')==cfg['owner']:change_role(cfg['gateway'],token,entry['id'],entry['role'],None)


def sample_once(cfg,state,task,entry,token,limit,input_tokens,attempt,release_card=True):
    """Called only after gateway admission was disabled atomically for this idle card."""
    root=Path(cfg['output']);record=dict(sample_id=task['sample_id'],state_id=state['state_id'],
        stage=state['stage'],case_id=state['case_id'],family_id=state['family_id'],sample_index=task['sample_index'],
        attempt=attempt,requested_model=cfg['model'],checkpoint_sha256=cfg['checkpoint_sha256'],gpu=entry['id'],
        started_utc=time.time(),input_sha256=state['input_sha256'],input_tokens=input_tokens,max_tokens=limit,
        review_status='unreviewed')
    seed=(cfg.get('seed',33550336)+int(task['sample_id'][:8],16))%(2**31)
    request=dict(model=cfg['model'],messages=state['messages'],max_tokens=limit,temperature=.7,top_p=.8,top_k=20,
                 seed=seed,chat_template_kwargs=dict(enable_thinking=True))
    record['generation_parameters']={k:v for k,v in request.items() if k!='messages'}
    try:
        data=call(entry['url']+'/v1/chat/completions',request,token,timeout=3600)
        # Save the entire response even if its structure or model identity is wrong.
        record['raw_response']=data
        record['response']=parse_response(data)
        if record['response']['returned_model']!=cfg['model']:raise ValueError('Returned model differs from requested model')
        record['response_status']='received'
    except Exception as error:
        record.update(response_status='error',error_type=type(error).__name__,error=str(error))
        if isinstance(error,HTTPError):record['error_body']=error.read().decode(errors='replace')
    finally:
        record['finished_utc']=time.time()
        path=root/'responses'/(task['sample_id']+'.'+str(attempt)+'.'+uuid.uuid4().hex+'.json')
        try:atomic(path,record)
        # A timed-out request may still occupy the engine. Keep it reserved until
        # actual metrics prove idle; never replay it or stop the engine blindly.
        finally:
            if release_card:release_when_idle(cfg,entry,token)
    return str(path),record


def collect_state(cfg,state,task,entry,token,limit,input_tokens,stopped):
    """One state is one leased-GPU task; persist each candidate immediately."""
    try:
        for sample in task['samples']:
            if sample['status']!='pending':continue
            if stopped.is_set() or (Path(cfg['output'])/'PAUSE').exists():break
            sample['attempts']+=1;sample['status']='running'
            path,row=sample_once(cfg,state,sample,entry,token,limit,input_tokens,sample['attempts'],release_card=False)
            sample.update(status='received' if row['response_status']=='received' else ('pending' if sample['attempts']<2 else 'error'),response_file=path)
            # After an uncertain API failure, first wait for idle and return the
            # lease; do not submit another request on that potentially busy card.
            if row['response_status']!='received':break
    finally:release_when_idle(cfg,entry,token)


def task_status(task):
    statuses=[s['status'] for s in task['samples']]
    if all(s=='received' for s in statuses):return 'received'
    if any(s in ('pending','running') for s in statuses):return 'pending'
    return 'error'


def main():
    import fcntl
    from transformers import AutoTokenizer
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    if (root/'stop_requested.json').exists():parser.error('User requested stop; explicit authorization is required before removing this marker and starting.')
    lock=(root/'generation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((root/'manifest.json').read_text());states=read_jsonl(root/'states.jsonl')
    assert manifest['version']=='stage-response-collection-v2'
    assert all('split' not in s for s in states)
    assert hashlib.sha256((root/'states.jsonl').read_bytes()).hexdigest()==manifest['states_sha256']
    assert cfg['model']==manifest['model'] and cfg['checkpoint_sha256']==manifest['checkpoint_sha256']
    recipe=dict(model=cfg['model'],checkpoint_sha256=cfg['checkpoint_sha256'],states_sha256=manifest['states_sha256'],
                max_tokens=32768,context_window=32768,temperature=.7,top_p=.8,top_k=20,seed=cfg.get('seed',33550336),thinking=True)
    saved=root/'generation_recipe.json'
    if saved.exists():assert json.loads(saved.read_text())==recipe,'Changed recipe requires a new output directory'
    else:atomic(saved,recipe)
    token=Path(cfg['token_file']).read_text().strip();tokenizer=AutoTokenizer.from_pretrained(cfg['tokenizer'])
    byid={s['state_id']:s for s in states};tasks=[];seen={};attempts={}
    (root/'responses').mkdir(exist_ok=True)
    for file in (root/'responses').glob('*.json'):
        row=json.loads(file.read_text());sample=row['sample_id'];attempts[sample]=max(attempts.get(sample,0),row['attempt'])
        if row['response_status']=='received':seen[sample]=str(file)
    for state in sorted(states,key=lambda s:digest(s['state_id'])):
        task=dict(state_id=state['state_id'],samples=[])
        for index in range(manifest['samples_per_state']):
            sample=digest([state['state_id'],index,digest(recipe)])
            task['samples'].append(dict(sample_id=sample,state_id=state['state_id'],sample_index=index,
                status='received' if sample in seen else ('error' if attempts.get(sample,0)>=2 else 'pending'),
                attempts=attempts.get(sample,0),response_file=seen.get(sample)))
        task['status']=task_status(task);tasks.append(task)
    stopped=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:stopped.set());signal.signal(signal.SIGINT,lambda *_:stopped.set())
    active={};counts={};reason='starting';last_export=0
    with ThreadPoolExecutor(max_workers=cfg.get('max_workers',9)) as pool:
        while True:
            for future,(task,entry) in list(active.items()):
                if not future.done():continue
                future.result();task['status']=task_status(task)
                del active[future]
            pending=[t for t in tasks if t['status']=='pending']
            remaining_leases=None
            try:
                health=call(cfg['gateway']+'/health')
                recover_orphan_leases(cfg,health,{entry['id'] for _,entry in active.values()},token)
                health=call(cfg['gateway']+'/health')
                remaining_leases=sum(e.get('owner')==cfg['owner'] for e in health['backends'])
            except Exception as error:reason='waiting: '+str(error)
            if not stopped.is_set() and not (root/'PAUSE').exists() and pending:
                try:
                    running,waiting=priority_load(cfg['priority_queues'])
                    health=call(cfg['gateway']+'/health');cards=[] if waiting else spare_cards(health,cfg['model'],cfg['owner'],running,len(active),cfg.get('max_workers',9))
                    reason='earlier_jobs_pending' if waiting else ('generating' if cards or active else 'waiting_for_idle_model_gpu')
                    for entry,task in zip(cards,pending):
                        state=byid[task['state_id']]
                        if task['state_id'] not in counts:counts[task['state_id']]=len(tokenizer.apply_chat_template(state['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=True))
                        limit=min(32768,32768-counts[task['state_id']]-8)
                        if limit<1:
                            for sample in task['samples']:
                                if sample['status']=='pending':sample.update(status='error',error='Input exceeds context; not truncated')
                            task['status']='error';continue
                        atomic(root/'leases'/(entry['id']+'.json'),dict(role=entry['role'],id=entry['id']))
                        try:change_role(cfg['gateway'],token,entry['id'],'reserved',cfg['owner'])
                        except HTTPError as error:
                            if error.code==409:continue
                            raise
                        task['status']='running'
                        future=pool.submit(collect_state,cfg,state,task,entry,token,limit,counts[task['state_id']],stopped)
                        active[future]=(task,entry)
                except Exception as error:reason='waiting: '+str(error)
            samples=[s for task in tasks for s in task['samples']]
            done=sum(s['status']=='received' for s in samples);errors=sum(s['status']=='error' for s in samples)
            terminal=all(t['status'] in ('received','error') for t in tasks)
            phase=('completed' if not errors else 'completed_with_errors') if terminal else (('stopped' if not active else 'stopping') if stopped.is_set() else reason)
            atomic(root/'tasks.json',tasks);atomic(root/'state.json',dict(pid=os.getpid(),phase=phase,total=len(samples),received=done,errors=errors,
                total_states=len(tasks),completed_states=sum(t['status']=='received' for t in tasks),running=len(active),scheduling_unit='state',utc=time.time(),scoring='not_started'))
            if (terminal or (stopped.is_set() and not active)) and remaining_leases==0:
                from dpo.export import export_records
                export_records(root,archive=True);break
            if done>=last_export+20:
                from dpo.export import export_records
                export_records(root);last_export=done
            time.sleep(5)


if __name__=='__main__':main()
