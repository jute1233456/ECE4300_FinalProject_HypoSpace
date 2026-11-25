"""Checkpoint-preserving switch to the full remaining 32K context allowance."""
from datetime import datetime,timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from urllib.request import urlopen
import fcntl

ROOT=Path('/opt/data/private/ECE4300 Final Project')


def main():
    logs=ROOT/'logs/mcts-2.1-32k-switch';logs.mkdir(exist_ok=True,parents=True)
    lock=(logs/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(requested_max_tokens=32768,context_window=32768)
    def update(status,**values):
        state.update(status=status,**values)
        p=logs/'state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(logs/'state.json')
        print(status,values,flush=True)
    try:
        update('waiting_for_saved_boundary')
        while True:
            h=json.load(urlopen('http://127.0.0.1:8002/health',timeout=15))
            if not any(h['inflight_requests']):break
            time.sleep(5)
        backends=h['backends']
        for p in Path('/proc').iterdir():
            try:
                a=(p/'cmdline').read_bytes().split(b'\0')
                if b'experiments.test_easy' not in a or b'--pause-file' not in a:continue
                pause=Path(a[a.index(b'--pause-file')+1].decode())
                if ROOT in pause.resolve().parents and pause.exists():
                    os.kill(int(p.name),signal.SIGTERM);print('STOPPED_SAVED_WORKER',p.name,flush=True)
            except (OSError,ValueError):pass
        source=ROOT/'releases/mcts-2.1-medium-pipeline'
        target=ROOT/'releases/mcts-2.1-medium-pipeline-32k'
        assert not target.exists(),'32K release exists; refusing overwrite'
        shutil.copytree(source,target,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','release_manifest.json'))
        p=target/'server/run_medium21_pipeline.py';s=p.read_text()
        assert "ECE_MAX_TOKENS='16384'" in s
        s=s.replace("ECE_MAX_TOKENS='16384'","ECE_MAX_TOKENS='32768'")
        s=s.replace('logs/mcts-2.1-medium-pipeline','logs/mcts-2.1-medium-pipeline-32k')
        s=s.replace("root/f'results/server-mcts-2.1-medium-{role}'","root/f'results/server-mcts-2.1-medium-{role}-32k'")
        s=s.replace("root/f'logs/mcts-2.1-medium-{role}'","root/f'logs/mcts-2.1-medium-{role}-32k'")
        s=s.replace("root/'results/server-mcts-2.1-medium-training'","root/'results/server-mcts-2.1-medium-training-32k'")
        p.write_text(s)
        p=target/'experiments/direct_evaluation.py';s=p.read_text()
        assert 'timeout=600, generation_options=local_options' in s
        s=s.replace('timeout=600, generation_options=local_options','timeout=3600, generation_options=local_options')
        p.write_text(s)
        p=target/'server/api_gateway.py';s=p.read_text()
        assert 'httpx.AsyncClient(timeout=600,' in s
        s=s.replace('httpx.AsyncClient(timeout=600,',"httpx.AsyncClient(timeout=float(os.environ.get('ECE_GATEWAY_TIMEOUT', '3600')),")
        s=s.replace("return {'status': 'ok', 'backends': BACKENDS,", "return {'status': 'ok', 'request_timeout_seconds': float(os.environ.get('ECE_GATEWAY_TIMEOUT','3600')), 'backends': BACKENDS,")
        p.write_text(s)
        old=ROOT/'results/server-mcts-2.1-medium-evaluation'
        new=ROOT/'results/server-mcts-2.1-medium-evaluation-32k'
        assert not new.exists(),'32K results exist; refusing overwrite'
        new.mkdir();migrated=[]
        for method in ('mcts','two-step'):
            if not (old/method).exists():continue
            shutil.copytree(old/method,new/method)
            for p in (new/method).glob('*.json'):
                if p.name=='status.json' or '.tree.' in p.name or '.failed-' in p.name:continue
                d=json.loads(p.read_text());old_cap=d.get('max_tokens')
                if not isinstance(d,dict):continue
                count=sum(len(c.get('records',[])) for c in d.get('cases',[]))
                count+=len(d.get('interrupted_case',{}).get('records',[]))
                d['resumed_from_16k_file']=str(old/method/p.name)
                d.setdefault('generation_limit_transitions',[]).append(dict(from_max_tokens=old_cap,
                    to_max_tokens=32768,records_saved_before_change=count,
                    utc=datetime.now(timezone.utc).isoformat(),completed_before_change=bool(d.get('completed'))))
                if not d.get('completed'):
                    d['max_tokens']=32768
                    d.setdefault('thinking_config',{})['max_output_tokens']=32768
                p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
                migrated.append(dict(method=method,file=p.name,records=count))
        change=dict(from_max_tokens=16384,to_max_tokens=32768,context_window=32768,
            context_policy='use all remaining context, reserving 8 tokens; never truncate input',
            old_results_preserved=str(old),migrated=migrated,backends=backends,http_timeout_seconds=3600)
        (new/'generation_configuration_change.json').write_text(json.dumps(change,indent=2))
        pin={}
        for name in ('agents','llm','experiments','server','tests','config','datasets','HypoSpace'):
            for p in (target/name).rglob('*'):
                if p.is_file() and p.suffix in ('.py','.json','.txt','.sh'):
                    pin[str(p.relative_to(target))]=sha256(p.read_bytes()).hexdigest()
        (target/'release_manifest.json').write_text(json.dumps(dict(version='2.1',sha256=pin,
            generation_configuration=change),indent=2))
        update('restarting_gateway_after_drain',migrated=migrated)
        for p in Path('/proc').iterdir():
            try:
                a=(p/'cmdline').read_bytes().split(b'\0')
                if b'server.api_gateway:app' in a and b'--port' in a and a[a.index(b'--port')+1]==b'8002':
                    os.kill(int(p.name),signal.SIGTERM)
            except (OSError,ValueError):pass
        time.sleep(2)
        env=dict(os.environ,ECE_BACKENDS=','.join(backends),ECE_GATEWAY_TIMEOUT='3600')
        with (logs/'gateway.log').open('a') as log:
            gateway=subprocess.Popen([str(ROOT/'.venv-inference/bin/python'),'-m','uvicorn',
                'server.api_gateway:app','--host','127.0.0.1','--port','8002'],cwd=target,
                env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        ready=False
        for _ in range(30):
            try:
                health=json.load(urlopen('http://127.0.0.1:8002/health',timeout=2))
                ready=health.get('request_timeout_seconds')==3600 and health['backends']==backends
                if ready:break
            except Exception:pass
            time.sleep(1)
        assert ready,'Gateway did not start with the expected backends and timeout'
        newlogs=ROOT/'logs/mcts-2.1-medium-pipeline-32k';newlogs.mkdir(parents=True,exist_ok=True)
        with (newlogs/'worker.log').open('a') as log:
            worker=subprocess.Popen([str(ROOT/'.venv-inference/bin/python'),'-u',
                str(target/'server/run_medium21_pipeline.py'),'--root',str(ROOT),'--release',str(target)],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        (newlogs/'worker.pid').write_text(str(worker.pid))
        update('32k_pipeline_started',pipeline_pid=worker.pid,gateway_pid=gateway.pid,
               new_results=str(new),new_release=str(target))
    except Exception as error:
        update('failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':main()
