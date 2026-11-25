"""Validate three new replicas, then adopt existing writers into nine question slots."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import Request,urlopen

ROOT=Path('/opt/data/private/ECE4300 Final Project')
NEW=['http://10.106.31.61:18000','http://10.106.254.57:18000','http://10.106.254.57:18001']


def main():
    import fcntl
    logs=ROOT/'logs/node4-provision';logs.mkdir(parents=True,exist_ok=True)
    lock=(logs/'activation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),status='waiting_for_new_replicas',new=NEW)
    def save(**values):
        state.update(values);p=logs/'activation-state.tmp';p.write_text(json.dumps(state,indent=2));p.replace(logs/'activation-state.json')
    newkey=json.loads((ROOT/'config/node4-api-credentials.json').read_text())['api_key']
    ready={};deadline=time.monotonic()+2400;save()
    while len(ready)<3:
        if time.monotonic()>deadline:raise RuntimeError('New replica startup timed out')
        for url in NEW:
            if url in ready:continue
            try:
                payload=dict(model='Qwen3-8B',messages=[dict(role='user',content='Reply exactly READY')],
                    max_tokens=32,temperature=0,chat_template_kwargs={'enable_thinking':False})
                request=Request(url+'/v1/chat/completions',data=json.dumps(payload).encode(),
                    headers={'Authorization':'Bearer '+newkey,'Content-Type':'application/json'})
                with urlopen(request,timeout=90) as f:response=json.load(f)
                assert 'READY' in response['choices'][0]['message']['content']
                ready[url]=True;save(validated=list(ready));print('VALIDATED',url,flush=True)
            except Exception as error:print('WAIT',url,type(error).__name__,flush=True)
        if len(ready)<3:time.sleep(10)
    oldhealth=json.load(urlopen('http://127.0.0.1:8003/health',timeout=10))
    old=oldhealth['backends'];assert len(old)==6
    backends=NEW+old
    oldkey=json.loads((ROOT/'config/temp-pool-credentials.json').read_text())['api_key']
    keys={url:newkey for url in NEW}
    keys.update({url:oldkey for url in old if ':1800' in url})
    keyfile=ROOT/'config/pool9-backend-credentials.json'
    os.umask(0o077);keyfile.write_text(json.dumps(keys));keyfile.chmod(0o600)
    env=dict(os.environ,ECE_POOL_BACKENDS=json.dumps(backends),ECE_POOL_KEYS_FILE=str(keyfile))
    with (logs/'gateway9.log').open('a') as log:
        gateway=subprocess.Popen([str(ROOT/'.venv-inference/bin/python'),'-m','uvicorn','server.api_pool_gateway:app',
            '--host','127.0.0.1','--port','8004'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    healthy=False
    for _ in range(30):
        try:
            health=json.load(urlopen('http://127.0.0.1:8004/health',timeout=10))
            healthy=len(health['backends'])==9 and all(x is not None for x in health['observed_engine_jobs'])
            if healthy:break
        except Exception:pass
        time.sleep(1)
    assert healthy,'Nine backend gateway failed health/metrics validation'
    save(status='gateway_ready',gateway_pid=gateway.pid,backends=backends)
    queue_state=ROOT/'logs/mixed-mcts-six-question-queue/state.json'
    previous=json.loads(queue_state.read_text());pid=previous['pid'];proc=Path(f'/proc/{pid}/cmdline')
    if proc.exists():
        args=proc.read_bytes().split(b'\0')
        assert any(a.endswith(b'/run_mixed_mcts_queue.py') for a in args),'Unexpected queue controller'
        os.kill(pid,signal.SIGTERM)
        for _ in range(30):
            if not proc.exists() or not proc.read_bytes():break
            time.sleep(0.1)
    # Question children continue undisturbed and are adopted by exact output filename.
    qlogs=ROOT/'logs/mixed-mcts-six-question-queue'
    with (qlogs/'worker.log').open('a') as log:
        worker=subprocess.Popen([str(ROOT/'.venv-tree-tools/bin/python'),'-u','server/run_mixed_mcts_queue.py',
            '--root',str(ROOT),'--workers','9','--api-base','http://127.0.0.1:8004/v1'],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    save(status='active',queue_pid=worker.pid,workers=9,api_base='http://127.0.0.1:8004/v1')
    for path in (ROOT/'logs/mcts-2.1-temp-pool6/state.json',ROOT/'logs/mcts-2.2-stage-medium-pilot-32k/state.json'):
        d=json.loads(path.read_text());d.update(scheduler_pid=worker.pid,scheduler_state=str(queue_state),pool_backends=9);path.write_text(json.dumps(d,indent=2))
    # Close the temporary model transfer service after all new replicas are verified.
    pidfile=logs/'transfer.pid'
    if pidfile.exists():
        transfer_pid=int(pidfile.read_text());proc=Path(f'/proc/{transfer_pid}/cmdline')
        if proc.exists() and b'server/serve_replica_assets.py' in proc.read_bytes().split(b'\0'):
            os.kill(transfer_pid,signal.SIGTERM);print('TRANSFER_CLOSED',transfer_pid,flush=True)
    print('NINE_QUESTION_POOL_ACTIVE',worker.pid,flush=True)


if __name__=='__main__':main()
