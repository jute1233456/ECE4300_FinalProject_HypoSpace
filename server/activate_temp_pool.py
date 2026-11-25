"""Validate temporary replicas, build authenticated six-backend pool, preserve active writers."""
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from urllib.request import Request,urlopen

ROOT=Path('/opt/data/private/ECE4300 Final Project')
NEW=['http://10.106.254.59:18000','http://10.106.254.59:18001','http://10.106.31.24:18000']


def main():
    logs=ROOT/'logs/temp-pool';logs.mkdir(parents=True,exist_ok=True)
    def state(status,**values):
        p=logs/'state.tmp';p.write_text(json.dumps(dict(status=status,**values),indent=2));p.replace(logs/'state.json')
        print(status,values,flush=True)
    key=json.loads((ROOT/'config/temp-pool-credentials.json').read_text())['api_key']
    ready={}
    state('waiting_for_replicas',backends=NEW)
    deadline=time.monotonic()+1800
    while len(ready)<3:
        if time.monotonic()>deadline:raise RuntimeError('Temporary replica startup timed out')
        for url in NEW:
            if url in ready:continue
            try:
                payload=dict(model='Qwen3-8B',messages=[{'role':'user','content':'Reply exactly READY'}],max_tokens=32,
                    temperature=0,chat_template_kwargs={'enable_thinking':False})
                request=Request(url+'/v1/chat/completions',data=json.dumps(payload).encode(),
                    headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                response=json.load(urlopen(request,timeout=120))
                assert 'READY' in response['choices'][0]['message']['content']
                ready[url]=response;state('replica_validated',ready=list(ready))
            except Exception as e:print('WAIT',url,type(e).__name__,flush=True)
        if len(ready)<3:time.sleep(10)
    old=json.load(urlopen('http://127.0.0.1:8002/health'))['backends']
    source=ROOT/'releases/mcts-2.1-reflection-only-32k';target=ROOT/'releases/mcts-2.1-temp-pool6-32k'
    assert not target.exists(),'Pool release already exists'
    shutil.copytree(source,target,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc','release_manifest.json'))
    p=target/'server/api_gateway.py';s=p.read_text();needle='app = FastAPI()';assert s.count(needle)==1
    s=s.replace(needle,"from pathlib import Path\nBACKEND_KEY = json.loads(Path(os.environ['ECE_BACKEND_KEYS_FILE']).read_text())['api_key']\nAUTH_BACKENDS = json.loads(os.environ['ECE_AUTH_BACKENDS'])\n\n"+needle)
    needle="client.post(BACKENDS[index] + '/v1/chat/completions', json=body)";assert s.count(needle)==1
    s=s.replace(needle,"client.post(BACKENDS[index] + '/v1/chat/completions', json=body, headers={'Authorization': 'Bearer '+BACKEND_KEY} if BACKENDS[index] in AUTH_BACKENDS else {})")
    p.write_text(s)
    p=target/'server/run_reflection21_only.py';s=p.read_text()
    s=s.replace('logs/mcts-2.1-reflection-only','logs/mcts-2.1-temp-pool6')
    s=s.replace('workers_per_method=3','workers_per_method=6').replace("ECE_WORKERS_PER_METHOD='3'","ECE_WORKERS_PER_METHOD='6'")
    needle="            update('starting',role=role,output=str(output))";assert s.count(needle)==1
    s=s.replace(needle,"            env['ECE_API_BASE']='http://127.0.0.1:8003/v1'\n"+needle);p.write_text(s)
    pin={}
    for name in ('agents','llm','experiments','server','tests','config','datasets','HypoSpace'):
        for f in (target/name).rglob('*'):
            if f.is_file() and f.suffix in ('.py','.json','.txt','.sh'):pin[str(f.relative_to(target))]=sha256(f.read_bytes()).hexdigest()
    (target/'release_manifest.json').write_text(json.dumps(dict(version='2.1',sha256=pin),indent=2))
    env=dict(os.environ,ECE_BACKENDS=','.join(NEW+old),ECE_BACKEND_KEYS_FILE=str(ROOT/'config/temp-pool-credentials.json'),
        ECE_AUTH_BACKENDS=json.dumps(NEW),ECE_GATEWAY_TIMEOUT='3600')
    with (logs/'gateway.log').open('a') as log:
        gateway=subprocess.Popen([str(ROOT/'.venv-inference/bin/python'),'-m','uvicorn','server.api_gateway:app',
            '--host','127.0.0.1','--port','8003'],cwd=target,env=env,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    healthy=False
    for _ in range(30):
        try:
            h=json.load(urlopen('http://127.0.0.1:8003/health',timeout=2));healthy=h['backends']==NEW+old
            if healthy:break
        except Exception:pass
        time.sleep(1)
    assert healthy,'Six-backend gateway not ready'
    oldstatepath=ROOT/'logs/mcts-2.1-reflection-only/state.json';oldstate=json.loads(oldstatepath.read_text())
    for pid in (oldstate['pid'],oldstate['child_pid']):
        p=Path('/proc')/str(pid)
        if p.exists():
            args=(p/'cmdline').read_bytes().split(b'\0')
            if any(a.endswith(b'/run_reflection21_only.py') or a==b'server/run_evaluations.py' for a in args):os.kill(pid,signal.SIGTERM)
    oldstate['status']='superseded_by_six_backend_pool';oldstatepath.write_text(json.dumps(oldstate,indent=2))
    out=ROOT/'logs/mcts-2.1-temp-pool6';out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.log').open('a') as log:
        worker=subprocess.Popen([str(ROOT/'.venv-inference/bin/python'),'-u',str(target/'server/run_reflection21_only.py'),
            '--root',str(ROOT),'--release',str(target)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    state('active',backends=NEW+old,gateway_pid=gateway.pid,worker_pid=worker.pid,
        workers_per_method=6,api_base='http://127.0.0.1:8003/v1',model='Qwen3-8B',method='two-step MCTS 2.1')
    # The authenticated artifact transfer is no longer required after verification.
    pidfile=ROOT/'logs/secure-temp-transfer.pid'
    if pidfile.exists():
        pid=int(pidfile.read_text());p=Path('/proc')/str(pid)
        if p.exists() and b'server/secure_temp_transfer.py' in (p/'cmdline').read_bytes().split(b'\0'):
            os.kill(pid,signal.SIGTERM);print('TRANSFER_SERVICE_STOPPED',pid,flush=True)


if __name__=='__main__':main()
