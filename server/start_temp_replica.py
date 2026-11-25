"""Launch one authenticated replica entirely inside the temporary owned directory."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess

ROOT=Path('/opt/data/private/ECE4300-temp-api-20261125')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--host',required=True)
    parser.add_argument('--gpu',required=True);parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args();assert (ROOT/'OWNERSHIP.json').exists() and (ROOT/'provision.complete').exists()
    auth=ROOT/'api_credentials.json'
    if not auth.exists():
        auth.write_text(json.dumps({'api_key':secrets.token_urlsafe(40)}));auth.chmod(0o600)
    token=json.loads(auth.read_text())['api_key'];node=socket.gethostname();tag=f'{node}-gpu{args.gpu}'
    pidfile=ROOT/f'logs/{tag}.pid'
    if pidfile.exists() and (Path('/proc')/pidfile.read_text().strip()).exists():
        raise RuntimeError('Replica already running')
    with socket.socket() as sock:sock.bind((args.host,args.port))
    cache=ROOT/'cache'/tag;temp=ROOT/'tmp'/tag;cache.mkdir(parents=True,exist_ok=True);temp.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu,HF_HOME=str(ROOT/'cache/huggingface'),
        VLLM_CACHE_ROOT=str(cache/'vllm'),XDG_CACHE_HOME=str(cache/'xdg'),TRITON_CACHE_DIR=str(cache/'triton'),
        TMPDIR=str(temp),VLLM_API_KEY=token,VLLM_USAGE_STATS='0',OMP_NUM_THREADS='4')
    command=[str(ROOT/'.venv-inference/bin/python'),'-m','vllm.entrypoints.openai.api_server',
        '--model',str(ROOT/'models/Qwen3-8B'),'--served-model-name','Qwen3-8B','--host',args.host,
        '--port',str(args.port),'--dtype','bfloat16','--max-model-len','32768',
        '--gpu-memory-utilization','0.92','--max-num-seqs','8','--max-num-batched-tokens','4096',
        '--enforce-eager','--generation-config','vllm','--disable-log-requests']
    with (ROOT/f'logs/{tag}.log').open('a') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,start_new_session=True)
    pidfile.write_text(str(p.pid))
    print(json.dumps(dict(node=node,gpu=args.gpu,port=args.port,pid=p.pid,endpoint=f'http://{args.host}:{args.port}')))


if __name__=='__main__':main()
