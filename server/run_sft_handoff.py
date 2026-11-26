"""Reserved QLoRA job followed by checked LoRA publication and a private receipt."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from urllib.error import HTTPError

from server.poolctl import call, change_role, drain
from server.run_finetune_quick6 import atomic


def validate_adapter(path, steps):
    import torch
    from safetensors.torch import load_file
    path=Path(path)
    state=json.loads((path/f'checkpoint-{steps}/trainer_state.json').read_text())
    assert state['global_step']==steps and state['epoch']==2.0,state
    weights=load_file(str(path/'adapter_model.safetensors'))
    assert weights and all(torch.isfinite(v).all().item() for v in weights.values())
    bs=[v for k,v in weights.items() if 'lora_B' in k]
    assert bs and all(torch.count_nonzero(v).item() for v in bs)
    return dict(global_step=steps,epoch=state['epoch'],tensors=len(weights))


def publish_adapter(cfg):
    """Resume partial publication without retraining or replacing immutable weights."""
    token=Path(cfg['token_file']).read_text().strip()
    asset=Path(cfg['asset_root'])/cfg['name'];asset.mkdir(exist_ok=True)
    for filename in ['adapter_config.json','adapter_model.safetensors']:
        source=Path(cfg['output'])/filename;dest=asset/filename
        if dest.exists():
            if hashlib.sha256(source.read_bytes()).digest()!=hashlib.sha256(dest.read_bytes()).digest():
                raise ValueError('Published asset differs from trained weights: '+filename)
        else:
            tmp=dest.with_suffix(dest.suffix+'.tmp');shutil.copy2(source,tmp);tmp.replace(dest)
    digest=hashlib.sha256((asset/'adapter_model.safetensors').read_bytes()).hexdigest()
    for identity in cfg['gpus']:
        entry=drain(cfg['gateway'],token,[identity],cfg['owner'])[0]
        try:
            actual=call(entry['url']+'/v1/models',token=token)
            loaded=next((m for m in actual['data'] if m['id']==cfg['name']),None)
            if loaded:
                if loaded['root']!=str(asset):raise ValueError('Loaded adapter path differs')
            else:
                call(entry['node_api']+'/gpus/'+str(entry['gpu'])+'/load',
                    dict(name=cfg['name'],path=str(asset),sha256=digest),token)
            call(cfg['gateway']+'/admin/backends/'+identity,
                dict(models=sorted(set(entry['models']+[cfg['name']])),role='shared',owner=None),token)
        except Exception:
            change_role(cfg['gateway'],token,identity,'shared',None)
            raise
    result=call(cfg['gateway']+'/validation/v1/chat/completions',dict(model=cfg['name'],
        messages=[dict(role='user',content='Reply OK.')],max_tokens=8,
        chat_template_kwargs=dict(enable_thinking=False)),token,timeout=180)
    assert result.get('choices'),result
    return digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--train-only',action='store_true');args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=args.config.resolve().parent
    receipt=root/'status.json'
    def status(phase,**extra):atomic(receipt,dict(model=cfg['name'],phase=phase,utc=time.time(),**extra))
    if args.train_only:
        assert hashlib.sha256(Path(cfg['train_file']).read_bytes()).hexdigest()==cfg['data_sha256']
        status('smoke')
        subprocess.run(cfg['smoke_command'],cwd=cfg['code'],check=True)
        status('training',samples=944,epochs=2,total_updates=118)
        subprocess.run(cfg['train_command'],cwd=cfg['code'],check=True)
        validation=validate_adapter(cfg['output'],118)
        atomic(root/'adapter-validation.json',validation)
        return

    import fcntl
    lock=(root/'handoff.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token=Path(cfg['token_file']).read_text().strip()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/status' or self.headers.get('Authorization')!='Bearer '+token:
                self.send_error(403);return
            data=receipt.read_bytes();self.send_response(200)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    status('queued')
    http=ThreadingHTTPServer((cfg['status_ip'],cfg['status_port']),Handler)
    threading.Thread(target=http.serve_forever,daemon=True).start()
    try:
        atomic(root/'reserved-job.json',dict(cwd=cfg['code'],command=[cfg['python'],'-u','-m',
            'server.run_sft_handoff','--config',str(args.config.resolve()),'--train-only']))
        status('draining')
        subprocess.run([cfg['python'],'-u','-m','server.run_reserved_training','--config',cfg['client_config'],
            '--gpus',*cfg['gpus'],'--job',str(root/'reserved-job.json'),'--owner',cfg['owner']],cwd=cfg['code'],check=True)
        status('publishing')
        digest=publish_adapter(cfg)
        status('ready',sha256=digest,gpus=cfg['gpus'],samples=944,epochs=2,global_step=118)
    except Exception as error:
        detail=error.read().decode(errors='replace') if isinstance(error,HTTPError) else str(error)
        status('failed',error=str(error),detail=detail);print(repr(error),detail,flush=True)
    # Keep the authenticated receipt available after the training process exits.
    # No GPU is owned by this lightweight status server.
    while True:time.sleep(30)


if __name__=='__main__':main()
