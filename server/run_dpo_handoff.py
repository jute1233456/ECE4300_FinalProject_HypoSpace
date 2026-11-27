"""Drain only requested cards, train DPO, return cards, publish, admit quick3."""
import argparse
import fcntl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from server.poolctl import call
from server.run_finetune_quick6 import atomic
from server.run_sft_handoff import publish_adapter


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--train-only',action='store_true');args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=Path(cfg['output']);root.mkdir(exist_ok=True)
    def status(phase,**extra):
        atomic(root/'status.json',dict(model=cfg['name'],phase=phase,utc=time.time(),pid=os.getpid(),**extra))
    if args.train_only:
        for smoke in [True,False]:
            command=[cfg['python'],'-u','-m','server.train_dpo','--config',str(args.config)]
            if smoke:command.append('--smoke')
            subprocess.run(command,cwd=cfg['code'],check=True)
        return
    lock=(root/'handoff.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token=Path(cfg['token_file']).read_text().strip()
    status('queued')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/status' or self.headers.get('Authorization')!='Bearer '+token:
                self.send_error(403);return
            data=(root/'status.json').read_bytes();self.send_response(200)
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    http=ThreadingHTTPServer(('127.0.0.1',cfg['status_port']),Handler)
    threading.Thread(target=http.serve_forever,daemon=True).start()
    request=Path(cfg['training_request'])
    def request_cards(cards):atomic(request,dict(active=True,gpus=cards,owner=cfg['owner']))
    def wait_leases(cards):
        while True:
            entries=[e for e in call(cfg['gateway']+'/health')['backends'] if e['id'] in cards]
            if all(not e.get('owner') or e['owner']==cfg['owner'] for e in entries):return
            status('waiting_for_collection_to_yield',gpus=cards)
            time.sleep(5)
    try:
        request_cards(cfg['gpus']);wait_leases(cfg['gpus'])
        job=root/'reserved-job.json'
        atomic(job,dict(cwd=cfg['code'],command=[cfg['python'],'-u','-m','server.run_dpo_handoff',
            '--config',str(args.config),'--train-only']))
        command=[cfg['python'],'-u','-m','server.run_reserved_training','--config',cfg['client_config'],
                 '--gpus',*cfg['gpus'],'--job',str(job),'--owner',cfg['owner']]
        process=subprocess.Popen(command,cwd=cfg['code'],stdin=subprocess.DEVNULL)
        while process.poll() is None:
            status('training',gpus=cfg['gpus'],training_pid=process.pid)
            time.sleep(5)
        if process.returncode:raise RuntimeError('Training wrapper failed: '+str(process.returncode))
        trained=json.loads((root/'adapter/status.json').read_text())
        assert trained['phase']=='completed'
        request_cards(cfg['publish_gpus']);wait_leases(cfg['publish_gpus'])
        publication=dict(cfg,output=str(root/'adapter'),gpus=cfg['publish_gpus'])
        for attempt in range(1,13):
            try:
                status('publishing',attempt=attempt)
                digest=publish_adapter(publication)
                break
            except Exception as error:
                if attempt==12:raise
                status('publication_retry',attempt=attempt,error=str(error));time.sleep(15)
        # Ready is consumed by the existing queue's future_models admission.
        status('ready',sha256=digest,epochs=cfg['epochs'],pairs=58,
               global_step=trained['global_step'],gpus=cfg['publish_gpus'])
        # Keep collection off the newly published cards until the live queue
        # has durably admitted all three tests; avoid a ready/admission race.
        while True:
            tasks=json.loads(Path(cfg['quick_tasks']).read_text())
            if sum(t['model']==cfg['name'] for t in tasks)==3:break
            time.sleep(2)
        atomic(request,dict(active=False,gpus=[],owner=cfg['owner']))
    except Exception as error:
        status('failed',error=repr(error))
        atomic(request,dict(active=False,gpus=[],owner=cfg['owner']))
        print(repr(error),flush=True)
    while True:time.sleep(30)


if __name__=='__main__':main()
