"""Publish a checked adapter without stopping live APIs; signal readiness behind old jobs."""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.error import HTTPError

from server.poolctl import call, change_role, drain
from server.run_finetune_quick6 import atomic


def earlier_jobs_waiting(paths):
    return any(t['status']=='pending' for path in paths for t in json.loads(Path(path).read_text()))


def main():
    import fcntl
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text());root=args.config.resolve().parent
    lock=(root/'publish.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    token=Path(cfg['token_file']).read_text().strip();receipt=root/'status.json'
    def status(phase,**extra):atomic(receipt,dict(model=cfg['name'],phase=phase,utc=time.time(),**extra))
    status('publishing')
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/status' or not hmac.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                self.send_error(403);return
            data=receipt.read_bytes();self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
        def log_message(self,*args):pass
    http=ThreadingHTTPServer((cfg['host'],cfg['port']),Handler)
    threading.Thread(target=http.serve_forever,daemon=True).start()
    loaded=[]
    try:
        asset=Path(cfg['asset']);assert hashlib.sha256((asset/'adapter_model.safetensors').read_bytes()).hexdigest()==cfg['sha256']
        for identity in cfg['gpus']:
            status('publishing',loaded=loaded,waiting_for=identity)
            entry=drain(cfg['gateway'],token,[identity],cfg['name'])[0]
            try:
                actual=call(entry['url']+'/v1/models',token=token)
                known=next((m for m in actual['data'] if m['id']==cfg['name']),None)
                if known:assert known['root']==cfg['asset']
                else:
                    call(entry['node_api']+'/gpus/'+str(entry['gpu'])+'/load',
                         dict(name=cfg['name'],path=cfg['asset'],sha256=cfg['sha256']),token)
                # The card remains drained while the generation check runs.
                result=call(entry['url']+'/v1/chat/completions',dict(model=cfg['name'],
                    messages=[dict(role='user',content='Reply OK.')],max_tokens=8,
                    chat_template_kwargs=dict(enable_thinking=False)),token,timeout=180)
                assert result.get('choices')
                call(cfg['gateway']+'/admin/backends/'+identity,
                     dict(models=sorted(set(entry['models']+[cfg['name']])),role='shared',owner=None),token)
                loaded.append(identity)
            finally:
                change_role(cfg['gateway'],token,identity,'shared',None)
        # Queue order is dispatch order, not a completion barrier. All existing
        # pending baseline jobs receive slots before this model is admitted.
        while earlier_jobs_waiting(cfg.get('after_task_files',[])):
            status('queued',loaded=loaded,reason='waiting for earlier pending baseline jobs to be dispatched')
            time.sleep(5)
        status('ready',loaded=loaded,sha256=cfg['sha256'],epoch=1,global_step=59)
    except Exception as error:
        detail=error.read().decode(errors='replace') if isinstance(error,HTTPError) else str(error)
        status('failed',loaded=loaded,error=str(error),detail=detail)
        print(repr(error),detail,flush=True)
    while True:time.sleep(30)


if __name__=='__main__':main()
