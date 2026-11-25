"""Short-lived TLS transfer with bearer auth, source-IP allowlist and fixed files only."""
from functools import partial
from hashlib import sha256
import hmac
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import ssl
import subprocess
import tarfile
import threading

ROOT=Path('/opt/data/private/ECE4300 Final Project')
ASSETS=ROOT/'transfer_exports/temp-pool-secure-20261125'
IP='10.106.195.10'


def main():
    ASSETS.mkdir(parents=True,exist_ok=False)
    token=secrets.token_urlsafe(40)
    auth=ASSETS/'transfer_credentials.json';auth.write_text(json.dumps({'token':token,'server':f'https://{IP}:18765'}));auth.chmod(0o600)
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2',
        '-keyout',str(ASSETS/'tls.key'),'-out',str(ASSETS/'tls.crt'),'-subj','/CN=ECE4300 temporary transfer',
        '-addext',f'subjectAltName=IP:{IP}'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    (ASSETS/'tls.key').chmod(0o600)
    records=json.loads((ROOT/'logs/model_integrity.json').read_text());assert all(r['matches'] for r in records)
    allowed={'/model/'+r['file']:ROOT/'models/Qwen3-8B'/r['file'] for r in records}
    allowed['/runtime.tar.gz']=ASSETS/'runtime.tar.gz'
    allowed['/runtime_ready.json']=ASSETS/'runtime_ready.json'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.client_address[0] not in {'10.106.254.59','10.106.31.24'} or not hmac.compare_digest(
                self.headers.get('Authorization',''),'Bearer '+token):
                self.send_error(403);return
            if self.path=='/manifest.json':
                body=json.dumps(records).encode();self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            p=allowed.get(self.path)
            if p is None or not p.is_file():self.send_error(404);return
            self.send_response(200);self.send_header('Content-Length',str(p.stat().st_size));self.end_headers()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):self.wfile.write(block)
        def log_message(self,fmt,*args):print(self.client_address[0],fmt%args,flush=True)

    def runtime():
        p=ASSETS/'runtime.tar.gz'
        with tarfile.open(p,'w:gz',compresslevel=1) as tar:
            tar.add(ROOT/'.venv-inference',arcname='.venv-inference',
                    filter=lambda info:None if '__pycache__' in Path(info.name).parts else info)
        h=sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
        (ASSETS/'runtime_ready.json').write_text(json.dumps({'sha256':h.hexdigest(),'bytes':p.stat().st_size}))
        print('RUNTIME_ARCHIVE_READY',p.stat().st_size,flush=True)
    server=ThreadingHTTPServer((IP,18765),Handler)
    ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER);ctx.load_cert_chain(ASSETS/'tls.crt',ASSETS/'tls.key')
    server.socket=ctx.wrap_socket(server.socket,server_side=True)
    threading.Thread(target=runtime,daemon=True).start()
    server.timeout=3600
    print('SECURE_TRANSFER_READY',IP,flush=True)
    server.serve_forever()


if __name__=='__main__':main()
