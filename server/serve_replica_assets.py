"""Temporary authenticated TLS server for verified model files and an existing runtime."""
import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import ssl
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--bind',required=True)
    p.add_argument('--client',required=True,action='append')
    p.add_argument('--port',type=int,default=18766)
    p.add_argument('--assets',type=Path,required=True)
    args=p.parse_args()
    args.assets.mkdir(parents=True,exist_ok=False)
    token=secrets.token_urlsafe(40)
    auth=args.assets/'transfer_credentials.json'
    auth.write_text(json.dumps(dict(token=token,server=f'https://{args.bind}:{args.port}')))
    auth.chmod(0o600)
    subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','2',
        '-keyout',str(args.assets/'tls.key'),'-out',str(args.assets/'transfer.crt'),
        '-subj','/CN=ECE4300 replica transfer','-addext',f'subjectAltName=IP:{args.bind}'],
        check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    (args.assets/'tls.key').chmod(0o600)
    records=json.loads((args.root/'logs/model_integrity.json').read_text())
    assert all(r['matches'] and Path(r['file']).name==r['file'] for r in records)
    runtime=args.root/'transfer_exports/temp-pool-secure-20261125'
    allowed={'/model/'+r['file']:args.root/'models/Qwen3-8B'/r['file'] for r in records}
    allowed.update({'/runtime.tar.gz':runtime/'runtime.tar.gz',
                    '/runtime_ready.json':runtime/'runtime_ready.json'})
    assert all(f.is_file() for f in allowed.values())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.client_address[0] not in args.client or not hmac.compare_digest(
                self.headers.get('Authorization',''),'Bearer '+token):
                self.send_error(403);return
            if self.path=='/manifest.json':
                body=json.dumps(records).encode();self.send_response(200)
                self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            file=allowed.get(self.path)
            if file is None:self.send_error(404);return
            self.send_response(200);self.send_header('Content-Length',str(file.stat().st_size));self.end_headers()
            try:
                with file.open('rb') as f:
                    for block in iter(lambda:f.read(1024*1024),b''):self.wfile.write(block)
            except (BrokenPipeError,ConnectionResetError):pass
        def log_message(self,fmt,*args):print(self.client_address[0],fmt%args,flush=True)

    server=ThreadingHTTPServer((args.bind,args.port),Handler)
    ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(args.assets/'transfer.crt',args.assets/'tls.key')
    server.socket=ctx.wrap_socket(server.socket,server_side=True)
    print('AUTHENTICATED_TLS_ASSETS_READY',args.bind,args.port,flush=True)
    server.serve_forever()


if __name__=='__main__':main()
