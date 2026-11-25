"""Receive exact verified model/runtime over a pinned TLS connection into the owned folder."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from hashlib import sha256
import json
from pathlib import Path
import ssl
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request,urlopen

ROOT=Path('/opt/data/private/ECE4300-temp-api-20261125')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    return provision(args.root)


def provision(ROOT):
    assert (ROOT/'OWNERSHIP.json').exists()
    auth=json.loads((ROOT/'transfer_credentials.json').read_text())
    ctx=ssl.create_default_context(cafile=str(ROOT/'transfer.crt'))
    def get(path):
        return urlopen(Request(auth['server']+path,headers={'Authorization':'Bearer '+auth['token']}),context=ctx,timeout=600)
    with get('/manifest.json') as f:records=json.load(f)
    (ROOT/'model_integrity_source.json').write_text(json.dumps(records,indent=2))
    model=ROOT/'models/Qwen3-8B';model.mkdir(parents=True,exist_ok=True)
    def download(path,target,digest):
        temporary=target.with_name(target.name+'.part');h=sha256();start=time.monotonic()
        with get(path) as source,temporary.open('wb') as out:
            for block in iter(lambda:source.read(1024*1024),b''):out.write(block);h.update(block)
        assert h.hexdigest()==digest,f'Checksum mismatch: {target.name}'
        temporary.replace(target)
        print('VERIFIED',target.name,'bytes',target.stat().st_size,'seconds',round(time.monotonic()-start,1),flush=True)
    def weight(r):download('/model/'+r['file'],model/r['file'],r['sha256'])
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(weight,records))
    while True:
        try:
            with get('/runtime_ready.json') as f:runtime=json.load(f)
            break
        except HTTPError as error:
            if error.code!=404:raise
            time.sleep(5)
    archive=ROOT/'runtime.tar.gz';download('/runtime.tar.gz',archive,runtime['sha256'])
    subprocess.run(['tar','-xzf',str(archive),'-C',str(ROOT)],check=True)
    subprocess.run([str(ROOT/'.venv-inference/bin/python'),'-m','pip','check'],check=True)
    (ROOT/'provision.complete').write_text(json.dumps(dict(source='pinned TLS transfer',model_files=len(records),runtime=runtime)))
    print('PROVISION_COMPLETE',flush=True)


if __name__=='__main__':main()
