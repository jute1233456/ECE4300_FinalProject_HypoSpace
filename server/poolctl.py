"""CLI for the private dynamic pool; reserve drains before releasing GPU memory."""
import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def call(url, payload=None, token=None, timeout=900):
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    request = Request(url, data=json.dumps(payload).encode() if payload is not None else None, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def change_role(base, token, identity, role, owner=None):
    return call(base+'/admin/backends/'+identity,dict(role=role,owner=owner),token)


def drain(base, token, identities, owner, timeout=3600):
    health = call(base+'/health')
    chosen = [e for e in health['backends'] if e['id'] in identities]
    if len(chosen) != len(identities):
        raise ValueError('Unknown or duplicate GPU')
    for entry in chosen:
        if entry.get('owner') not in (None, owner):
            raise ValueError('GPU is reserved by another owner')
    for identity in identities:
        change_role(base, token, identity, 'draining', owner)
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        entries = [e for e in call(base+'/health')['backends'] if e['id'] in identities]
        unknown = [e['id'] for e in entries if e['observed'] is None]
        if unknown:
            raise TimeoutError('Cannot prove idle (engine metrics unavailable) for '+','.join(unknown)
                               +'; role left as draining, no service touched')
        if all(e['inflight'] == 0 and e['observed'] == 0 for e in entries):
            return entries
        time.sleep(2)
    raise TimeoutError('Drain timed out; active requests and services were preserved')


def activate(base, token, identities, role, owner=None):
    entries = {e['id']:e for e in call(base+'/health')['backends']}
    for identity in identities:
        entry=entries[identity]
        if entry.get('owner') not in (None, owner):
            raise ValueError('GPU is reserved by another owner')
        ready=call(entry['node_api']+'/gpus/'+str(entry['gpu'])+'/serve',{},token)
        patch=dict(url=ready['url'],models=ready.get('models',entry['models']),role=role,owner=None)
        if ready.get('managed_auth',ready.get('status')=='ready'):
            patch['api_key']=token
        call(base+'/admin/backends/'+identity,patch,token)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('action',choices=['status','assign','drain','reserve','release','workers','load'])
    parser.add_argument('gpus',nargs='*')
    parser.add_argument('--role',choices=['baseline','validation','shared','draining'],default='validation')
    parser.add_argument('--owner',default='manual')
    parser.add_argument('--count',type=int)
    parser.add_argument('--name');parser.add_argument('--path');parser.add_argument('--sha256')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    token=Path(config['token_file']).read_text().strip();base=config.get('gateway','http://127.0.0.1:8005')
    if args.action=='status':
        print(json.dumps(call(base+'/health'),indent=2));return
    if args.action=='workers':
        print(call(base+'/admin/validation-workers',dict(count=args.count),token));return
    if not args.gpus:parser.error('Specify at least one GPU ID')
    if args.action=='assign':
        entries={e['id']:e for e in call(base+'/health')['backends']}
        for identity in args.gpus:
            if entries[identity].get('owner') not in (None,args.owner):raise ValueError('Reserved by another owner')
            change_role(base,token,identity,args.role,None)
    elif args.action in ('drain','reserve','load'):
        entries=drain(base,token,args.gpus,args.owner)
        for entry in entries:
            if args.action=='reserve':
                call(entry['node_api']+'/gpus/'+str(entry['gpu'])+'/stop',{},token)
                change_role(base,token,entry['id'],'training',args.owner)
            elif args.action=='load':
                if not all([args.name,args.path,args.sha256]):raise ValueError('load requires name, path and sha256')
                call(entry['node_api']+'/gpus/'+str(entry['gpu'])+'/load',dict(name=args.name,path=args.path,sha256=args.sha256),token)
                call(base+'/admin/backends/'+entry['id'],dict(models=sorted(set(entry['models']+[args.name])),role=args.role,owner=None),token)
    elif args.action=='release':
        activate(base,token,args.gpus,args.role,args.owner)
    print(json.dumps(dict(status='ok',action=args.action,gpus=args.gpus)))


if __name__=='__main__':main()
