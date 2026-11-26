"""Drain a local GPU pair, run a training job, always return the pair to inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from server.poolctl import call, change_role, drain, activate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--gpus',nargs='+',required=True)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--owner',default='training')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());job=json.loads(args.job.read_text())
    base=cfg['gateway'];token=Path(cfg['token_file']).read_text().strip()
    selected=[e for e in call(base+'/health')['backends'] if e['id'] in args.gpus]
    if len(selected)!=len(args.gpus) or len({e['node'] for e in selected})!=1:
        raise ValueError('All selected GPUs must be registered on one node')
    if selected[0]['node']!=cfg['local_node']:
        raise ValueError('Run the training wrapper on the GPU node, not on another VM')
    if not isinstance(job.get('command'),list) or not all(isinstance(x,str) for x in job['command']):
        raise ValueError('Job must contain an argv list; shell commands are not accepted')
    reserved=[];code=1
    try:
        entries=drain(base,token,args.gpus,args.owner)
        for entry in entries:
            call(entry['node_api']+'/gpus/'+str(entry['gpu'])+'/stop',{},token)
            change_role(base,token,entry['id'],'training',args.owner)
            reserved.append(entry['id'])
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(str(e['gpu']) for e in selected),
                 OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false')
        print(json.dumps(dict(phase='training',gpus=args.gpus,command=job['command'])),flush=True)
        code=subprocess.run(job['command'],cwd=job['cwd'],env=env).returncode
    finally:
        # No training process is killed. If it failed, release is still attempted after it exits.
        with ThreadPoolExecutor(max_workers=len(selected)) as pool:
            futures=[pool.submit(activate,base,token,[e['id']],'shared',args.owner) for e in selected]
            for future in futures:future.result()
        print(json.dumps(dict(phase='returned_to_inference',exit_code=code,gpus=args.gpus)),flush=True)
    return code


if __name__=='__main__':sys.exit(main())
