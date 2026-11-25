"""Continue only two-step MCTS 2.1, preserving existing active question writers."""
import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--release',type=Path,required=True);args=parser.parse_args()
    root,release=args.root.resolve(),args.release.resolve()
    logs=root/'logs/mcts-2.1-reflection-only';logs.mkdir(parents=True,exist_ok=True)
    lock=(logs/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(version='2.1',methods=['two-step'],requested_max_tokens=32768,workers_per_method=3,pid=os.getpid(),release=str(release))
    def update(status,**fields):
        state.update(status=status,**fields);p=logs/'state.tmp'
        p.write_text(json.dumps(state,indent=2));p.replace(logs/'state.json')
        print(status,fields,flush=True)
    def run(command,env=None,log=None):
        with (log or logs/'steps.log').open('a') as f:
            p=subprocess.Popen(command,cwd=release,env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            update('running',child_pid=p.pid)
            if p.wait():raise RuntimeError('Child process failed; inspect logs')
    try:
        pin=json.loads((release/'release_manifest.json').read_text())
        for name,digest in pin['sha256'].items():
            assert sha256((release/name).read_bytes()).hexdigest()==digest,f'Source changed: {name}'
        training=root/'datasets/medium_training_v1'
        for role,data in [('evaluation',release/'datasets/medium_sampled'),('training',training)]:
            if role=='training' and not training.exists():
                update('generating_training_questions',role=role)
                run([str(root/'.venv-tree-tools/bin/python'),'-u','-m','experiments.generate_medium_training',
                    '--project-root',str(release),'--output',str(training)])
            output=root/f'results/server-mcts-2.1-medium-{role}-32k'
            stage_logs=root/f'logs/mcts-2.1-medium-{role}-32k';stage_logs.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ)
            env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONHASHSEED='0',
                TOKENIZERS_PARALLELISM='false',ECE_THINKING='1',ECE_MAX_TOKENS='32768',
                ECE_WORKERS_PER_METHOD='3',ECE_DATASET_SUITE='medium',ECE_DATASET_ROOT=str(data),
                ECE_TREE_TOOLS='1',ECE_METHODS='two-step',ECE_MCTS_VERSION='2.1',
                ECE_RESULT_DIR=str(output),ECE_LOG_DIR=str(stage_logs),ECE_WAIT_EXISTING_WRITERS='1')
            for k in ('ECE_FULL_TREE','ECE_API_BASE','ECE_API_PORTS'):env.pop(k,None)
            update('starting',role=role,output=str(output))
            run([str(root/'.venv-tree-tools/bin/python'),'-u','server/run_evaluations.py'],env,stage_logs/'runner.log')
            manifest=json.loads((output/'manifest.json').read_text())
            for domain,case_id in manifest['jobs']:
                d=json.loads((output/'two-step'/f'{domain}-{case_id}.json').read_text())
                assert d.get('completed') and len(d.get('cases',[]))==1
                assert d['cases'][0].get('stop_reason')!='request_errors'
            update('stage_complete',role=role)
        run([str(root/'.venv-tree-tools/bin/python'),'-u','server/export_training_trajectories.py',
            '--run-dir',str(root/'results/server-mcts-2.1-medium-training-32k')])
        update('complete',child_pid=None)
    except Exception as error:
        update('failed',error=f'{type(error).__name__}: {error}');raise


if __name__=='__main__':main()
