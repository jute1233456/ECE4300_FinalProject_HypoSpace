"""Run MCTS 2.1 medium evaluation, then independent training-question rollouts."""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen

from queue_mcts21 import atomic,normal_complete
import fcntl


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--release',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();release=args.release.resolve()
    logdir=root/'logs/mcts-2.1-medium-pipeline-32k';logdir.mkdir(parents=True,exist_ok=True)
    lock=(logdir/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(version='2.1',pid=os.getpid(),release=str(release),workers_per_method=2)
    def update(status,**kwargs):
        state.update(status=status,**kwargs);atomic(logdir/'state.json',state)
        print(status,kwargs,flush=True)
    def run(command,env=None,log=None):
        with (log or logdir/'steps.log').open('a') as f:
            p=subprocess.Popen(command,cwd=release,env=env,stdin=subprocess.DEVNULL,
                               stdout=f,stderr=subprocess.STDOUT)
            update('running_step',child_pid=p.pid)
            code=p.wait()
            if code:raise RuntimeError(f'Step exited with code {code}')
    try:
        update('waiting_for_old_requests_to_save')
        while True:
            health=json.load(urlopen('http://127.0.0.1:8002/health',timeout=15))
            if not any(health['inflight_requests']):break
            time.sleep(5)
        # Terminate only paused legacy question workers, never API servers.
        for p in Path('/proc').iterdir():
            if not p.name.isdigit():continue
            try:
                a=(p/'cmdline').read_bytes().split(b'\0')
                if b'experiments.test_easy' not in a or b'--pause-file' not in a:continue
                pause=Path(a[a.index(b'--pause-file')+1].decode())
                if root not in pause.resolve().parents or not pause.exists():continue
                os.kill(int(p.name),signal.SIGTERM);print('STOPPED_SAVED_WORKER',p.name,flush=True)
            except (OSError,ValueError):pass
        pin=json.loads((release/'release_manifest.json').read_text())
        for name,digest in pin['sha256'].items():
            assert sha256((release/name).read_bytes()).hexdigest()==digest,f'Release changed: {name}'
        training=root/'datasets/medium_training_v1'
        for role,data in [('evaluation',release/'datasets/medium_sampled'),('training',training)]:
            if role=='training':
                update('generating_independent_training_questions')
                if not training.exists():
                    run([str(root/'.venv-tree-tools/bin/python'),'-u','-m',
                         'experiments.generate_medium_training','--project-root',str(release),'--output',str(training)])
                else:
                    assert (training/'generation_manifest.json').exists(),'Unverified training dataset exists'
            output=root/f'results/server-mcts-2.1-medium-{role}-32k'
            logs=root/f'logs/mcts-2.1-medium-{role}-32k';logs.mkdir(parents=True,exist_ok=True)
            env=dict(os.environ)
            env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                PYTHONHASHSEED='0',TOKENIZERS_PARALLELISM='false',ECE_THINKING='1',
                ECE_MAX_TOKENS='32768',ECE_WORKERS_PER_METHOD='2',ECE_DATASET_SUITE='medium',
                ECE_DATASET_ROOT=str(data),ECE_TREE_TOOLS='1',ECE_METHODS='mcts,two-step',
                ECE_MCTS_VERSION='2.1',ECE_RESULT_DIR=str(output),ECE_LOG_DIR=str(logs))
            for k in ('ECE_FULL_TREE','ECE_API_BASE','ECE_API_PORTS'):env.pop(k,None)
            update('starting_dataset',role=role,dataset=str(data),output=str(output))
            run([str(root/'.venv-tree-tools/bin/python'),'-u','server/run_evaluations.py'],env,logs/'runner.log')
            manifest=json.loads((output/'manifest.json').read_text())
            assert manifest.get('mcts_version')=='2.1'
            assert normal_complete(output,manifest['jobs']),'Some questions did not finish normally'
            update('dataset_complete',role=role)
        update('exporting_training_trajectories')
        run([str(root/'.venv-tree-tools/bin/python'),'-u','server/export_training_trajectories.py',
             '--run-dir',str(root/'results/server-mcts-2.1-medium-training-32k')])
        update('complete',child_pid=None)
    except Exception as error:
        update('failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':main()
