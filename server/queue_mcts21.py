"""Persistently start pinned MCTS 2.1 runs only after the 2.0 run is complete."""
import argparse
from datetime import datetime, timezone
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

METHODS = ('mcts', 'two-step')


def normal_complete(folder, jobs):
    for method in METHODS:
        for domain, case_id in jobs:
            p = folder/method/f'{domain}-{case_id}.json'
            if not p.exists():
                return False
            d = json.loads(p.read_text())
            if not d.get('completed') or len(d.get('cases', [])) != 1:
                return False
            c = d['cases'][0]
            if (c.get('domain'), c.get('case_id')) != (domain, case_id) or c.get('stop_reason')=='request_errors':
                return False
    return True


def active_run(folder):
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p/'cmdline').read_bytes().split(b'\0')
            if b'server/run_evaluations.py' in args:
                env = (p/'environ').read_bytes().split(b'\0')
                if ('ECE_RESULT_DIR='+str(folder)).encode() in env:
                    return True
            if b'experiments.test_easy' in args and any(str(folder).encode()+b'/' in a for a in args):
                return True
        except (OSError, ValueError):
            pass
    return False


def atomic(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--release', type=Path, required=True)
    args = parser.parse_args()
    root, release = args.root.resolve(), args.release.resolve()
    queue = root/'logs/mcts-2.1-queue'
    queue.mkdir(parents=True, exist_ok=True)
    lock = (queue/'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pin = json.loads((release/'release_manifest.json').read_text())
    predecessor = root/'results/server-easy-tree-tools-v2'
    predecessor_jobs = json.loads((predecessor/'manifest.json').read_text())['jobs']
    stages = [('easy', root/'results/server-mcts-2.1-easy', root/'logs/mcts-2.1-easy'),
              ('medium', root/'results/server-mcts-2.1-medium', root/'logs/mcts-2.1-medium')]
    state = dict(method_version='2.1', release=str(release), predecessor=str(predecessor),
                 stages=[dict(suite=s,output=str(o),logs=str(l)) for s,o,l in stages],
                 worker_pid=os.getpid())

    def update(status, **values):
        state.update(status=status, updated=datetime.now(timezone.utc).isoformat(), **values)
        atomic(queue/'state.json', state)
        print(status, values, flush=True)

    def verify_pin():
        for name, digest in pin['sha256'].items():
            if sha256((release/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'Pinned release changed: {name}')

    update('waiting_for_2.0')
    while not normal_complete(predecessor, predecessor_jobs) or active_run(predecessor):
        time.sleep(15)
    verify_pin()
    for suite, output, logs in stages:
        manifest = output/'manifest.json'
        if manifest.exists():
            data = json.loads(manifest.read_text())
            if data.get('mcts_version')!='2.1':
                raise RuntimeError('Existing output is not MCTS 2.1')
            if normal_complete(output, data['jobs']) and not active_run(output):
                update('stage_already_complete', suite=suite)
                continue
        if active_run(output):
            update('waiting_for_existing_stage', suite=suite)
            while active_run(output):
                time.sleep(15)
            if manifest.exists() and normal_complete(output, json.loads(manifest.read_text())['jobs']):
                continue
            raise RuntimeError('Existing stage stopped before completion; inspect before restarting')
        verify_pin()
        logs.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            PYTHONHASHSEED='0', TOKENIZERS_PARALLELISM='false',
            ECE_THINKING='1', ECE_MAX_TOKENS='16384', ECE_WORKERS_PER_METHOD='1',
            ECE_DATASET_SUITE=suite, ECE_TREE_TOOLS='1', ECE_METHODS='mcts,two-step',
            ECE_MCTS_VERSION='2.1', ECE_RESULT_DIR=str(output), ECE_LOG_DIR=str(logs))
        for key in ('ECE_FULL_TREE','ECE_API_BASE','ECE_API_PORTS'):
            env.pop(key, None)
        env.update(pin.get('runtime_env', {}))
        update('starting', suite=suite)
        with (logs/'runner.log').open('a') as log:
            child = subprocess.Popen([str(root/'.venv-tree-tools/bin/python'), '-u',
                'server/run_evaluations.py'], cwd=release, env=env, stdout=log,
                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            update('running', suite=suite, child_pid=child.pid)
            code = child.wait()
        if code or not manifest.exists() or not normal_complete(output, json.loads(manifest.read_text())['jobs']):
            update('failed', suite=suite, exit_code=code)
            raise SystemExit(1)
        update('stage_complete', suite=suite)
    update('complete', child_pid=None)


if __name__=='__main__':
    main()
