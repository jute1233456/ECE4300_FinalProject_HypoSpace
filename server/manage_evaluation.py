"""Pause and stop only this project's evaluator processes at saved boundaries."""
import os
from pathlib import Path
import signal
import sys
from urllib.request import urlopen

root = Path(__file__).resolve().parents[1]
methods = ('direct','memory','mcts','two-step')
action = sys.argv[1]


def processes(needle):
    found=[]
    for item in Path('/proc').iterdir():
        if not item.name.isdigit(): continue
        try:
            if (item/'cwd').resolve() != root: continue
            args=(item/'cmdline').read_bytes().split(b'\0')
            if needle in args: found.append(int(item.name))
        except (OSError,ValueError): pass
    return found


if action == 'pause':
    for name in methods: (root/f'results/server/PAUSE_{name}').touch()
    for pid in processes(b'server/run_evaluations.py'):
        os.kill(pid, signal.SIGTERM)
        print('Stopped coordinator',pid)
    print('Pause files created; wait for active API requests to finish before stop.')
elif action == 'stop':
    for port in (8000,8001):
        text=urlopen(f'http://127.0.0.1:{port}/metrics').read().decode()
        for line in text.splitlines():
            if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')) and float(line.split()[-1]) != 0:
                raise SystemExit('API requests still active; try stop after they finish.')
    for pid in processes(b'experiments.test_hard_balanced'):
        os.kill(pid,signal.SIGTERM)
        print('Stopped paused evaluator',pid)
elif action == 'resume':
    for name in methods: (root/f'results/server/PAUSE_{name}').unlink(missing_ok=True)
else:
    raise SystemExit('Expected pause, stop or resume')
