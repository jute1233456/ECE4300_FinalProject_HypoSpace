"""Wait for the verified shared runtime, then start this node's dedicated replicas."""
import argparse
import json
from pathlib import Path
import socket
import time
from types import SimpleNamespace

from start_temp_replica import start


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--host',required=True)
    parser.add_argument('--gpus',required=True)
    args=parser.parse_args()
    deadline=time.monotonic()+1800
    while not (args.root/'provision.complete').exists():
        if time.monotonic()>deadline:raise RuntimeError('Shared provision timed out')
        time.sleep(5)
    for port,gpu in enumerate(args.gpus.split(','),18000):
        start(SimpleNamespace(host=args.host,gpu=gpu,port=port),args.root)
    (args.root/f'logs/start-{socket.gethostname()}.json').write_text(json.dumps(dict(
        status='replicas_started',host=args.host,gpus=args.gpus)))


if __name__=='__main__':main()
