#!/usr/bin/env python3
"""Queue-compatible Python launcher; routes only explicitly listed experiments."""
import json
import os
from pathlib import Path
import sys


def prepare(args, cfg):
    args = list(args)
    model = args[args.index('--model')+1] if '--model' in args else None
    route = cfg['routes'].get(model)
    if route:
        if args[:2] != ['-u','-m'] or args[2] != 'experiments.test_hard_balanced':
            raise ValueError('Unexpected hybrid queue entry point')
        if '--reflection-model' in args:
            raise ValueError('Do not override a frozen stage routing')
        args[args.index('--model')+1] = route['search_model']
        if route.get('reflection_model'):
            args.extend(['--reflection-model', route['reflection_model']])
        return args, route['code']
    return args, None


def main():
    cfg = json.loads(Path(__file__).with_name('hybrid-worker-config.json').read_text())
    args, cwd = prepare(sys.argv[1:], cfg)
    if cwd: os.chdir(cwd)
    os.execv(cfg['python'], [cfg['python'], *args])


if __name__ == '__main__':
    main()
