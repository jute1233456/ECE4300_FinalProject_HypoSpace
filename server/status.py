"""Compact progress snapshot; safe while evaluation writers save atomically."""
import json
import argparse
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir',type=Path,default=root/'results/server')
parser.add_argument('--log-dir',type=Path,default=root/'logs')
args=parser.parse_args()
manifest = args.run_dir/'manifest.json'
total = len(json.loads(manifest.read_text())['jobs']) if manifest.exists() else 136
for name in ('bootstrap.complete','model_source.json','trainability.json','evaluations.started','evaluations.complete'):
    path = (root/'logs' if name in ('bootstrap.complete','model_source.json','trainability.json') else args.log_dir)/name
    print(name, 'exists' if path.exists() else 'pending')
methods = json.loads(manifest.read_text()).get('methods', ('direct','memory','mcts','two-step')) if manifest.exists() else ('direct','memory','mcts','two-step')
for method in methods:
    folder = args.run_dir/method
    done, attempts, skipped, maximum = 0, 0, 0, 0
    for path in folder.glob('*.json'):
        if path.name == 'status.json' or '.failed-' in path.name:
            continue
        try:
            data = json.loads(path.read_text())
            done += int(data.get('completed', False))
            for case in data.get('cases', []) + ([data['interrupted_case']] if data.get('interrupted_case') else []):
                attempts += len(case['records'])
                skipped += int(case.get('stop_reason') == 'request_errors')
                maximum = max(maximum, max((r.get('prompt_tokens') or 0 for r in case['records']), default=0))
        except (ValueError, OSError):
            pass
    print(f'{method}: completed={done}/{total} search_attempts={attempts} skipped={skipped} max_prompt_tokens={maximum}')
