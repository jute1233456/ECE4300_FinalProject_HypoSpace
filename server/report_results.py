"""Aggregate completed per-case files and expose failures/truncation explicitly."""
import csv
import argparse
import json
from pathlib import Path
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.direct_evaluation import summarize

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-dir',type=Path,default=ROOT/'results/server')
args=parser.parse_args()
out=args.run_dir
methods = ('direct', 'memory', 'mcts', 'two-step')
runs = {}
for name in methods:
    cases = []
    sources = []
    for path in sorted((out / name).glob('*.json')):
        if path.name == 'status.json' or '.failed-' in path.name:
            continue
        data = json.loads(path.read_text())
        if data.get('completed'):
            cases.extend(data['cases'])
            sources.append(path.name)
    runs[name] = {'cases': cases, 'sources': sources, 'summary': summarize(cases)}
common = set.intersection(*[{(c['domain'], c['case_id']) for c in runs[m]['cases']
                            if c.get('stop_reason') != 'request_errors'} for m in methods])
rows = []
for name, data in runs.items():
    cases = data['cases']
    records = [r for c in cases for r in c['records']]
    common_cases = [c for c in cases if (c['domain'], c['case_id']) in common]
    data['common_summary'] = summarize(common_cases)
    row = {'method': name, 'completed_cases': len(cases), 'common_cases': len(common),
           'recovery': data['summary'].get('all', {}).get('recovery'),
           'common_recovery': data['common_summary'].get('all', {}).get('recovery'),
           'validity': data['summary'].get('all', {}).get('validity'),
           'uniqueness': data['summary'].get('all', {}).get('uniqueness'),
           'search_requests': len(records),
           'api_requests': sum(len(r.get('api_requests', [])) or 1 for r in records)
               + sum('reflection_response' in r for r in records),
           'request_errors': sum('error' in r for r in records),
           'sum_case_wall_seconds': sum(c.get('seconds', 0) for c in cases),
           'truncated_responses': sum(
               sum(a.get('response', {}).get('choices', [{}])[0].get('finish_reason') == 'length'
                   for a in r['api_requests']) if r.get('api_requests') else r.get('finish_reason') == 'length'
               for r in records)
               + sum(r.get('reflection_response', {}).get('finish_reason') == 'length' for r in records),
           'total_tokens': data['summary'].get('all', {}).get('total_tokens', 0)}
    row['reasoning_tokens'] = sum((r.get('reasoning_tokens') or 0)+(r.get('reflection_response',{}).get('reasoning_tokens') or 0) for r in records)
    row['missing_final_answers'] = sum(bool(r.get('final_answer_missing'))+bool(r.get('reflection_response',{}).get('final_answer_missing')) for r in records)
    row['tool_reads'] = sum(len(r.get('tool_trace', [])) for r in records)
    row['protocol_errors'] = sum(bool(r.get('protocol_error')) for r in records)
    rows.append(row)
    row['archived_failed_attempts'] = len(list((out/name).glob('*.failed-*.json')))
    (out / f'qwen3_8b_{name}_aggregate.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
(out / 'comparison.json').write_text(json.dumps(rows, indent=2))
with (out / 'comparison.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(rows, indent=2))
