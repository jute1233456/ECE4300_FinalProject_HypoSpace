"""Compare only matched, normally completed original/new easy questions."""
import argparse
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]


def load(folder):
    cases = {}
    errors = []
    for p in sorted(folder.glob('*.json')):
        if p.name == 'status.json' or '.tree.' in p.name or '.failed-' in p.name:
            continue
        d = json.loads(p.read_text())
        if not isinstance(d, dict):
            continue
        for c in d.get('cases', []):
            if d.get('completed') and c.get('stop_reason') != 'request_errors':
                cases[(c['domain'], c['case_id'])] = (d, c)
            elif c.get('stop_reason') == 'request_errors':
                errors.append(p.name)
    return cases, errors


def metrics(rows):
    if not rows:
        return dict(cases=0)
    records = [r for _, c in rows for r in c['records']]
    requests = [a for r in records for a in r.get('api_requests', [])]
    valid = [r for r in records if r.get('valid')]
    hits = [(d, c, {str(r.get('hypothesis_key')) for r in c['records'] if r.get('reference_hit')}) for d,c in rows]
    distinct = sum(len(keys) for _,_,keys in hits)
    api_calls = sum(len(r.get('api_requests', [])) or 1 for r in records)
    api_calls += sum(bool(r.get('reflection_response')) for r in records)
    tokens = sum((r.get('total_tokens') or 0)+(r.get('reflection_response',{}).get('total_tokens') or 0) for r in records)
    truncated = sum(a.get('response',{}).get('choices',[{}])[0].get('finish_reason')=='length' for a in requests)
    truncated += sum(r.get('finish_reason')=='length' for r in records if not r.get('api_requests'))
    truncated += sum(r.get('reflection_response',{}).get('finish_reason')=='length' for r in records)
    return dict(cases=len(rows), recovery=mean(c['metrics']['recovery'] for _,c in rows),
        distinct_reference_hits=distinct, search_rounds=len(records), valid_search_fraction=len(valid)/max(1,len(records)),
        valid_duplicate_fraction=(1-sum(len({str(r.get('hypothesis_key')) for r in c['records'] if r.get('valid')})
            for _,c in rows)/len(valid)) if valid else None,
        api_calls=api_calls, tool_reads=sum(len(r.get('tool_trace',[])) for r in records), total_tokens=tokens,
        tokens_per_distinct_hit=tokens/distinct if distinct else None,
        format_repairs=sum(r.get('format_repairs',0) for r in records), protocol_errors=sum(bool(r.get('protocol_error')) for r in records),
        request_errors=sum('error' in r for r in records), truncated=truncated)


def compare(old, new):
    report = dict(old=str(old), new=str(new), groups={},
        interpretation='Same empty start, dataset hashes, scoring, model and search/submission budgets; tool requests are additional work. Single stochastic run, not matched random seeds.')
    for method in ('mcts','two-step'):
        older, oe = load(old/method)
        newer, ne = load(new/method)
        common = sorted(older.keys() & newer.keys())
        matched = []
        excluded = []
        for key in common:
            od,oc = older[key]; nd,nc = newer[key]
            fields = ('requested_model','max_tokens','datasets','local_api','thinking_config','model_manifest','method')
            old_tree = dict(od.get('tree_config',{})); new_tree = dict(nd.get('tree_config',{}))
            new_tree.pop('tree_tools',None)
            if old_tree!=new_tree or any(od.get(f)!=nd.get(f) for f in fields) or any(oc.get(f)!=nc.get(f) for f in ('prompt','reference_count','submission_cap','model_call_cap')):
                excluded.append(key)
            else:
                matched.append(key)
        report['groups'][method] = dict(old_completed=len(older), new_completed=len(newer),
            matched_cases=matched, excluded_config_mismatch=excluded, failed_cases={'old':oe,'new':ne},
            old=metrics([older[k] for k in matched]), new=metrics([newer[k] for k in matched]),
            domains={domain:dict(old=metrics([older[k] for k in matched if k[0]==domain]),
                                new=metrics([newer[k] for k in matched if k[0]==domain]))
                     for domain in ('causal','3d','boolean')})
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old',type=Path,default=ROOT/'results/server-easy-thinking-v1')
    parser.add_argument('--new',type=Path,default=ROOT/'results/server-easy-tree-tools-v2')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=compare(args.old,args.new)
    if args.output:
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
