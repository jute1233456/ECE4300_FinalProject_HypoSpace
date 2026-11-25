"""Re-score saved raw candidates against the fixed datasets and audit completeness."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.direct_evaluation import load_domain

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--method', choices=['direct','memory','mcts','two-step'], action='append')
args=parser.parse_args()
jobs={}
for dataset in ('causal','3d','boolean','boolean_extended'):
    domain='boolean' if dataset=='boolean_extended' else dataset
    benchmark,prepare,score=load_domain(domain,ROOT/f'datasets/balanced/{dataset}.json')
    for case in benchmark.all_observation_sets:
        case_id=case.get('observation_set_id') or case['observation_id']
        observations,keys=prepare(case)
        jobs[f'{dataset}-{case_id}']=(domain,case_id,observations,keys,score)

failed=False
for method in args.method or ('direct','memory','mcts','two-step'):
    report={'method':method,'expected_cases':len(jobs),'completed_cases':0,'requests':0,
            'request_errors':0,'truncated_responses':0,'mismatches':[],'missing_cases':[]}
    for name,(domain,case_id,observations,keys,score) in jobs.items():
        path=ROOT/f'results/server/{method}/{name}.json'
        if not path.exists():
            report['missing_cases'].append(name);continue
        data=json.loads(path.read_text())
        if not data.get('completed') or len(data.get('cases',[]))!=1:
            report['missing_cases'].append(name);continue
        case=data['cases'][0]
        report['completed_cases']+=1
        if (case['domain'],case['case_id'],case['reference_count'])!=(domain,case_id,len(keys)):
            report['mismatches'].append([name,'case_identity'])
        if case.get('stop_reason')=='request_errors':report['mismatches'].append([name,'request_errors_stop'])
        unique,recovered=set(),set()
        valid_count=submissions=0
        for index,record in enumerate(case['records']):
            report['requests']+=1+int('reflection_response' in record)
            report['truncated_responses']+=int(record.get('finish_reason')=='length')+int(record.get('reflection_response',{}).get('finish_reason')=='length')
            if 'error' in record:
                report['request_errors']+=1;continue
            answer=record.get('answer',record['text'])
            key,valid=score(answer,observations) if answer.strip() else (None,False)
            if valid!=record.get('valid') or json.dumps(key)!=json.dumps(record.get('hypothesis_key')):
                report['mismatches'].append([name,index,'raw_candidate_score'])
            valid_count+=int(valid)
            submitted=record.get('submitted',False)
            submissions+=int(submitted)
            if submitted and key is not None:
                unique.add(key)
                if valid and key in keys:recovered.add(key)
        expected={'validity':valid_count/max(submissions,1),'uniqueness':len(unique)/max(submissions,1),
                  'recovery':len(recovered)/len(keys)}
        if submissions!=case['submitted'] or any(not math.isclose(value,case['metrics'][metric],abs_tol=1e-12) for metric,value in expected.items()):
            report['mismatches'].append([name,'aggregate_metrics'])
    report['passed']=(report['completed_cases']==len(jobs) and not report['missing_cases']
                      and not report['mismatches'] and not report['request_errors'] and not report['truncated_responses'])
    (ROOT/f'logs/audit_{method}.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    failed|=not report['passed']
if failed:raise SystemExit(1)
