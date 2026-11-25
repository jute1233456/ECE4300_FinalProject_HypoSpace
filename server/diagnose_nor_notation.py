"""Offline sensitivity check: normalize only unambiguous parenthesized infix NOR."""
import json,re,sys
from pathlib import Path
from statistics import mean
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from experiments.direct_evaluation import load_domain


def normalize_infix_nor(expression,variables):
    atom='(?:'+'|'.join(re.escape(v) for v in sorted(variables,key=len,reverse=True))+')'
    pair=re.compile(r'\(\s*('+atom+r')\s+NOR\s+('+atom+r')\s*\)',re.I)
    expression=pair.sub(lambda m:'(NOR('+m[1]+','+m[2]+'))',expression)
    whole=re.fullmatch(r'\s*('+atom+r')\s+NOR\s+('+atom+r')\s*',expression,re.I)
    if whole:expression='NOR('+whole[1]+','+whole[2]+')'
    return expression

summaries=[];examples=[]
for dataset in ('boolean','boolean_extended'):
    benchmark,prepare,score=load_domain('boolean',root/f'datasets/balanced/{dataset}.json')
    jobs={}
    for c in benchmark.all_observation_sets:
        jobs[c.get('observation_set_id') or c['observation_id']]=prepare(c)
    for cid,(observations,keys) in jobs.items():
        path=root/f'results/server/memory/{dataset}-{cid}.json'
        run=json.loads(path.read_text());case=run['cases'][0]
        new_valid=old_valid=0;new_recovered=set();old_recovered=set();changed=rescued=0
        for i,r in enumerate(case['records']):
            if 'text' not in r:continue
            key,valid=score(r['text'],observations)
            old_valid+=valid
            if valid and key in keys:old_recovered.add(key)
            expr=benchmark.parse_llm_response(r['text'])
            converted=normalize_infix_nor(expr,benchmark.variables) if expr and key is None else expr
            if expr and converted!=expr:
                changed+=1
                key2,valid2=score('Expression: '+converted,observations)
                if valid2:
                    rescued+=1
                    if len(examples)<8:examples.append({'case':cid,'record':i,'original':expr,'normalized':converted,'original_valid':valid,'normalized_valid':valid2})
                key,valid=key2,valid2
            new_valid+=valid
            if valid and key in keys:new_recovered.add(key)
        n=len(case['records'])
        summaries.append({'case':cid,'calls':n,'changed_notation':changed,'rescued_valid_answers':rescued,
            'old_validity':old_valid/n,'new_validity':new_valid/n,
            'old_recovery':len(old_recovered)/len(keys),'new_recovery':len(new_recovered)/len(keys)})
out={'cases':len(summaries),'changed_notation':sum(r['changed_notation'] for r in summaries),
     'rescued_valid_answers':sum(r['rescued_valid_answers'] for r in summaries),
     **{k:mean(r[k] for r in summaries) for k in ['old_validity','new_validity','old_recovery','new_recovery']},
     'examples':examples,'per_case':summaries}
(root/'logs/nor_notation_diagnosis.json').write_text(json.dumps(out,indent=2))
print(json.dumps({k:v for k,v in out.items() if k!='per_case'},indent=2))
