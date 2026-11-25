"""Isolate operator vocabulary, task constraints, and thinking mode."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import sys,time
from urllib.request import Request,urlopen

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from experiments.direct_evaluation import load_domain

benchmarks={}
for domain in ('boolean','3d','causal'):
    b,prepare,score=load_domain(domain,root/f'datasets/balanced/{domain}.json')
    benchmarks[domain]=(b,prepare,score)

def case_data(domain,cid):
    b,prepare,score=benchmarks[domain]
    c=next(c for c in b.all_observation_sets if (c.get('observation_set_id') or c.get('observation_id'))==cid)
    observations,keys=prepare(c)
    record=json.loads((root/f'results/server/memory/{domain}-{cid}.json').read_text())['cases'][0]
    return observations,score,record

obs,score,original=case_data('boolean','n1_inputs_00_outputs_0')
base='Find ONE Boolean expression using variables x and y. It must output 0 when x=0,y=0. Return exactly one line beginning Expression:. '
prompts={
    'simple_AND_OR_NOT':base+'Allowed operators: AND, OR, NOT.',
    'simple_with_NOR':base+'Allowed operators: AND, OR, NOT, NOR.',
    'with_depth':base+'Allowed operators: AND, OR, NOT, NOR. Maximum nesting depth is 2. No constants.',
    'with_structural_rules':base+'Allowed operators: AND, OR, NOT, NOR. Maximum nesting depth is 2. No constants. Operand reordering, duplicate AND/OR inputs and associative regrouping count as the same mechanism; distributivity, absorption and De Morgan transformations do not identify mechanisms.',
    'short_labeled_error':base+'Allowed operators: AND, OR, NOT. Previous candidate NOT (x OR y) is INVALID because it outputs 1 at x=0,y=0, while the required output is 0. Return a different valid expression.',
    'formal_prompt':original['prompt'],
}
tasks=[]
for name,prompt in prompts.items():
    for seed in (101,102,103):tasks.append((name,'boolean',prompt,False,seed,256,obs,score))
for domain,cid in [('boolean','n1_inputs_00_outputs_0'),('3d','g3_h3_b3_010'),('causal','n6_hard_016')]:
    observations,scorer,data=case_data(domain,cid)
    for thinking in (False,True):tasks.append((f'mode_{domain}',domain,data['prompt'],thinking,20261120,8192,observations,scorer))
feedback=json.loads((root/'results/feedback-memory-pilot-v1/feedback-memory-boolean-n1_inputs_00_outputs_0-r0.json').read_text())['records'][1]['prompt']
for thinking in (False,True):tasks.append(('mode_boolean_feedback','boolean',feedback,thinking,20261120,8192,obs,score))

def run(task):
    name,domain,prompt,thinking,seed,limit,observations,scorer=task
    body={'model':'Qwen3-8B','messages':[{'role':'user','content':prompt}],
          'temperature':0.7,'top_p':0.8,'top_k':20,'seed':seed,'max_tokens':limit,
          'chat_template_kwargs':{'enable_thinking':thinking}}
    start=time.monotonic()
    request=Request('http://127.0.0.1:8002/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    with urlopen(request,timeout=600) as response:data=json.load(response)
    text=data['choices'][0]['message']['content']
    has_final=not thinking or '</think>' in text
    answer=text.rsplit('</think>',1)[-1].strip() if has_final else ''
    key,valid=scorer(answer,observations) if answer else (None,False)
    result={'name':name,'domain':domain,'thinking':thinking,'seed':seed,'prompt':prompt,
            'answer':answer,'has_final':has_final,'valid':valid,'hypothesis_key':key,
            'finish_reason':data['choices'][0]['finish_reason'],'usage':data.get('usage'),
            'seconds':time.monotonic()-start}
    print(json.dumps({k:v for k,v in result.items() if k not in ('prompt','hypothesis_key')}),flush=True)
    return result

results=[]
with ThreadPoolExecutor(max_workers=2) as pool:
    futures=[pool.submit(run,task) for task in tasks]
    for future in as_completed(futures):
        results.append(future.result())
        (root/'logs/constraint_diagnosis.json').write_text(json.dumps(results,indent=2))
print('CONSTRAINT_DIAGNOSIS_COMPLETE',len(results),flush=True)
