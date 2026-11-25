"""Read-only history integrity checks and small inference controls."""
import concurrent.futures
import json
from pathlib import Path
import sys
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.search import memory_prompt

def freeze(x):return tuple(freeze(v) for v in x) if isinstance(x,list) else x
integrity={'records_checked':0,'prompt_mismatches':[],'invalid_unique_history':0,
           'all_unique_history':0,'zero_recovery_cases':0,'domains':{}}
for path in sorted((ROOT/'results/server/memory').glob('*.json')):
    if path.name=='status.json' or '.failed-' in path.name:continue
    run=json.loads(path.read_text())
    for case in run.get('cases',[]):
        history={}
        domain=case['domain']
        stats=integrity['domains'].setdefault(domain,{'records':0,'invalid':0,'unparseable':0,'valid_duplicate':0,'valid_new':0})
        valid_seen=set()
        integrity['zero_recovery_cases']+=int(case['metrics']['recovery']==0)
        for index,r in enumerate(case['records']):
            integrity['records_checked']+=1
            if r.get('prompt')!=memory_prompt(case['prompt'],history):
                integrity['prompt_mismatches'].append([path.name,index])
            if 'error' in r:continue
            stats['records']+=1
            key=freeze(r.get('hypothesis_key'))
            if r['valid']:
                stats['valid_duplicate' if key in valid_seen else 'valid_new']+=1
                valid_seen.add(key)
            else:
                stats['invalid']+=1
                stats['unparseable']+=int(key is None)
            if key is not None and key not in history:
                integrity['all_unique_history']+=1
                integrity['invalid_unique_history']+=int(not r['valid'])
                history[key]=r['text']

base_case=json.loads((ROOT/'results/server/memory/boolean-n1_inputs_00_outputs_0.json').read_text())['cases'][0]
tests=[
 ('arithmetic',8000,'Reply with only the integer: 2 + 2 = ?',None),
 ('nor_truth',8000,'Evaluate NOT (x OR y) when x=0 and y=0. Reply with only 0 or 1.',None),
 ('simple_boolean',8000,'Find one Boolean expression using x,y,AND,OR,NOT that evaluates to 0 when x=0,y=0. Reply only with Expression: followed by the expression.',None),
 ('full_prompt_replica0',8000,base_case['prompt'],None),
 ('full_prompt_replica1',8001,base_case['prompt'],None),
 ('full_prompt_fresh_cache',8000,base_case['prompt'],'ece4300-runtime-control-20261120'),
 ('bad_history',8000,base_case['records'][1]['prompt'],None),
 ('labeled_bad_history',8000,base_case['prompt']+'\n\nAn earlier candidate was NOR(x,y). It is INVALID: at x=0,y=0 it evaluates to 1, but the observation requires 0. Generate a different valid expression.',None),
]
def call(item):
    name,port,prompt,salt=item
    body={'model':'Qwen3-8B','messages':[{'role':'user','content':prompt}],
          'temperature':0,'seed':314159,'max_tokens':256,'chat_template_kwargs':{'enable_thinking':False}}
    if salt:body['cache_salt']=salt
    request=Request(f'http://127.0.0.1:{port}/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    try:
        with urlopen(request,timeout=600) as response:data=json.load(response)
        result={'name':name,'text':data['choices'][0]['message']['content'],'finish_reason':data['choices'][0]['finish_reason'],'usage':data.get('usage')}
    except Exception as error:result={'name':name,'error':str(error)}
    print(json.dumps(result),flush=True)
    return result
print(json.dumps(integrity,indent=2),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(call,tests))
out={'history_integrity':integrity,'inference_controls':results}
(ROOT/'logs/runtime_diagnosis.json').write_text(json.dumps(out,indent=2))
