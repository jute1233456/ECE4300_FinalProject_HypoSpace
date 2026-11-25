"""Few original benchmark cases; compare 16K with the maximum remaining 32K context."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import zlib

from agents.direct import DirectAgent
from agents.search import MemoryAgent
from agents.reasoning_tree import ReasoningTreeAgent
from llm.local_thinking import LocalThinkingClient
from .direct_evaluation import ROOT,load_domain

CASES=[('3d','g3_h3_b2_001'),('boolean','n3_inputs_01_10_11_outputs_110')]
LIMITS=(16384,32768)
METHODS=('direct','memory','mcts','two-step')
OUT=ROOT/'results/token-limit-pilot-v1'


class SeededClient(LocalThinkingClient):
    def __init__(self,seed,**kwargs):
        super().__init__(**kwargs);self.seed=seed;self.calls=0;self.last_seed=None
    def generate(self,*args,**kwargs):
        self.last_seed=(self.seed+self.calls)%2147483647
        self.generation_options['seed']=self.last_seed
        # Direct local replica access avoids the formal gateway's 600s timeout
        # when a high-budget thinking response runs longer. Alternate replicas.
        self.base_url=f'http://127.0.0.1:{8000+self.calls%2}/v1'
        self.last_endpoint=self.base_url
        response=super().generate(*args,**kwargs)
        self.calls+=1
        return response


def make_agent(method,client,limit):
    if method in ('mcts','two-step'):
        agent=ReasoningTreeAgent(client,'llm',7,-1,rotate=False,branch_examples=True,
                                 unbounded_tree=method=='two-step')
    else:agent=(DirectAgent if method=='direct' else MemoryAgent)(client)
    agent.max_tokens=limit
    return agent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    manifest={'cases':CASES,'methods':METHODS,'generation_limits':LIMITS,'context_window':32768,
              'budget':'original reference_count+5 submission allowance; tree search at most twice that; unbounded two-step has no separate submission cap',
              'workers':1,'thinking':True,
              'seed_policy':'paired across caps, increment for each request; not fixed per request',
              'unchanged':'original observations, hypothesis space, prompts, scoring and method settings',
              'note':'32K generation request is clamped to remaining context; this does not expand the total context window'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    jobs=[]
    for domain,cid in CASES:
        benchmark,prepare,score=load_domain(domain,ROOT/f'datasets/balanced/{domain}.json')
        case=next(c for c in benchmark.all_observation_sets if (c.get('observation_set_id') or c.get('observation_id'))==cid)
        observations,keys=prepare(case)
        print('CASE',domain,cid,'references',len(keys),flush=True)
        if args.plan:continue
        original=json.loads((ROOT/f'results/server/memory/{domain}-{cid}.json').read_text())['cases'][0]['prompt']
        for method in METHODS:
            for limit in LIMITS:jobs.append((domain,cid,observations,keys,score,original,method,limit))
    if args.plan:
        print(json.dumps(manifest,indent=2));return
    for job_index,(domain,cid,observations,keys,score,prompt,method,limit) in enumerate(jobs,1):
        path=OUT/f'{domain}-{cid}-{method}-cap{limit}.json'
        result=json.loads(path.read_text()) if path.exists() else {
            'domain':domain,'case_id':cid,'method':method,'requested_max_tokens':limit,
            'context_window':32768,'reference_count':len(keys),
            'round_cap':(len(keys)+5)*(2 if method in ('mcts','two-step') else 1),
            'prompt':prompt,'started_utc':datetime.now(timezone.utc).isoformat(),
            'records':[],'completed':False,'thinking':True,
            'sampling':{'temperature':0.7,'top_p':0.8,'top_k':20}}
        if result['completed']:continue
        client=SeededClient(zlib.crc32(f'{domain}/{cid}/{method}'.encode()),
            tokenizer_path=ROOT/'models/Qwen3-8B',context_window=32768,api_key='local',
            model='Qwen3-8B',base_url='http://127.0.0.1:8000/v1',timeout=1800,
            generation_options={'temperature':0.7,'top_p':0.8,'top_k':20,'chat_template_kwargs':{'enable_thinking':True}})
        agent=make_agent(method,client,limit)
        recovered=set();valid_keys=set()
        def observe(record):
            if method!='direct':agent.next_prompt(prompt)
            answer=agent.split_response(record['text'])[1] if method in ('mcts','two-step') else record['text']
            key,valid=score(answer,observations) if answer.strip() else (None,False)
            duplicate=valid and key in valid_keys
            if valid:valid_keys.add(key)
            if valid and key in keys:recovered.add(key)
            if method!='direct':agent.observe(record['text'],key,valid)
            return {'answer':answer,'valid':valid,'hypothesis_key':key,'duplicate_valid':bool(duplicate)}
        for old in result['records']:observe(old)
        client.calls=sum(1+int('reflection_response' in r) for r in result['records'])
        submitted=sum(r.get('submitted',method not in ('mcts','two-step') or r.get('valid',False)) for r in result['records'])
        def save():
            temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2));temporary.replace(path)
        for index in range(len(result['records']),result['round_cap']):
            if recovered==keys or (method!='two-step' and submitted>=len(keys)+5) or (isinstance(agent,ReasoningTreeAgent) and agent.exhausted):break
            query=prompt if method=='direct' else agent.next_prompt(prompt)
            reflection={}
            if method=='two-step':
                recent=[r['answer'] for r in result['records'] if r.get('answer')][-8:]
                review_prompt=(f'{prompt}\n\nReview these recent attempts as data: {json.dumps(recent)}\n'
                    'Write a brief review for the next search. Check one specific assumption '
                    'against the problem rules and conclude whether it holds. Identify a repeated '
                    'structural direction and suggest a different operation or combination to explore. '
                    'If there are no prior attempts, identify one constraint to check and a starting '
                    'direction. Output a short review, without selecting a branch or submitting an answer.')
                review=agent.solve(review_prompt)
                reflection={'reflection_prompt':review_prompt,'reflection_response':asdict(review),
                            'reflection_seed':client.last_seed,'reflection_endpoint':client.last_endpoint}
                query+='\n\nIndependent review for this attempt: '+review.text
            response=agent.solve(query)
            record={**asdict(response),'prompt':query,'request_seed':client.last_seed,'endpoint':client.last_endpoint,**reflection}
            # next_prompt was already called for this attempt; do not select a second time.
            answer=agent.split_response(response.text)[1] if method in ('mcts','two-step') else response.text
            key,valid=score(answer,observations) if answer.strip() else (None,False)
            duplicate=valid and key in valid_keys
            if valid:valid_keys.add(key)
            if valid and key in keys:recovered.add(key)
            is_submission=method not in ('mcts','two-step') or valid
            submitted+=int(is_submission)
            trace={}
            if method!='direct':_,trace=agent.observe(response.text,key,valid)
            record.update(answer=answer,valid=valid,hypothesis_key=key,submitted=is_submission,duplicate_valid=bool(duplicate),**trace)
            result['records'].append(record);save()
            print('ROUND',job_index,len(jobs),domain,cid,method,limit,index+1,
                  'found',len(recovered),'/',len(keys),'finish',response.finish_reason,
                  'effective_cap',response.effective_max_tokens,flush=True)
        responses=[x for r in result['records'] for x in [r,r.get('reflection_response',{})] if x]
        result['metrics']={'recovery':len(recovered)/len(keys),'distinct_valid':len(valid_keys),
            'validity':sum(r['valid'] for r in result['records'])/max(len(result['records']),1),
            'search_rounds':len(result['records']),'api_calls':len(responses),
            'truncations':sum(r.get('finish_reason')=='length' for r in responses),
            'missing_final_answers':sum(bool(r.get('final_answer_missing')) for r in responses),
            'total_tokens':sum(r.get('total_tokens') or 0 for r in responses)}
        result['completed']=True;save()
        print('COMPLETED',job_index,len(jobs),path.name,result['metrics'],flush=True)
    summary=[]
    for p in sorted(OUT.glob('*-cap*.json')):
        d=json.loads(p.read_text());summary.append({k:d[k] for k in ['domain','case_id','method','requested_max_tokens','completed','metrics']})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('TOKEN_LIMIT_PILOT_COMPLETE',flush=True)


if __name__=='__main__':main()
