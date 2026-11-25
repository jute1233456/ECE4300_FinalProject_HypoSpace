"""Two-repeat, paired-seed pilot; independent of the ongoing formal evaluator."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
from statistics import mean
import zlib

from agents.search import MemoryAgent
from agents.feedback_memory import FeedbackMemoryAgent
from llm import OpenAICompatibleClient
from .direct_evaluation import ROOT,load_domain
from .candidate_feedback import describe_candidate

CASES={'boolean':['n1_inputs_00_outputs_0','n1_inputs_00_outputs_1','n2_inputs_00_01_outputs_00'],
       '3d':['g3_h3_b3_006','g3_h3_b3_010'],'causal':['n6_hard_016']}
CALLS=12
REPEATS=2
OUT=ROOT/'results/feedback-memory-pilot-v1'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prompt-style',choices=('formal','compact'),default='formal')
    parser.add_argument('--output-dir',type=Path,help='Separate results for a controlled rerun')
    args=parser.parse_args()
    output_dir=OUT if args.prompt_style=='formal' else ROOT/'results/feedback-memory-pilot-compact-v1'
    if args.output_dir:output_dir=args.output_dir
    output_dir.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for domain,ids in CASES.items():
        if args.prompt_style=='compact' and domain!='boolean':continue
        benchmark,prepare,score=load_domain(domain,ROOT/f'datasets/balanced/{domain}.json')
        previous=ROOT/f'results/server/memory'
        for case in benchmark.all_observation_sets:
            cid=case.get('observation_set_id') or case['observation_id']
            if cid not in ids:continue
            observations,keys=prepare(case)
            # Reuse the exact formal question prompt, without any earlier answers.
            prompt=json.loads((previous/f'{domain}-{cid}.json').read_text())['cases'][0]['prompt']
            if args.prompt_style=='compact':
                prompt=(f'Generate ONE Boolean expression using only variables {benchmark.variables} '
                        f'and operators {sorted(benchmark.operators)}. Maximum nesting depth: {benchmark.max_depth}. '
                        'No Boolean constants. AND/OR are infix; NOT is unary; NOR uses function notation.\n'
                        'Required input-output observations:\n'
                        + '\n'.join(json.dumps({'inputs':obs.inputs,'output':obs.output}) for obs in observations)
                        + '\nYour expression MUST match every observed output when evaluated on its inputs. '
                        'For structural diversity, operand order, repeated AND/OR inputs and associative '
                        'regrouping count as the same mechanism. Distributivity, absorption, De Morgan '
                        'transformations and double-negation removal do not identify mechanisms. '
                        'NOR(a,b) and NOT (a OR b) represent the same NOR gate for depth and scoring. '
                        'Return exactly one line beginning Expression:, with uppercase operators. No explanation.')
            for repeat in range(REPEATS):
                for method in ('raw-memory','feedback-memory'):
                    jobs.append((domain,cid,benchmark,observations,keys,score,prompt,repeat,method))

    def run(job):
        domain,cid,benchmark,observations,keys,score,prompt,repeat,method=job
        path=output_dir/f'{method}-{domain}-{cid}-r{repeat}.json'
        result=json.loads(path.read_text()) if path.exists() else {'method':method,'domain':domain,
            'case_id':cid,'repeat':repeat,'prompt':prompt,'reference_count':len(keys),
            'calls':CALLS,'max_tokens':4096,'completed':False,'records':[],
            'started_utc':datetime.now(timezone.utc).isoformat(),
            'prompt_style':args.prompt_style,
            'generation_options':{'temperature':0.7,'top_p':0.8,'top_k':20,'enable_thinking':False}}
        if result['completed']:return result
        client=OpenAICompatibleClient(api_key='local',model='Qwen3-8B',base_url='http://127.0.0.1:8002/v1',
            timeout=600,generation_options={'temperature':0.7,'top_p':0.8,'top_k':20,'chat_template_kwargs':{'enable_thinking':False}})
        agent=MemoryAgent(client) if method=='raw-memory' else FeedbackMemoryAgent(client)
        recovered=set();seen_valid=set();unique=set()
        def observe(record):
            key,valid=score(record['text'],observations)
            candidate,issues=describe_candidate(domain,benchmark,observations,record['text'],key,valid)
            if method=='feedback-memory':agent.observe(record['text'],key,valid,candidate=candidate,issues=issues)
            else:agent.observe(record['text'],key,valid)
            duplicate=valid and key in seen_valid
            if valid:seen_valid.add(key)
            if key is not None:unique.add(key)
            if valid and key in keys:recovered.add(key)
            return {'valid':valid,'hypothesis_key':key,'candidate':candidate,'violations':issues,'valid_duplicate':bool(duplicate)}
        for record in result['records']:observe(record)
        def save():
            temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(path)
        seed=zlib.crc32(f'{domain}/{cid}/{repeat}'.encode())
        for index in range(len(result['records']),CALLS):
            query=agent.next_prompt(prompt)
            client.generation_options['seed']=(seed+index)%2147483647
            response=client.generate(query,max_tokens=4096)
            record={**asdict(response),'prompt':query,'seed':client.generation_options['seed']}
            record.update(observe(record));result['records'].append(record);save()
        result['metrics']={'recovery':len(recovered)/len(keys),
            'validity':sum(r['valid'] for r in result['records'])/CALLS,
            'valid_duplicate_rate':sum(r['valid_duplicate'] for r in result['records'])/CALLS,
            'unique_valid':len(seen_valid),'unique_reference':len(recovered)}
        result['completed']=True;save()
        return result

    results=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(run,job) for job in jobs]
        for future in as_completed(futures):
            result=future.result();results.append(result)
            print(len(results),'/',len(jobs),result['method'],result['domain'],result['case_id'],result['repeat'],result['metrics'],flush=True)
    summary={method:{k:mean(r['metrics'][k] for r in results if r['method']==method)
                    for k in ('recovery','validity','valid_duplicate_rate','unique_valid')}
             for method in ('raw-memory','feedback-memory')}
    summary['truncated_responses']=sum(r['finish_reason']=='length' for data in results for r in data['records'])
    summary['completed_runs']=len(results)
    (output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
