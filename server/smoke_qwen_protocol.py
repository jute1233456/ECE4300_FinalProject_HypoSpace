"""Six single-round end-to-end diagnostics, not a restart of the nineteen-case run."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,json,os,subprocess,sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variants',default='isolated,examples')
    parser.add_argument('--only-case');parser.add_argument('--tag',default='e2e')
    options=parser.parse_args()
    release=Path.cwd();root=release.parents[1]
    output=root/'logs/qwen-protocol-diagnosis-20261126'/options.tag;output.mkdir(parents=True,exist_ok=True)
    jobs=[('boolean','n4_inputs_00_01_10_11_outputs_0110'),
          ('boolean','n4_inputs_00_01_10_11_outputs_1110'),('3d','g3_h3_b2_001')]
    if options.only_case:jobs=[j for j in jobs if j[1]==options.only_case]
    def run(variant,domain,case):
        target=output/f'{variant}-{domain}-{case}.json'
        args=[sys.executable,'-u','-m','experiments.test_hard_balanced','--model','Qwen3-8B',
            '--enable-thinking','--base-url','http://127.0.0.1:8004/v1','--max-tokens','32768',
            '--method','mcts-tree','--tree-policy','llm','--duplicate-reward','-1','--exploration-list','off',
            '--two-step-reflection','--unbounded-tree','--tree-tools','--mcts-version','2.2',
            '--stage22-revision','--dataset',domain,'--case-id',case,'--search-cap','1',
            '--output',str(target),'--pause-file',str(output/'PAUSE')]
        env=dict(os.environ,ECE_PROTOCOL_EXAMPLES='1' if variant=='examples' else '0',
                 OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        with target.with_suffix('.log').open('w') as log:
            code=subprocess.run(args,cwd=release,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        row=dict(variant=variant,domain=domain,case_id=case,exit_code=code)
        if not code:
            d=json.loads(target.read_text());records=d['cases'][0]['records']
            row.update(protocol_errors=[r.get('protocol_error') for r in records if r.get('protocol_error')],
                valid=sum(bool(r.get('valid')) for r in records),
                stages=[[a.get('stage') for a in r.get('api_requests',[])] for r in records])
        return row
    rows=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(run,v,ds,c) for v in options.variants.split(',') for ds,c in jobs]):
            row=f.result();rows.append(row);print(row,flush=True)
            (output/'summary.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':main()
