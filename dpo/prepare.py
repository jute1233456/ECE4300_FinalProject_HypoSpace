"""Freeze stage inputs and family-level DPO holdout before generating any outputs."""
import argparse
from collections import Counter,defaultdict
from hashlib import sha256
import json
from pathlib import Path

from dpo.common import atomic,digest,read_jsonl,write_jsonl
from experiments.generate_medium_training import rows
from server.prepare_quick_validation import family_key
from llm.qwen_stage_protocol import isolate_stage

REFLECTION_INSTRUCTION=('Only review prior attempts. Output a brief Review:, Constraint check:, and '
    'Next direction:. State one evidence-based adjustment for the next search. '
    'Do not submit Expression:, Structure:, Graph:, Branch:, Plan:, or a complete candidate.')
STAGES=('reflection','selection','expansion','candidate')


def assign_splits(states,fraction):
    grouped=defaultdict(Counter)
    for state in states:grouped[state['domain']][state['family_id']]+=1
    held=set()
    for domain,families in grouped.items():
        # With very few families the actual ratio cannot be exactly ten percent.
        if len(families)>1:
            target=sum(families.values())*fraction
            held.add(min(families,key=lambda key:(abs(families[key]-target),digest([4300,key]))))
    for state in states:state['split']='validation' if state['family_id'] in held else 'train'
    return sorted(held)


def prepare(root,output,stages,samples,fraction,max_per_case):
    if output.exists():raise ValueError('Refusing to overwrite frozen generation input')
    source=root/'training/data/ds22-sft-pruned-v1/train.jsonl';canonical=root/'training/sources/ds22-sampled19-canonical.jsonl'
    sft=read_jsonl(source);cases=read_jsonl(canonical);allowed={r['case_id'] for r in sft};questions={}
    for domain in ('causal','3d','boolean'):
        for ds in (('boolean','boolean_extended') if domain=='boolean' else (domain,)):
            data=json.loads((root/f'datasets/balanced/{ds}.json').read_text(encoding='utf-8'))
            for row in rows(data,domain):
                cid=row.get('observation_set_id',row.get('observation_id'))
                if cid in allowed:questions[cid]=dict(case_id=cid,domain=domain,metadata=data['metadata'],question=row,
                                                     family_id=digest(family_key(row,domain,data['metadata'])))
    assert set(questions)==allowed
    states=[];seen=set();origins={};skipped=Counter()
    def add(cid,stage,messages,provenance):
        assert cid in allowed and [m['role'] for m in messages]==['system','user']
        key=digest(dict(stage=stage,messages=messages))
        if key in seen:skipped['identical_input']+=1;return
        seen.add(key);q=questions[cid]
        states.append(dict(state_id=key,case_id=cid,domain=q['domain'],family_id=q['family_id'],stage=stage,
            messages=messages,input_sha256=digest(messages),source=provenance,input_reconstructed=True,
            protocol='MCTS-2.2/isolated_examples_v1',context_file='contexts/'+key+'.json'))
        origins[key]=dict(case_id=cid,stage=stage,original_input=messages,source=provenance,
                          note='No teacher answer or future outcome is used as the generation target.')
    if 'reflection' in stages:
        for case in cases:
            if case['case_id'] not in allowed:continue
            records=list(enumerate(case.get('records',[])))
            if max_per_case and len(records)>max_per_case:
                indices={round(i*(len(records)-1)/(max_per_case-1)) for i in range(max_per_case)} if max_per_case>1 else {0}
                records=[v for v in records if v[0] in indices]
            for index,record in records:
                prompt=record.get('reflection_prompt')
                if not prompt:skipped['missing_reflection_prompt']+=1;continue
                user,system=isolate_stage('reflection',prompt,REFLECTION_INSTRUCTION,examples=True)
                add(case['case_id'],'reflection',[dict(role='system',content=system),dict(role='user',content=user)],
                    dict(file=canonical.relative_to(root).as_posix(),record_index=index,exploration_round=record.get('exploration_round')))
    for row in sft:
        if row['stage'] in stages:
            add(row['case_id'],row['stage'],row['messages'][:-1],dict(file=source.relative_to(root).as_posix(),sample_id=row['sample_id'],**row['source']))
    if not states:raise ValueError('No eligible inputs')
    holdout=assign_splits(states,fraction);states.sort(key=lambda s:(s['domain'],s['case_id'],s['stage'],s['state_id']))
    output.mkdir(parents=True)
    for state in states:atomic(output/state['context_file'],origins[state['state_id']])
    for cid,question in questions.items():atomic(output/'questions'/(cid+'.json'),question)
    write_jsonl(output/'states.jsonl',states)
    manifest=dict(version='stage-dpo-generation-v1',model='sft-pruned-v1-ep1',checkpoint_sha256='8fa112bf26333cd24ea3c2c3759cd5584453a2cce83e75ba8781f569e4bac63e',
        stages=stages,samples_per_state=samples,states=len(states),planned_responses=len(states)*samples,source_cases=len(allowed),
        source_sha256={p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in [source,canonical]},
        split_counts=dict(Counter(s['split'] for s in states)),domain_counts=dict(Counter(s['domain'] for s in states)),
        heldout_families=holdout,requested_validation_fraction=fraction,actual_validation_fraction=sum(s['split']=='validation' for s in states)/len(states),
        split_note='DPO-stage holdout by family; these questions were already seen during SFT. Not an unseen-question generalization test.',
        skipped=dict(skipped),scoring='NONE; raw outputs for later independent CC review',states_sha256=sha256((output/'states.jsonl').read_bytes()).hexdigest())
    atomic(output/'manifest.json',manifest);return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stages',nargs='+',choices=STAGES,default=['reflection']);p.add_argument('--samples-per-state',type=int,default=4)
    p.add_argument('--validation-fraction',type=float,default=.1);p.add_argument('--max-per-case',type=int,default=0);a=p.parse_args()
    if not 0<a.validation_fraction<.5 or a.samples_per_state<2 or a.max_per_case<0:p.error('Invalid generation settings')
    print(json.dumps(prepare(a.root,a.output,a.stages,a.samples_per_state,a.validation_fraction,a.max_per_case),indent=2))
