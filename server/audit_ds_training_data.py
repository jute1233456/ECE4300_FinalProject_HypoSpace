"""Read-only DS inventory, offline candidate verification and leakage-aware shortlist.

Writes a new audit folder only. Never edits source experiments, calls an LLM,
starts training, or treats unverified search rationale as an SFT label.
"""
import argparse
from collections import Counter,defaultdict
from contextlib import redirect_stdout
from datetime import datetime,timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import re


def digest(value):
    return sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def case_rows(data):
    if 'observation_sets' in data:return data['observation_sets']
    if 'datasets' in data:return data['datasets']
    return [r for group in data.get('datasets_by_n_observations',{}).values() for r in group]


def identity(domain,row,meta):
    constraints={k:meta[k] for k in ('nodes','variables','operators','max_depth','grid_size','max_height','max_blocks','max_edges','mechanistic_opts') if k in meta}
    for k in ('nodes','variables','operators'):
        if k in constraints:constraints[k]=sorted(constraints[k])
    if domain=='3d':observations=row['observation']
    else:observations=sorted([{k:v for k,v in x.items() if k!='string'} for x in row['observations']],key=lambda x:json.dumps(x,sort_keys=True))
    return digest(dict(domain=domain,constraints=constraints,observations=observations))


def variant(data):
    if data.get('mcts_version')!='2.2':return 'other_DS'
    c=data.get('tree_config',{})
    if c.get('stage22_revision'):return '2.2_revision'
    if c.get('partition_reflection'):return '2.2_partition'
    return '2.2_original'


def truncated(record):
    responses=[record,record.get('reflection_response',{})]
    for request in record.get('api_requests',[]):
        responses.append(request.get('stage_response',{}))
        responses.extend((request.get('response',{}).get('choices') or []))
    return any(r.get('finish_reason')=='length' or r.get('final_answer_missing') for r in responses)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--snapshot-dir',type=Path,help='Re-audit a frozen source inventory instead of live runs')
    args=parser.parse_args()
    root=args.root.resolve();stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output=root/f'local-backups/ds-training-audit-{stamp}';output.mkdir(parents=True)
    snapshots=output/'snapshots';snapshots.mkdir()
    import sys
    sys.path.insert(0,str(root))
    from experiments.generate_medium_training import family
    from experiments.direct_evaluation import load_domain
    datasets={};data_paths={}
    for path in (root/'datasets').rglob('*.json'):
        raw=path.read_bytes()
        try:data=json.loads(raw)
        except ValueError:continue
        if isinstance(data,dict) and 'metadata' in data:
            h=sha256(raw).hexdigest();datasets[h]=data;data_paths[h]=path
    eval_ids=defaultdict(set);eval_families=defaultdict(set);eval_relaxed=defaultdict(set)
    for domain in ('causal','3d','boolean'):
        data=json.loads((root/f'datasets/medium_sampled/{domain}.json').read_bytes());meta=data['metadata']
        for row in case_rows(data):
            cid=row.get('observation_set_id')or row['observation_id'];eval_ids[domain].add(cid)
            fam=family(row,domain,meta);eval_families[domain].add(fam)
            if domain=='boolean':eval_relaxed[domain].add((len(meta['variables']),fam[-1]))
    scorer_cache={};problem_cache={};runs=defaultdict(Counter);source_index=[];skips=Counter();turns={}
    if args.snapshot_dir:
        old=json.loads((args.snapshot_dir/'sources.json').read_text(encoding='utf-8'))
        inputs=[(root/x['path'],args.snapshot_dir/'snapshots'/x['snapshot']) for x in old]
    else:inputs=[(p,p) for p in sorted((root/'results').rglob('*.json'))]
    for path,read_path in inputs:
        if '.tree.' in path.name or path.name in ('manifest.json','state.json','status.json','merge-manifest.json'):continue
        try:
            raw=read_path.read_bytes();data=json.loads(raw)
        except (OSError,ValueError):skips['unreadable_or_changing_json']+=1;continue
        if not isinstance(data,dict) or 'requested_model' not in data:continue
        if 'deepseek' not in str(data.get('requested_model','')).lower():
            skips['non_confirmed_DS_result']+=1;continue
        source=str(path.relative_to(root));source_hash=sha256(raw).hexdigest()
        target=snapshots/f'{source_hash}.json'
        if not target.exists():target.write_bytes(raw)
        group=source.split('/')[1] if '/' in source else path.parent.name
        # pathlib on Windows emits backslashes; normalize report paths explicitly.
        source=path.relative_to(root).as_posix();group=path.relative_to(root/'results').parts[0]
        v=variant(data);run=runs[(group,v)];run['files']+=1
        cases=list(data.get('cases',[]))
        current=data.get('interrupted_case')
        if current and not any((x.get('domain'),x.get('case_id'))==(current.get('domain'),current.get('case_id')) for x in cases):cases.append(current)
        source_index.append(dict(path=source,sha256=source_hash,snapshot=target.name,variant=v,completed=data.get('completed',False),cases=len(cases)))
        for case in cases:
            domain=case.get('domain');cid=case.get('case_id')
            if domain not in ('causal','3d','boolean'):continue
            entry_key='boolean_extended' if str(cid).startswith('extended_') else domain
            md=data.get('datasets',{}).get(entry_key) or data.get('datasets',{}).get(domain) or {}
            dataset_sha=md.get('sha256');cache_key=(domain,dataset_sha,cid)
            if cache_key not in problem_cache:
                ds=datasets.get(dataset_sha)
                row=next((x for x in case_rows(ds) if (x.get('observation_set_id')or x.get('observation_id'))==cid),None) if ds else None
                if not row:problem_cache[cache_key]=None
                else:
                    meta=ds['metadata'];fingerprint=identity(domain,row,meta);leak=[]
                    if cid in eval_ids[domain]:leak.append('shared_case_id_with_evaluation')
                    fam=None
                    if domain!='causal' or len(meta['nodes'])==4:fam=family(row,domain,meta)
                    if fam in eval_families[domain]:leak.append('equivalent_evaluation_family')
                    if domain=='boolean' and fam and (len(meta['variables']),fam[-1]) in eval_relaxed[domain]:leak.append('same_observations_under_variable_swap_even_if_operators_differ')
                    if (domain,dataset_sha) not in scorer_cache:
                        with redirect_stdout(io.StringIO()):scorer_cache[domain,dataset_sha]=load_domain(domain,data_paths[dataset_sha])
                    benchmark,prepare,score=scorer_cache[domain,dataset_sha]
                    obs,keys=prepare(row)
                    problem_cache[cache_key]=dict(fingerprint=fingerprint,leak=leak,obs=obs,keys=keys,score=score)
            problem=problem_cache[cache_key]
            run['cases']+=1
            if any(x is case for x in data.get('cases',[])) and case.get('stop_reason')!='request_errors':run['completed_cases']+=1
            if data.get('mcts_version')=='2.2':run['stage22_cases']+=1
            occurrences=Counter()
            for i,record in enumerate(case.get('records',[]),1):
                run['rounds']+=1
                if record.get('valid'):run['logged_valid']+=1
                stage_outputs={a['stage']:a['stage_response'].get('text','') for a in record.get('api_requests',[]) if a.get('stage') and a.get('stage_response')}
                tree=next((x.get('result') for x in record.get('tool_trace',[]) if x.get('name')=='read_tree'),None)
                compact_tree={k:val for k,val in (tree or {}).items() if k not in ('file_sha256','case_id')}
                answer=record.get('answer') or (record.get('text','') if data.get('method') in ('direct','memory','memory-structured') else '')
                valid=False;key=None;verification='dataset_unresolved'
                if problem:
                    try:
                        key,ok=problem['score'](answer,problem['obs']);valid=bool(ok and key in problem['keys']);verification='verified_reference_hit' if valid else 'rejected_by_local_verifier'
                    except Exception as error:verification='verification_exception_'+type(error).__name__
                if valid:run['verified_valid']+=1
                if record.get('protocol_error'):run['protocol_errors']+=1
                if truncated(record):run['truncated_rounds']+=1
                if problem and problem['leak']:run['leakage_rounds']+=1
                fingerprint=problem['fingerprint'] if problem else digest([domain,dataset_sha,cid])
                signature=digest([fingerprint,record.get('text'),stage_outputs,record.get('reflection_response',{}).get('text'),compact_tree,
                    record.get('prompt'),record.get('error'),record.get('error_stage'),record.get('protocol_error'),
                    [x.get('seconds') for x in record.get('api_requests',[])]])
                occurrences[signature]+=1
                tid=digest([signature,occurrences[signature]])
                provenance=dict(source=source,snapshot_sha256=source_hash,round=i,variant=v,case_completed=bool(case.get('metrics')))
                reasons=[]
                if not valid:reasons.append(verification)
                if record.get('protocol_error'):reasons.append('protocol_error')
                if record.get('error'):reasons.append('request_error')
                if truncated(record):reasons.append('incomplete_or_truncated_round')
                if problem and problem['leak']:reasons.append('evaluation_overlap')
                stage_inputs=all(a.get('request',{}).get('prompt') for a in record.get('api_requests',[]) if a.get('stage') in ('selection','expansion','candidate'))
                has_stages=all(stage_outputs.get(s) for s in ('selection','expansion','candidate'))
                branch=re.match(r'^\s*Branch:\s*(\d+)',stage_outputs.get('selection',''))
                allowed={n.get('branch') for n in (tree or {}).get('nodes',[]) if n.get('branch') is not None}
                branch_ok=bool(branch and int(branch[1]) in allowed)
                whole_candidate=valid and not reasons and has_stages and branch_ok and bool(record.get('valid'))
                tier='primary_candidate_with_provisional_decisions' if whole_candidate else 'candidate_only_secondary' if valid and not reasons else 'excluded'
                item=dict(id=tid,problem_fingerprint=fingerprint,domain=domain,case_id=cid,dataset_sha256=dataset_sha,
                    variant=v,teacher_model=data['requested_model'],verification=verification,verified_valid=valid,
                    original_valid=bool(record.get('valid')),hypothesis_key=key,tier=tier,exclusion_reasons=reasons,
                    leakage_reasons=problem['leak'] if problem else [],answer=answer,stage_targets=stage_outputs,
                    reflection_target=record.get('reflection_response',{}).get('text'),tree_available=bool(tree),
                    exact_stage_inputs_logged=bool(has_stages and stage_inputs),plan_semantics_verified=False,
                    branch_target_verified=branch_ok,logged_reward=record.get('reward'),
                    decisions_require_review=True,tokenizer_ready=False,sources=[provenance])
                if tid in turns:
                    existing=turns[tid];all_sources=existing['sources']+[provenance]
                    ranks={'2.2_revision':0,'2.2_partition':1,'2.2_original':2,'other_DS':3}
                    if ranks.get(v,9)<ranks.get(existing['variant'],9):item['sources']=all_sources;turns[tid]=item
                    else:existing['sources']=all_sources
                else:turns[tid]=item
    unique=list(turns.values());counts=Counter(x['tier'] for x in unique);by_variant={}
    for v in sorted({x['variant'] for x in unique}):
        subset=[x for x in unique if x['variant']==v];safe=[x for x in subset if x['tier']!='excluded']
        by_variant[v]=dict(unique_rounds=len(subset),verified_valid=sum(x['verified_valid'] for x in subset),
            primary=sum(x['tier']=='primary_candidate_with_provisional_decisions' for x in subset),
            secondary=sum(x['tier']=='candidate_only_secondary' for x in subset),
            safe_problems=len({x['problem_fingerprint'] for x in safe}),
            unique_safe_answers=len({(x['problem_fingerprint'],digest(x['hypothesis_key'])) for x in safe}))
    candidates=[x for x in unique if x['tier']!='excluded']
    priorities={'2.2_revision':0,'2.2_partition':1,'2.2_original':2,'other_DS':3}
    candidates.sort(key=lambda x:(priorities.get(x['variant'],9),x['domain'],x['case_id'],x['id']))
    seen_answers=set();answer_shortlist=[]
    for item in candidates:
        k=(item['problem_fingerprint'],digest(item['hypothesis_key']))
        if k not in seen_answers:seen_answers.add(k);answer_shortlist.append(item)
    exclusion_cases=defaultdict(set)
    for x in unique:
        for reason in x['leakage_reasons']:exclusion_cases[reason].add((x['domain'],x['case_id']))
    summary=dict(snapshot_utc=stamp,source_files=len(source_index),raw_rounds=sum(v['rounds'] for v in runs.values()),
        unique_rounds=len(unique),copied_or_exact_duplicate_rounds=sum(v['rounds'] for v in runs.values())-len(unique),
        tier_counts=dict(counts),variants=by_variant,shortlisted_unique_answers=len(answer_shortlist),
        shortlisted_problems=len({x['problem_fingerprint'] for x in answer_shortlist}),
        exclusion_cases={k:sorted(v) for k,v in exclusion_cases.items()},skips=dict(skips),
        source_runs=[dict(run=k[0],variant=k[1],**v) for k,v in runs.items()],
        caveats=['Snapshot of live files; no DS process or source result was modified.',
          'Valid candidate is not proof that branch rationale, plan or reflection is correct.',
          'No stage labels are tokenizer-ready; missing exact inputs require reconstruction.',
          'Evaluation overlaps are conservatively quarantined, including same observations with different operator sets.',
          'Answer dedup is per problem; varied search states remain separate provisional trajectories.'])
    stage22=[x for x in candidates if x['variant'].startswith('2.2')]
    stage22_answers=[];seen=set()
    for item in stage22:
        key=(item['problem_fingerprint'],digest(item['hypothesis_key']))
        if key not in seen:seen.add(key);stage22_answers.append(item)
    summary['preferred_stage22']=dict(eligible_turns=len(stage22),unique_answers=len(stage22_answers),
        problems=len({x['problem_fingerprint'] for x in stage22}),
        exact_stage_inputs_logged=sum(x['exact_stage_inputs_logged'] for x in stage22),
        raw_stage_output_targets=sum(len(x['stage_targets']) for x in stage22),
        reflections=sum(bool(x['reflection_target']) for x in stage22),
        positive_reward_turns=sum((x['logged_reward']or 0)>0 for x in stage22),
        duplicate_reward_turns=sum((x['logged_reward']or 0)<0 for x in stage22),
        excluded_question_ids=sorted({(x['domain'],x['case_id']) for x in unique if x['variant'].startswith('2.2') and x['leakage_reasons']}))
    summary['input_snapshot']=str(args.snapshot_dir) if args.snapshot_dir else None
    for name,value in [('summary',summary),('sources',source_index)]:
        (output/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,items in [('all_turns',unique),('candidate_shortlist',answer_shortlist),
                       ('stage22_verified_turns',stage22),('stage22_unique_answers',stage22_answers)]:
        with (output/f'{name}.jsonl').open('w',encoding='utf-8') as f:
            for x in items:f.write(json.dumps(x,ensure_ascii=False,default=str)+'\n')
    print('AUDIT_OUTPUT',output)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('source_runs','caveats')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
