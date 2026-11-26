"""Extend medium-dedup-v1 towards 100 references/domain, reusing completed baselines."""
import argparse
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from experiments.generate_medium_training import rows
from server.prepare_quick_validation import family_key
from server.prepare_dedup_medium_validation import cid, REFS, DOMAINS


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def nearest_subset(candidates,target):
    options={0:()}
    for i,item in enumerate(candidates):
        for total,chosen in list(options.items()):
            score=total+item['references'];new=chosen+(i,)
            if score not in options or (len(new),new)<(len(options[score]),options[score]):options[score]=new
    best=min(options,key=lambda total:(abs(total-target),len(options[total]),options[total]))
    return [candidates[i] for i in options[best]]


def build(root,inventory_path,output):
    if output.exists():raise ValueError('Refusing to overwrite frozen mixed suite')
    medium=read(root/'datasets/medium_dedup_v1/manifest.json')
    inventory=read(inventory_path);historical={(i['suite'],i['dataset'],i['case_id']):i for i in inventory}
    train=root/medium['training_file'];assert sha256(train.read_bytes()).hexdigest()==medium['training_sha256']
    training_ids={json.loads(line)['case_id'] for line in train.read_text(encoding='utf-8').splitlines()}
    forbidden=set();hashes={};sources={};selected=[];excluded=[]
    def source(relative):
        if relative not in sources:
            path=root/relative;sources[relative]=read(path);hashes[relative]=sha256(path.read_bytes()).hexdigest()
        return sources[relative]
    for domain in DOMAINS:
        for dataset in (('boolean','boolean_extended') if domain=='boolean' else (domain,)):
            data=source(f'datasets/balanced/{dataset}.json')
            forbidden.update(family_key(r,domain,data['metadata']) for r in rows(data,domain) if cid(r) in training_ids)
    seen=set()
    def candidate(domain,difficulty,row,relative):
        data=source(relative);key=family_key(row,domain,data['metadata']);digest=sha256(repr(key).encode()).hexdigest()
        return dict(dataset=domain,difficulty=difficulty,case_id=cid(row),references=len(row[REFS[domain]]),
                    family_sha256=digest,source=relative,_key=key,_row=row)
    for job in medium['jobs']:
        relative=f"datasets/medium_sampled/{job['dataset']}.json";data=source(relative)
        row=next(r for r in rows(data,job['dataset']) if cid(r)==job['case_id'])
        item=candidate(job['dataset'],'medium',row,relative)
        assert item['_key'] not in forbidden and item['family_sha256'] not in seen
        selected.append(item);seen.add(item['family_sha256'])
    for domain in DOMAINS:
        relative=f'datasets/{domain}.json';data=source(relative);pool=[];local_seen=set()
        for row in sorted(rows(data,domain),key=cid):
            item=candidate(domain,'easy',row,relative);key=item['_key'];digest=item['family_sha256']
            record=historical.get(('easy',domain,cid(row)))
            if key in forbidden or digest in seen or digest in local_seen:
                excluded.append(dict(dataset=domain,case_id=cid(row),source=relative,reason='training_or_test_equivalent'));continue
            if not record or not all(record['methods'].get(m,{}).get('completed') for m in ['direct','memory']):continue
            pool.append(item);local_seen.add(digest)
        target=100-sum(j['references'] for j in selected if j['dataset']==domain)
        extra=nearest_subset(pool,target)
        for item in extra:selected.append(item);seen.add(item['family_sha256'])
    # User explicitly allows additional difficulty in 3D: two unique 27-solution
    # cases bring 39 to 93, closer to 100 than adding a third (120).
    relative='datasets/balanced/3d.json';data=source(relative);pool=[];local_seen=set()
    for row in sorted(rows(data,'3d'),key=cid):
        if len(row['ground_truth_structures'])!=27:continue
        item=candidate('3d','supplemental',row,relative);digest=item['family_sha256']
        if item['_key'] in forbidden or digest in seen or digest in local_seen:continue
        pool.append(item);local_seen.add(digest)
    extra=nearest_subset(pool,100-sum(j['references'] for j in selected if j['dataset']=='3d'))
    for item in extra:selected.append(item);seen.add(item['family_sha256'])
    assert len(seen)==len(selected)
    output.mkdir(parents=True);jobs=[];datasets=defaultdict(list);summary={}
    for item in selected:
        row=item.pop('_row');key=item.pop('_key');assert key not in forbidden
        difficulty=item['difficulty'];domain=item['dataset'];datasets[(difficulty,domain,item['source'])].append(row)
        item['dataset_root']=difficulty
        item['id']=difficulty+'/'+domain+'/'+item['case_id']
        record=historical.get((difficulty,domain,item['case_id']))
        item['baselines']={}
        for method in ['direct','memory']:
            info=record['methods'][method] if record else None
            if info and info['completed']:
                original=source(item['source']);assert info['datasets'][domain]['sha256']==hashes[item['source']]
                assert info['model']=='Qwen3-8B' and info['thinking'] is True and info['max_tokens']==16384
                item['baselines'][method]=dict(status='reused',**info,recovered=round(info['coverage']*item['references']))
            else:item['baselines'][method]=dict(status='pending')
        jobs.append(item)
    for (difficulty,domain,relative),items in datasets.items():
        data=source(relative);meta=deepcopy(data['metadata']);meta.update(total_observation_sets=len(items),split='mixed_validation')
        out=dict(metadata=meta)
        if domain=='3d':out['observation_sets']=items;meta['n_observation_sets']=len(items)
        elif 'datasets' in data:out['datasets']=items
        else:
            grouped={}
            for row in items:grouped.setdefault(str(row['n_observations']),[]).append(row)
            out['datasets_by_n_observations']=grouped
        folder=output/difficulty;folder.mkdir(exist_ok=True)
        (folder/f'{domain}.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    for domain in DOMAINS:
        subset=[j for j in jobs if j['dataset']==domain]
        summary[domain]=dict(cases=len(subset),references=sum(j['references'] for j in subset),
            by_difficulty={diff:dict(cases=sum(j['difficulty']==diff for j in subset),references=sum(j['references'] for j in subset if j['difficulty']==diff)) for diff in ['easy','medium','supplemental'] if any(j['difficulty']==diff for j in subset)})
    manifest=dict(version='mixed100-v1',target_references_per_domain=100,training_sha256=medium['training_sha256'],
        source_sha256=hashes,summary=summary,jobs=jobs,excluded_easy=excluded,total_cases=len(jobs),total_references=sum(j['references'] for j in jobs),
        reuse=dict(complete_pairs=sum(all(j['baselines'][m]['status']=='reused' for m in ['direct','memory']) for j in jobs),
                   missing_method_runs=sum(j['baselines'][m]['status']=='pending' for j in jobs for m in ['direct','memory'])),
        rules=medium['rules'],note='All medium-dedup-v1 cases retained. Easy cases reuse old direct/memory. 3D exceptions authorized; no automatic MCTS evaluation.')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--inventory',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('datasets/mixed100_v1'));args=parser.parse_args()
    m=build(args.root,args.inventory,args.root/args.output)
    print(json.dumps({k:m[k] for k in ['summary','total_cases','total_references','reuse']},indent=2))
