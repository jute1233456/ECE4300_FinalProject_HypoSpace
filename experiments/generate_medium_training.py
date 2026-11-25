"""Select independently checked medium training questions outside evaluation families."""
import argparse
import ast
from collections import Counter
from copy import deepcopy
from hashlib import sha256
from itertools import combinations, permutations, product
import json
from pathlib import Path
from random import Random


def rows(data, domain):
    if domain=='3d':
        return data['observation_sets']
    if 'datasets' in data:
        return data['datasets']
    return [r for group in data['datasets_by_n_observations'].values() for r in group]


def family(row, domain, meta):
    if domain=='causal':
        names=meta['nodes']; signatures=[]
        for order in permutations(names):
            signatures.append(tuple(sorted((order.index(o['perturbed_node']),
                tuple(o['effects'][n] for n in order)) for o in row['observations'])))
        return ('causal',len(names),min(signatures))
    if domain=='boolean':
        names=meta['variables']; signatures=[]
        for order in permutations(names):
            signatures.append(tuple(sorted((tuple(o['inputs'][n] for n in order),o['output'])
                                            for o in row['observations'])))
        return ('boolean',tuple(sorted(meta['operators'])),meta['max_depth'],min(signatures))
    size=meta['grid_size'];coords=[divmod(i,size) for i,v in enumerate(row['observation']) if v=='1']
    signatures=[]
    for swap,flipx,flipy in product((False,True),repeat=3):
        transformed=[(y if swap else x,x if swap else y) for x,y in coords]
        transformed=[((-x if flipx else x),(-y if flipy else y)) for x,y in transformed]
        mx=min(x for x,y in transformed);my=min(y for x,y in transformed)
        signatures.append(tuple(sorted((x-mx,y-my) for x,y in transformed)))
    return ('3d',meta['max_height'],min(signatures))


def graph_universe(nodes):
    pairs=list(combinations(nodes,2));universe=[]
    for directions in product((0,1,2),repeat=len(pairs)):
        edges=[pair if d==1 else pair[::-1] for pair,d in zip(pairs,directions) if d]
        reach={n:set() for n in nodes}
        for a,b in edges:reach[a].add(b)
        for k in nodes:
            for n in nodes:
                if k in reach[n]:reach[n].update(reach[k])
        if any(n in reach[n] for n in nodes):continue
        universe.append((frozenset(edges),reach))
    return universe


def verify(data, domain):
    meta=data['metadata']
    if domain=='causal':
        universe=graph_universe(meta['nodes'])
        assert len(universe)==543
        for r in rows(data,domain):
            expected={edges for edges,reach in universe if all(
                all(int(n in reach[o['perturbed_node']])==v for n,v in o['effects'].items())
                for o in r['observations'])}
            actual={frozenset(tuple(e) for e in g['edges']) for g in r['ground_truth_graphs']}
            assert expected==actual, r['observation_set_id']
    elif domain=='boolean':
        from experiments.direct_evaluation import load_domain,ROOT
        load_domain('boolean',ROOT/'datasets/medium/boolean.json')
        from boolean_dataset import BooleanDiscoveryGame
        expressions=BooleanDiscoveryGame.generate_all_expressions(meta['variables'],meta['operators'],
            meta['max_depth'],meta['mechanistic_opts'])
        for r in rows(data,domain):
            expected={e.mechanistic_key(**meta['mechanistic_opts']) for e in expressions if all(
                e.evaluate(o['inputs'])==o['output'] for o in r['observations'])}
            actual={ast.literal_eval(e['mechanistic_key']) for e in r['ground_truth_expressions']}
            assert expected==actual,r['observation_set_id']
    else:
        size=meta['grid_size'];height=meta['max_height']
        for r in rows(data,domain):
            positions=[i for i,v in enumerate(r['observation']) if v=='1']
            expected=set()
            for heights in product(range(1,height+1),repeat=len(positions)):
                layers=[]
                for z in range(max(heights)):
                    layer=['0']*(size*size)
                    for i,h in zip(positions,heights):
                        if h>z:layer[i]='1'
                    layers.append(''.join(layer))
                expected.add(tuple(layers))
            assert expected=={tuple(s['layers']) for s in r['ground_truth_structures']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',action='store_true')
    args=parser.parse_args();root=args.project_root
    if args.output.exists() and not args.plan:
        raise SystemExit('Training dataset already exists; refusing to overwrite')
    rng=Random(430021);datasets={};report={}
    evaluation={d:json.loads((root/f'datasets/medium_sampled/{d}.json').read_text())
                for d in ('causal','3d','boolean')}
    for domain in ('causal','3d','boolean'):
        source=root/f'datasets/medium/{domain}.json';data=json.loads(source.read_text())
        forbidden={family(r,domain,evaluation[domain]['metadata']) for r in rows(evaluation[domain],domain)}
        pool=rows(data,domain)
        if domain=='3d':
            # New spatial families: same two occupied columns and 3^2=9 references.
            from experiments.direct_evaluation import load_domain,ROOT
            load_domain('3d',ROOT/'datasets/medium_sampled/3d.json')
            import sys
            sys.path.insert(0,str(ROOT/'HypoSpace/_HypoSpace-main/3d'))
            from generate_3d_dataset_complete import enumerate_all_structures_by_heights
            import numpy as np
            data=deepcopy(data);data['metadata']['grid_size']=4
            pool=[]
            for i,occupied in enumerate(combinations(range(16),2),1):
                top=np.zeros((4,4),dtype=int)
                for pos in occupied:top[divmod(pos,4)]=1
                pool.append(dict(observation_id=f'train_g4_h3_b2_{i:03d}',
                    observation=''.join(map(str,top.flatten())),
                    ground_truth_structures=[s.to_dict() for s in enumerate_all_structures_by_heights(top,3)]))
        rng.shuffle(pool);selected=[];seen=set();excluded=0
        for row in pool:
            count=len(row['ground_truth_graphs'] if domain=='causal' else row['ground_truth_structures']
                      if domain=='3d' else row['ground_truth_expressions'])
            if not 4<=count<=(48 if domain=='causal' else 16):continue
            key=family(row,domain,data['metadata'])
            if key in forbidden or key in seen:
                excluded+=1;continue
            seen.add(key);row=deepcopy(row)
            keyname='observation_id' if domain=='3d' else 'observation_set_id'
            if not row[keyname].startswith('train_'):row[keyname]='train_'+row[keyname]
            selected.append(row)
            if len(selected)>=12:break
        assert selected,f'No independent training families for {domain}'
        out=dict(metadata=deepcopy(data['metadata']))
        out['metadata'].update(split='train',seed=430021,reference_validation='exhaustive',
            excluded_eval_sha256=sha256((root/f'datasets/medium_sampled/{domain}.json').read_bytes()).hexdigest())
        if domain=='3d':
            out['observation_sets']=selected;out['metadata']['n_observation_sets']=len(selected)
        else:
            groups={}
            for row in selected:groups.setdefault(str(row['n_observations']),[]).append(row)
            out['datasets_by_n_observations']=groups;out['metadata']['total_observation_sets']=len(selected)
            if domain=='causal':out['metadata']['max_edges']=6
        verify(out,domain)
        datasets[domain]=out
        report[domain]=dict(cases=len(selected),excluded_equivalent=excluded,
            references=sum(len(r['ground_truth_graphs'] if domain=='causal' else r['ground_truth_structures']
                if domain=='3d' else r['ground_truth_expressions']) for r in selected))
    print(json.dumps(report,indent=2),flush=True)
    if args.plan:return
    args.output.mkdir(parents=True)
    for d,data in datasets.items():
        (args.output/f'{d}.json').write_text(json.dumps(data,indent=2))
    (args.output/'generation_manifest.json').write_text(json.dumps(dict(seed=430021,report=report,
        role='training_only',dedup='causal node permutations; Boolean variable permutations; 3D D4 and translation',
        note='3D uses grid 4 with two occupied columns, max height 3, nine solutions per question'),indent=2))


if __name__=='__main__':main()
