"""Subtract actual SFT training families and deduplicate medium27, without new cases."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from experiments.generate_medium_training import rows
from server.prepare_quick_validation import family_key

DOMAINS=('causal','3d','boolean')
REFS={'causal':'ground_truth_graphs','3d':'ground_truth_structures','boolean':'ground_truth_expressions'}


def cid(row):return row.get('observation_set_id',row.get('observation_id'))
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def fingerprint(key):return sha256(repr(key).encode()).hexdigest()


def build(root, train_path, output):
    if output.exists():raise ValueError('Refusing to overwrite frozen dataset')
    samples=[json.loads(line) for line in train_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    training_ids={s['case_id'] for s in samples};located=set();forbidden=defaultdict(list);hashes={}
    for domain in DOMAINS:
        for dataset in (('boolean','boolean_extended') if domain=='boolean' else (domain,)):
            path=root/f'datasets/balanced/{dataset}.json';data=read(path)
            hashes[path.relative_to(root).as_posix()]=sha256(path.read_bytes()).hexdigest()
            for row in rows(data,domain):
                if cid(row) in training_ids:
                    located.add(cid(row))
                    forbidden[family_key(row,domain,data['metadata'])].append(dict(dataset=dataset,case_id=cid(row)))
    assert located==training_ids,('Training cases not found',training_ids-located)
    original=read(root/'datasets/medium_sampled.json')
    expected={(d,j['case_id']):j['references'] for d in DOMAINS for j in original[d]}
    jobs=[];excluded=[];datasets={};summary={}
    for domain in DOMAINS:
        path=root/f'datasets/medium_sampled/{domain}.json';data=read(path)
        hashes[path.relative_to(root).as_posix()]=sha256(path.read_bytes()).hexdigest()
        pool=sorted(rows(data,domain),key=cid);retained=[];seen={};counts=Counter()
        for row in pool:
            name=cid(row);count=len(row[REFS[domain]]);key=family_key(row,domain,data['metadata'])
            assert expected[(domain,name)]==count
            info=dict(dataset=domain,case_id=name,references=count,family_sha256=fingerprint(key))
            if key in forbidden:
                excluded.append(dict(**info,reason='training_equivalent_family',training_matches=forbidden[key]));counts['training_overlap']+=1
            elif key in seen:
                excluded.append(dict(**info,reason='duplicate_test_family',retained_case=seen[key]));counts['internal_duplicates']+=1
            else:
                seen[key]=name;retained.append(row);jobs.append(info)
        datasets[domain]=dict(metadata=deepcopy(data['metadata']))
        meta=datasets[domain]['metadata'];meta.update(split='validation',dedup_version='medium-dedup-v1',total_observation_sets=len(retained))
        if domain=='3d':
            datasets[domain]['observation_sets']=retained;meta['n_observation_sets']=len(retained)
        elif 'datasets' in data:datasets[domain]['datasets']=retained
        else:
            grouped={}
            for row in retained:grouped.setdefault(str(row['n_observations']),[]).append(row)
            datasets[domain]['datasets_by_n_observations']=grouped
        summary[domain]=dict(original_cases=len(pool),training_overlap=counts['training_overlap'],
            internal_duplicates=counts['internal_duplicates'],retained_cases=len(retained),references=sum(len(r[REFS[domain]]) for r in retained))
    assert len(expected)==27 and sum(s['original_cases'] for s in summary.values())==27
    assert len(jobs)+len(excluded)==27
    assert len({j['family_sha256'] for j in jobs})==len(jobs)
    result=dict(version='medium-dedup-v1',source_pool='medium_sampled_27',training_file=train_path.relative_to(root).as_posix(),
        training_sha256=sha256(train_path.read_bytes()).hexdigest(),training_samples=len(samples),training_cases=len(training_ids),
        rules=dict(causal='node permutations',boolean='variable permutations; conservative observation match across operator sets; same variable count and depth',
                   spatial='D4 rotations/reflections and translation; same maximum height; conservative across grid sizes',
                   representative='lexicographically first case ID; training overlap removed before internal dedup'),
        summary=summary,jobs=jobs,excluded=excluded,total_cases=len(jobs),total_references=sum(j['references'] for j in jobs),source_sha256=hashes)
    output.mkdir(parents=True)
    for domain,data in datasets.items():(output/f'{domain}.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    (output/'manifest.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--train-file',type=Path,default=Path('training/data/ds22-sft-pruned-v1/train.jsonl'))
    parser.add_argument('--output',type=Path,default=Path('datasets/medium_dedup_v1'));args=parser.parse_args()
    result=build(args.root,args.root/args.train_file,args.root/args.output)
    print(json.dumps({k:result[k] for k in ['summary','jobs','excluded','total_cases','total_references']},indent=2))


if __name__=='__main__':main()
