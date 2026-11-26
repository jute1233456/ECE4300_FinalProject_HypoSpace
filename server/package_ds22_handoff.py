"""Freeze ONLY completed cases in the current DS-2.2 revision and build an offline handoff."""
import ast
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT=Path(__file__).resolve().parents[1]


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def functions(path,names):
    source=path.read_text(encoding='utf-8');tree=ast.parse(source)
    return '\n\n'.join(ast.get_source_segment(source,node) for node in tree.body
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in names)


PREPARE=r'''"""Offline verification only. No LLM calls, training, or source-result modifications."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
from hashlib import sha256
import io,json
from pathlib import Path
from helpers import case_rows,digest,family,identity,truncated
from offline_verifier import load_domain

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args()
    root=Path(__file__).resolve().parents[1];out=args.output or root/'review-output';out.mkdir(parents=True,exist_ok=True)
    scope=json.loads((root/'scope.json').read_text(encoding='utf-8'))
    registry=json.loads((root/'reference/dataset_registry.json').read_text(encoding='utf-8'))
    eval_ids={};eval_families={};eval_obs=set()
    for ds in ('causal','3d','boolean'):
        data=json.loads((root/f'reference/evaluation_medium27/{ds}.json').read_text(encoding='utf-8'))
        eval_ids[ds]=set();eval_families[ds]=set()
        for row in case_rows(data):
            eval_ids[ds].add(row.get('observation_set_id') or row['observation_id'])
            f=family(row,ds,data['metadata']);eval_families[ds].add(f)
            if ds=='boolean':eval_obs.add((len(data['metadata']['variables']),f[-1]))
    questions=[];loaded={};raw_cases={}
    for source in scope['included_sources']:
        raw=(root/source['bundle_path']).read_bytes()
        assert sha256(raw).hexdigest()==source['sha256'],source['bundle_path']
        data=json.loads(raw);assert data['completed'] and data['mcts_version']=='2.2'
        assert data['tree_config']['stage22_revision'] and len(data['cases'])==1
        case=data['cases'][0];assert case['stop_reason']!='request_errors'
        ds=case['domain'];cid=case['case_id'];name='boolean_extended' if cid.startswith('extended_') else ds
        assert (name,cid) in {(x['dataset'],x['case_id']) for x in scope['completed_questions']}
        h=data['datasets'][name]['sha256'];entry=registry[h];path=root/entry['path']
        assert sha256(path.read_bytes()).hexdigest()==entry['filtered_sha256']
        dataset=json.loads(path.read_text(encoding='utf-8'));meta=dataset['metadata']
        row=next(x for x in case_rows(dataset) if (x.get('observation_set_id') or x.get('observation_id'))==cid)
        fp=identity(ds,row,meta);fam=family(row,ds,meta);leak=[]
        if cid in eval_ids[ds]:leak.append('shared_case_id_with_evaluation')
        if fam in eval_families[ds]:leak.append('equivalent_evaluation_family')
        if ds=='boolean' and (len(meta['variables']),fam[-1]) in eval_obs:leak.append('same_observations_under_variable_swap_even_with_different_operators')
        questions.append(dict(dataset=name,domain=ds,case_id=cid,problem_fingerprint=fp,family_fingerprint=digest(fam),
            evaluation_exclusion=leak,duplicate_of=None,raw_path=source['bundle_path'],source_sha256=source['sha256'],
            rounds=len(case['records']),reference_count=case['reference_count']))
        raw_cases[cid]=(data,case,ds,h,row)
        if (ds,h) not in loaded:
            with redirect_stdout(io.StringIO()):loaded[ds,h]=load_domain(ds,path)
    representatives={}
    for q in sorted(questions,key=lambda x:(x['domain'],x['case_id'])):
        if q['evaluation_exclusion']:continue
        key=q['family_fingerprint']
        if key in representatives:q['duplicate_of']=representatives[key]
        else:representatives[key]=q['case_id']
    rows=[];candidates=[];seen=set();excluded=Counter()
    for q in questions:
        data,case,ds,h,truth=raw_cases[q['case_id']]
        _,prepare,score=loaded[ds,h];obs,keys=prepare(truth)
        for i,r in enumerate(case['records'],1):
            reason=[]
            if q['evaluation_exclusion']:reason.append('evaluation_overlap')
            if q['duplicate_of']:reason.append('duplicate_question_family')
            if r.get('error'):reason.append('request_error')
            if r.get('protocol_error'):reason.append('protocol_error')
            if truncated(r):reason.append('incomplete_or_truncated_round')
            answer=r.get('answer','');key=None;valid=False
            try:key,ok=score(answer,obs);valid=bool(ok and key in keys)
            except Exception as error:reason.append('verifier_exception_'+type(error).__name__)
            if not valid:reason.append('not_verified_reference_hit')
            targets={a['stage']:a['stage_response'].get('text','') for a in r.get('api_requests',[]) if a.get('stage') and a.get('stage_response')}
            item=dict(case_id=q['case_id'],dataset=q['dataset'],domain=ds,round=i,raw_path=q['raw_path'],
                source_sha256=q['source_sha256'],problem_fingerprint=q['problem_fingerprint'],family_fingerprint=q['family_fingerprint'],
                verified_valid=valid,exclusion_reasons=reason,answer=answer,hypothesis_key=key,
                logged_reward=r.get('reward'),stage_outputs=targets,
                reflection_output=r.get('reflection_response',{}).get('text'),
                tree_available=any(x.get('name')=='read_tree' and x.get('result') for x in r.get('tool_trace',[])),
                precise_stage_inputs_saved=all(bool(a.get('request',{}).get('prompt')) for a in r.get('api_requests',[]) if a.get('stage') in ('selection','expansion','candidate')),
                plan_and_rationale_verified=False,sft_ready=False)
            rows.append(item)
            if reason:excluded.update(reason);continue
            answer_id=(q['problem_fingerprint'],digest(key))
            if answer_id not in seen:seen.add(answer_id);candidates.append(item)
    summary=dict(completed_question_files=len(questions),pending_question_count=len(scope['pending_questions']),
        raw_rounds=len(rows),evaluation_overlap_questions=[q['case_id'] for q in questions if q['evaluation_exclusion']],
        duplicate_question_families=[dict(case_id=q['case_id'],representative=q['duplicate_of']) for q in questions if q['duplicate_of']],
        eligible_unique_question_families=sum(not q['evaluation_exclusion'] and not q['duplicate_of'] for q in questions),
        eligible_rounds=sum(not x['exclusion_reasons'] for x in rows),unique_verified_candidates=len(candidates),
        exclusion_reason_counts=dict(excluded),stage_inputs_need_reconstruction=sum(not x['precise_stage_inputs_saved'] for x in candidates),
        per_question=[dict(case_id=q['case_id'],eligible=not q['evaluation_exclusion'] and not q['duplicate_of'],
            unique_candidates=sum(x['case_id']==q['case_id'] for x in candidates)) for q in questions],
        note='Candidate shortlist only: not a tokenizer-ready training set. Reasons, plans and reflections are not certified by final-answer validity.')
    for name,data in (('questions',questions),('summary',summary)):(out/f'{name}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    for name,data in (('all_rounds',rows),('candidate_pool',candidates)):
        with (out/f'{name}.jsonl').open('w',encoding='utf-8') as f:
            for row in data:f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
'''


VERIFY=r'''"""Validate packaged file integrity. Requires only Python's standard library."""
from hashlib import sha256
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'FILES_SHA256.json').read_text(encoding='utf-8'))
for name,expected in manifest.items():
    path=(root/name).resolve();assert path.is_relative_to(root),name
    assert path.is_file() and sha256(path.read_bytes()).hexdigest()==expected,name
scope=json.loads((root/'scope.json').read_text(encoding='utf-8'))
assert len(scope['completed_questions'])==13 and len(scope['pending_questions'])==6
for item in scope['included_sources']:
    d=json.loads((root/item['bundle_path']).read_text(encoding='utf-8'))
    assert d['completed'] and d['mcts_version']=='2.2' and d['tree_config']['stage22_revision']
print('PASS:',len(manifest),'files; exactly 13 completed current-revision results; no pending-case records included')
'''


def main():
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    name=f'DS_MCTS22_completed13_handoff_{stamp}'
    parent=ROOT/'local-backups/handoffs';parent.mkdir(parents=True,exist_ok=True)
    bundle=parent/name;bundle.mkdir()
    sample=json.loads((ROOT/'datasets/sampled.json').read_bytes())
    fixed=[dict(dataset='boolean_extended' if x['case_id'].startswith('extended_') else ds,case_id=x['case_id'],references=x['references']) for ds in ('causal','3d','boolean') for x in sample[ds]]
    source_root=ROOT/'results/ds-stage22-merged/revision'
    completed=[];pending=[];sources=[];dataset_selections={}
    dataset_paths={sha256(p.read_bytes()).hexdigest():p for p in (ROOT/'datasets').rglob('*.json')}
    for job in fixed:
        p=source_root/f"{job['dataset']}-{job['case_id']}.json"
        raw=p.read_bytes();data=json.loads(raw)
        assert data.get('mcts_version')=='2.2' and data.get('tree_config',{}).get('stage22_revision')
        assert 'deepseek' in data.get('requested_model','').lower()
        if not data.get('completed'):
            pending.append(dict(**job,status='pending_do_not_include'));continue
        assert len(data['cases'])==1 and data['cases'][0]['case_id']==job['case_id']
        assert data['cases'][0].get('stop_reason')!='request_errors'
        dest='raw/completed13/'+p.name;(bundle/dest).parent.mkdir(parents=True,exist_ok=True);(bundle/dest).write_bytes(raw)
        completed.append(job);sources.append(dict(**job,original_path=p.relative_to(ROOT).as_posix(),bundle_path=dest,
            sha256=sha256(raw).hexdigest(),requested_model=data['requested_model'],rounds=len(data['cases'][0]['records'])))
        entry=data['datasets'][job['dataset']];dataset_selections.setdefault(entry['sha256'],set()).add(job['case_id'])
    assert len(completed)==13 and len(pending)==6,'Completion scope changed: do not silently enlarge this handoff'
    scope=dict(bundle_version='v1',snapshot_utc=stamp,source_scope='results/ds-stage22-merged/revision ONLY',
        method='MCTS2.2 stage22_revision=true',completed_questions=completed,pending_questions=pending,included_sources=sources,
        prohibited_sources=['old stage22-sampled','Qwen results','incomplete results','other DS methods'])
    write(bundle/'scope.json',scope)
    registry={}
    def rows(d):
        if 'observation_sets' in d:return d['observation_sets']
        if 'datasets' in d:return d['datasets']
        return [x for group in d['datasets_by_n_observations'].values() for x in group]
    for original_sha,ids in dataset_selections.items():
        original=dataset_paths[original_sha];d=json.loads(original.read_bytes());selected=[x for x in rows(d) if (x.get('observation_set_id') or x.get('observation_id')) in ids]
        assert len(selected)==len(ids)
        filtered={'metadata':d['metadata'].copy()}
        if 'observation_sets' in d:filtered['observation_sets']=selected
        elif 'datasets' in d:filtered['datasets']=selected
        else:
            groups={}
            for row in selected:groups.setdefault(str(row['n_observations']),[]).append(row)
            filtered['datasets_by_n_observations']=groups
        for key in ('n_observation_sets','total_observation_sets','n_datasets'):
            if key in filtered['metadata']:filtered['metadata'][key]=len(selected)
        relative=f'reference/datasets/{original_sha}.json';write(bundle/relative,filtered)
        registry[original_sha]=dict(path=relative,original_path=original.relative_to(ROOT).as_posix(),
            filtered_sha256=sha256((bundle/relative).read_bytes()).hexdigest(),included_case_ids=sorted(ids),
            note='Filtered to completed handoff questions; raw result SHA refers to original full dataset, not this filtered copy')
    write(bundle/'reference/dataset_registry.json',registry)
    for ds in ('causal','3d','boolean'):
        write(bundle/f'reference/evaluation_medium27/{ds}.json',(ROOT/f'datasets/medium_sampled/{ds}.json').read_text(encoding='utf-8'))
    upstream=ROOT/'HypoSpace/_HypoSpace-main'
    for p in upstream.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts and (p.suffix=='.py' or p.name in ('README.md','LICENSE','LICENSE.txt')):
            dest=bundle/'vendor/HypoSpace'/p.relative_to(upstream);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    header='import ast,re,sys\nfrom pathlib import Path\nUPSTREAM=Path(__file__).resolve().parents[1]/"vendor/HypoSpace"\n\n'
    write(bundle/'tools/offline_verifier.py',header+functions(ROOT/'experiments/boolean_scoring.py',{'expand_nor'})+'\n\n'+functions(ROOT/'experiments/direct_evaluation.py',{'load_domain'})+'\n')
    helpers='import json\nfrom hashlib import sha256\nfrom itertools import permutations,product\n\n'
    helpers+=functions(ROOT/'server/audit_ds_training_data.py',{'digest','case_rows','identity','truncated'})+'\n\n'
    helpers+=functions(ROOT/'experiments/generate_medium_training.py',{'family'})+'\n'
    write(bundle/'tools/helpers.py',helpers)
    write(bundle/'tools/prepare_candidates.py',PREPARE);write(bundle/'tools/verify_bundle.py',VERIFY)
    write(bundle/'requirements-offline.txt',(ROOT/'requirements.txt').read_text(encoding='utf-8'))
    write(bundle/'environment-used.json',dict(python=sys.version,platform=sys.platform))
    subprocess.run([sys.executable,str(bundle/'tools/prepare_candidates.py'),'--output',str(bundle/'derived')],check=True)
    summary=json.loads((bundle/'derived/summary.json').read_bytes())
    write(bundle/'README_先读我.md',readme(scope,summary))
    write(bundle/'docs/01_处理方法与接入规则.md',METHODS)
    write(bundle/'docs/02_字段说明与SFT制作.md',FIELDS)
    write(bundle/'docs/03_交接验收清单.md',ACCEPTANCE)
    write(bundle/'NOTICE.txt','离线评分代码与vendor/HypoSpace来自本项目现有代码快照，保留其原始README和注释。不额外声明上游许可。包内无模型权重、虚拟环境、SSH文件或API配置。\n')
    for p in bundle.rglob('*'):
        if p.is_file():
            text=p.read_text(encoding='utf-8',errors='replace')
            import re
            assert not re.search(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|\bsk-[A-Za-z0-9]{20,}',text),'Sensitive material detected; do not print content'
    hashes={p.relative_to(bundle).as_posix():sha256(p.read_bytes()).hexdigest() for p in bundle.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    write(bundle/'FILES_SHA256.json',hashes)
    subprocess.run([sys.executable,str(bundle/'tools/verify_bundle.py')],check=True)
    archive=parent/f'{name}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in bundle.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,name+'/'+p.relative_to(bundle).as_posix())
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    digest=sha256(archive.read_bytes()).hexdigest();write(archive.with_suffix('.zip.sha256'),digest+'  '+archive.name+'\n')
    print('HANDOFF_ZIP',archive);print('BYTES',archive.stat().st_size);print('SHA256',digest)


def readme(scope,summary):
    pending='\n'.join('- '+x['dataset']+'/'+x['case_id'] for x in scope['pending_questions'])
    return f'''# DS MCTS 2.2 已完成13题数据交接包 v1

## 交接目的与边界

交给下一位处理人制作Qwen的阶段式SFT数据，不是已经可直接训练的数据集。本包不启动任何模型或训练，不访问API，也不操作DS/Qwen运行中的任务。

**唯一数据源：当前新版 `results/ds-stage22-merged/revision`。只收 `completed=true`、单题正常结束、`mcts_version=2.2` 且 `stage22_revision=true` 的13份结果。** 旧版目录曾有16题完成，这个“16”不能替代当前新版13题的统计；本包没有混入旧版或Qwen记录。

快照UTC：`{scope['snapshot_utc']}`。严格边界为固定19题中的13题已完成，另6题未完成。原始13份数据均保留，不代表13题都允许进入训练。

## 本次离线复核结果

- 原始完成题：{summary['completed_question_files']}题，{summary['raw_rounds']}轮。
- 测评重合隔离：{len(summary['evaluation_overlap_questions'])}题；题号见 `derived/summary.json`。
- 内部等价重复：{len(summary['duplicate_question_families'])}题；保留代表题，其余原始记录仅供审计。
- 最终无重合、无等价重复的题目族：{summary['eligible_unique_question_families']}个。
- 通过基本筛选的轮次：{summary['eligible_rounds']}；按题＋结构化答案去重后的候选：{summary['unique_verified_candidates']}个。
- **这些候选只通过答案评分与基础清洗，不表示Plan、选择理由、复盘都正确，不是SFT-ready。**
- 当前可用题目族覆盖布尔和3D，因果题尚无已完成结果；两道因果题列在pending中，不能把本包说成三个领域齐全的训练集。

## 目录

- `raw/completed13/`：13份完整原始JSON，包括成功/失败轮次、阶段输出、思考文本、工具树、奖励。仅存档，不可整文件直接训练。
- `scope.json`：13/6边界、源路径、SHA256、固定题单。新增数据不能悄悄混入v1。
- `reference/datasets/`：只含这13题的教师评分参考，已从原始题库过滤；原数据集SHA与过滤文件SHA的对应见registry。
- `reference/evaluation_medium27/`：固定中等27题，仅用于查重和测评，禁止作为训练目标来源。
- `derived/questions.json`：逐题的测评隔离、等价重复关系与源文件。
- `derived/all_rounds.jsonl`：逐轮审核索引，含排除原因，便于定位原始记录。
- `derived/candidate_pool.jsonl`：可优先考虑的不同有效候选；仍需重建输入、语义审查、角色掩码和长度检查。
- `tools/`、`vendor/`：离线复核方法及必要评分代码，无LLM依赖。
- `docs/`：详细处理规范、字段与验收要求。

## 使用方法

Python 3.10及以上。完整性检查只需要标准库：

```text
python tools/verify_bundle.py
```

离线重评分需要 `requirements-offline.txt` 中的普通科学计算依赖；不需要torch、transformers、模型权重或API密钥。建议在自己的独立环境安装：

```text
python -m pip install -r requirements-offline.txt
python tools/prepare_candidates.py --output review-output
```

用 `review-output` 复核，避免修改冻结的 `derived`。相同代码与快照应重现摘要；不同Python/依赖版本若结果有差异，应记录并检查，不能默默沿用旧标签。

## 下一位处理人要做什么

先读docs，确认边界与评测隔离，再按题审查候选和路径，重建当前Qwen阶段输入，制定train/validation题目族划分与损失掩码，完成token长度检查。向用户汇报后再进入训练；本交接不授权启动训练。

## 待接入的6题（没有打包其部分作答数据）

{pending}

等它们在当前新版目录整题正常完成后，另做v2：仅新增其中已经完成的题，仍执行哈希、验证、评测排除和内部等价去重。既不能把旧版完成状态算进来，也不能改写本v1。
'''


METHODS='''# 处理流程与边界

## 1. 冻结范围

以scope.json为入口，不扫描接手人机器上的整个results目录。原始文件字节已冻结并附SHA256，只读取此包。一个JSON必须标记completed=true，且cases仅含该题、stop_reason不是request_errors；原始记录都来自当前新版2.2。不收未完成题的interrupted_case，不用旧版16/19补新版13/19，不接入Qwen数据。

## 2. 去重分三层

1. 同一道题在不同源目录或合并目录被复制：按题、问题约束、源SHA和轨迹内容辨认，不能把复制前缀当新增数据。本包已限制单一当前源，一题一份。
2. 题目等价：布尔题检查变量交换，3D检查旋转/镜像/平移，因果题检查变量重命名。同一等价族只选一个代表题作为本版训练候选。不要只因编号不同就分到训练/验证两侧。
3. 同题答案重复：以评分器得到的结构化hypothesis_key去重。不能按字符串排版去重。不同树状态下的重复答案可作为少量fallback样本，但不能无上限重复；本candidate_pool默认每题每结构答案一份。

对于变量交换后的等价题，本版直接隔离非代表题，不把它的答案硬套到代表题：那需要对题目、变量、Plan、答案、复盘整体一致变换。

## 3. 测评隔离

中等27题固定保留作测评。共享题号、等价题目族都隔离。布尔题即使算子集合不同，只要观测在变量交换后重合，也保守隔离，避免训练-测评混淆。此项独立于“答案是否正确”：正确答案也可能属于测评题，不能训练。

reference/evaluation_medium27中所有答案仅供检查，不进入训练输入或标签，也不能用于补写教师推理。

## 4. 答案离线重验证

程序按每题对应的源题库约束加载评分器，解析答案，检查满足观测/结构约束且其结构化键属于该题参考解集合。Boolean沿用项目NOR规范化与生成器假设空间，不能只比较真值表而丢掉结构限制；3D验证高度与投影；因果题未来接入时验证图与扰动观测。

以下轮次不进候选池：请求错误、protocol_error、主响应/任一阶段/复盘finish_reason=length、final_answer_missing、无法独立重评分为参考解、题目被评测或等价去重隔离。valid字段仅作原日志信息，不能取代重新评分。

## 5. 判断阶段标签是否适合SFT

- 候选：优先有效且符合所选路径及本轮Plan的答案。程序只证明答案满足题目，不证明它遵守自然语言Plan。应检查不一致的路径/Plan；不能为了保留答案修改教师理由或偷换输入。
- 扩展：Plan需具体、合法、与已有子步骤不同，且候选确实执行它。无法判断者放人工复核，不直接当金标准。
- 复盘：必须用当时可见的历史反馈核实其批评和建议，不能用将来才出现的答案回填“事前已知”。正解轨迹也可能含错误复盘。
- 选节点：合法Branch不代表最优Branch；一次成功不足以证明它优于其他节点。初版可不训练这个阶段，或只保留充分审查的样本。
- 原始reasoning_text是模型隐藏思考，不自动作为监督标签；优先使用实际提交的阶段输出。不要编造<think>内容。

## 6. 状态与奖励

reward=+1通常是新有效解，-1通常是有效但重复，0不是笼统的“所有推理都错误”。应保留当时已发现答案集合，不能把不同时间的+1/-1拼成相同输入下的偏好对。本交接仅SFT准备，不生成DPO数据。

## 7. 追加未完成题

六题完成后先确认当前新版source/completed状态，再冻结新字节和SHA，建立v2及parent_bundle字段。重新执行评测隔离与全局题目族去重；只增加新完成题，不重新读取v1路径作为活动文件。最终按题目族划分训练/验证，不能按轮次随机拆分。同一题多次搜索、同一题的变量替换都不能跨split。
'''


FIELDS='''# 字段与SFT样本制作

## 原始结果结构

- requested_model：教师模型请求名称；mcts_version=2.2，tree_config.stage22_revision=true：当前新版标识。
- datasets[dataset].sha256：原完整题库SHA。包内reference是过滤副本，不能拿过滤后的SHA冒充原SHA，映射见dataset_registry.json。
- cases[0].prompt：原始题目。cases[0].records[i]：一轮搜索，索引从0开始；derived里的round从1开始。
- record.prompt：本轮组合提示，含历史及Independent review；并不等于每个阶段精确发送的wire请求。
- record.tool_trace：按当时实际读树结果保存的树及节点统计。优先用本轮trace.result，不用实验结束后单独.tree.json代替历史树。
- record.api_requests[*].stage：read_tree/selection/expansion/candidate；stage_response.text：对应教师阶段输出。部分日志无完整request.prompt/system。
- record.reflection_prompt与reflection_response.text：复盘输入/输出；valid、reference_hit、hypothesis_key、reward是当时评分标签，需重评分对照。
- record.plan_path：所选路径与扩展后的Plan序列；record.answer是提交给评分器的候选，不含控制头；record.text包含Branch/Plan等组合结果。

## 必须重建/核对的输入

大部分DS阶段输出没有保存精确wire输入，不可直接把所有阶段输出拼在一个“题目→答案”样本里并称其为阶段训练。重建时需要原题、当时已发现答案、当前完整树/选中路径、当轮已生成的前序决策，以及对应版本的阶段指令。

可以把经过审查的教师输出映射到当前Qwen“原题是参考数据、当前阶段单独下指令、附格式示例”的任务结构，但必须标记input_reconstructed=true，并核对教师输出在重建输入中仍然成立。这不是声称逐字恢复了原教师wire请求。请先和Qwen侧确认最终模板版本；模板快照和SHA也应随训练集冻结。

## 建议的样本结构（模板，不是包内已完成数据）

每条包含sample_id、question_id、family_id、stage、messages、source_path/source_sha256/round、quality_flags。messages中system写当前阶段契约，user写已知题目/历史/树或路径等，assistant只放本阶段审查通过的输出。read_tree工具事件不能当作自然语言候选答案；若训练工具调用，需保留正确的tool_call/tool_response角色结构。

候选训练不能忽略输入中的Plan限制；如果无法确认Plan语义一致性，先排除而非自动改成无约束题目。恢复历史时不能让未来答案进入当前输入。

## Tokenizer和loss mask

待使用Qwen3实际chat_template做逐条编码检查。训练loss只覆盖需要学习的assistant目标，system/user/tool观察及padding都应mask。是否训练隐藏思考尚未确定，本包不擅自决定。不能把DS隐藏思考直接当Qwen原生思考模板。检查截断位置，超长样本排队处理，不静默截掉答案或必要约束。4K是原先建议的起点，不保证这些完整树样本都能放下。

## 当前交接包不是最终训练集

derived/candidate_pool.jsonl没有messages或token标签，sft_ready=false；它是待加工候选索引。其Plan/理由/复盘均没有因为答案正确而自动获得“正确”认证。下一位处理人应做人工/规则审查、输入重建、按题目族split、token长度与loss mask验证后，再输出正式train.jsonl和validation.jsonl及manifest。
'''


ACCEPTANCE='''# 接手验收清单

1. 校验ZIP外部SHA256和包内FILES_SHA256.json，再运行tools/verify_bundle.py。
2. 确认只含当前新版13题原始结果；pending6只有题号，没有部分作答记录；无旧版16题混入，无Qwen结果。
3. 在独立环境离线重跑tools/prepare_candidates.py --output review-output，与derived/summary.json比较。
4. 先检查questions.json中的evaluation_exclusion和duplicate_of；这两类不进训练。
5. 对候选及计划、复盘分别复核，记录接受/拒绝及理由；不要把“评分有效”当所有阶段都正确。
6. 冻结当前Qwen阶段模板，构建messages，记录输入重建标记及源SHA/round。
7. 以题目族划分train/validation；测评27题及其答案不进入训练。
8. 在实际Qwen tokenizer上检查长度、角色掩码与思考模板；本包没有做这一步。
9. 导出新数据版本manifest，包括source_bundle_hash、样本ID、split、阶段分布、排除清单、模板hash与依赖版本。
10. 只交付数据与检查报告；本任务没有授权开始微调。未完成6题后续另接新版本，不覆盖v1。
'''


if __name__=='__main__':main()
