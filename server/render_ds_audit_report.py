"""Self-contained, local-only audit viewer and compact case inventory."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('audit',type=Path);args=p.parse_args()
    out=args.audit;s=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    turns=[json.loads(line) for line in (out/'all_turns.jsonl').read_text(encoding='utf-8').splitlines()]
    cases={}
    for x in turns:
        if not x['variant'].startswith('2.2'):continue
        k=(x['domain'],x['case_id'])
        row=cases.setdefault(k,dict(domain=k[0],case_id=k[1],revision=0,original=0,positive=0,duplicate=0,
            excluded=0,leakage=set(),answers=set(),exact_inputs=0))
        row['leakage'].update(x['leakage_reasons'])
        if x['tier']=='excluded':row['excluded']+=1;continue
        row['revision' if x['variant']=='2.2_revision' else 'original']+=1
        row['positive']+=int((x['logged_reward']or 0)>0);row['duplicate']+=int((x['logged_reward']or 0)<0)
        row['answers'].add((x['problem_fingerprint'],json.dumps(x['hypothesis_key'],sort_keys=True)))
        row['exact_inputs']+=int(x['exact_stage_inputs_logged'])
    rows=[]
    for row in cases.values():
        row['leakage']=sorted(row['leakage']);row['answers']=len(row['answers']);rows.append(row)
    rows.sort(key=lambda x:(bool(x['leakage']),x['domain'],x['case_id']))
    (out/'stage22_case_inventory.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    payload=json.dumps(dict(summary=s,cases=rows),ensure_ascii=False).replace('<','\\u003c')
    template='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DS 2.2 教师数据盘点</title>
<style>body{font:15px/1.6 system-ui,sans-serif;margin:32px auto;max-width:1200px;padding:0 24px;color:#20252b;background:#fafafa}h1{font-size:24px}h2{font-size:19px;margin-top:30px}p{max-width:1000px}.metrics{display:flex;gap:42px;border-block:1px solid #ddd;padding:18px 0}.metrics b{font-size:24px;display:block}.note{border-left:3px solid #496960;padding-left:16px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;border-bottom:1px solid #ddd;padding:9px 7px;vertical-align:top}th{background:#eee}input,select{font:inherit;padding:7px;border:1px solid #bbb;margin:0 10px 15px 0}code{font-size:12px;overflow-wrap:anywhere}.muted{color:#60666e}.excluded{color:#8a4b31}a{color:#245c51}details{margin-top:20px}.scroll{overflow:auto}</style>
<h1>DS 教师数据盘点：优先使用 MCTS 2.2</h1>
<p class="muted" id="source"></p><div class="metrics" id="metrics"></div>
<p class="note">这是现有数据的离线审计，不会操作 DS 任务，也没有调用模型或启动训练。有效候选已重新评分；选节点理由、Plan 与候选的一致性、复盘事实仍需复查，不能因答案正确就全部视为正确标签。</p>
<h2>优先级与可用范围</h2><ul><li>首先使用新版理由反馈版的有效轨迹；原版 2.2 用来补充候选和不同搜索状态。</li><li>默认以历史奖励 +1 的新有效解为主体；重复解只保留少量，避免把低效重复训练成习惯。</li><li>19 道训练题中有 4 道与中等 27 题重合或等价，已隔离；当前安全范围为 15 道。</li><li>全部入选 2.2 记录均缺少精确阶段请求输入。题目、完整树、阶段输出和复盘仍在快照中；需重建并校验输入，再制作角色掩码和长度检查。这里的 JSONL 不是可直接训练的 SFT 文件。</li></ul>
<div class="scroll"><table><thead><tr><th>教师方法</th><th>去重后的记录</th><th>离线复核有效</th><th>隔离泄漏与失败后优先轮次</th><th>安全范围内不同答案</th></tr></thead><tbody id="variants"></tbody></table></div>
<h2>2.2 按题清单</h2><p class="muted">轮次保留不同树状态；不同答案按“题目＋结构化答案”去重。两种轮次不能简单相加当作不同答案数。</p>
<input id="filter" placeholder="筛选题号或领域"><select id="scope"><option value="all">全部 19 题</option><option value="safe">无重合的 15 题</option><option value="excluded">隔离题目</option></select>
<div class="scroll"><table><thead><tr><th>领域 / 题号</th><th>新版轮次</th><th>原版轮次</th><th>不同有效答案</th><th>历史 +1 / -1</th><th>建议</th></tr></thead><tbody id="cases"></tbody></table></div>
<h2>源运行盘点</h2><p class="muted">以下是原文件统计，包含合并目录的复制前缀，不能直接加总为独立样本。分区版仅有预建题目文件，快照中没有搜索轮次。</p><div class="scroll"><table><thead><tr><th>来源目录</th><th>版本</th><th>文件 / 已完成题</th><th>原始轮次</th><th>原始有效轮次</th></tr></thead><tbody id="runs"></tbody></table></div>
<h2>文件与下一步</h2><ul><li><a href="stage22_verified_turns.jsonl">2.2 优先轮次清单</a>：附源路径、快照 SHA256、轮次、阶段输出，含不同搜索状态。</li><li><a href="stage22_unique_answers.jsonl">2.2 不同有效答案清单</a>：同题同答案只保留一份，优先新版。</li><li><a href="summary.json">机器可读汇总</a>；<a href="sources.json">原始来源索引</a>；<a href="stage22_case_inventory.json">按题清单</a>。</li><li>先重建新版阶段输入并抽查理由/步骤，再按题目分训练和验证；中等27题始终留作评测。不要把同题不同轮次拆到训练和验证两侧。</li></ul>
<details><summary>次要候选池与统计口径</summary><p id="secondary"></p><p>排除规则含：评分器无法验证、请求失败、协议失败、任一阶段或复盘截断、与测评共享编号、变量交换/空间变换等价。相同观测但允许算子不同也保守隔离。原始快照只保存在本地；不提交 Git。</p></details>
<script>const D=PAYLOAD;
function cell(row,text,cls){const td=document.createElement('td');td.textContent=text;if(cls)td.className=cls;row.append(td)}
function row(target,values,cls){const tr=document.createElement('tr');values.forEach(v=>cell(tr,v,cls));document.getElementById(target).append(tr)}
const S=D.summary,P=S.preferred_stage22;document.getElementById('source').textContent='来源：本地 DS 结果 JSON 的冻结快照；采集源 '+(S.input_snapshot||S.snapshot_utc)+'。计数为单次快照，运行中的任务以后可能增加数据。';
[[P.eligible_turns,'可优先整理的2.2轮次'],[P.unique_answers,'不同有效答案'],[P.problems,'无评测重合的题目'],[P.positive_reward_turns,'历史奖励+1轮次']].forEach(([n,t])=>{const x=document.createElement('div');const b=document.createElement('b');b.textContent=n;x.append(b,document.createTextNode(t));document.getElementById('metrics').append(x)});
const names={'2.2_revision':'新版2.2：理由与反馈','2.2_original':'原版2.2','other_DS':'其他DS方法（备用）'};Object.entries(S.variants).sort(([a],[b])=>a==='2.2_revision'?-1:b==='2.2_revision'?1:a.localeCompare(b)).forEach(([k,v])=>row('variants',[names[k]||k,v.unique_rounds,v.verified_valid,v.primary,v.unique_safe_answers]));
function render(){document.getElementById('cases').replaceChildren();const q=document.getElementById('filter').value.toLowerCase(),scope=document.getElementById('scope').value;D.cases.filter(x=>(x.domain+' '+x.case_id).toLowerCase().includes(q)&&(scope==='all'||(scope==='excluded')===Boolean(x.leakage.length))).forEach(x=>row('cases',[x.domain+' / '+x.case_id,x.revision,x.original,x.answers,x.positive+' / '+x.duplicate,x.leakage.length?'隔离：共享题目或等价观测':'候选可优先用；决策/输入待审'],x.leakage.length?'excluded':''))}document.getElementById('filter').oninput=render;document.getElementById('scope').onchange=render;render();
S.source_runs.filter(x=>x.variant.startsWith('2.2')).forEach(x=>row('runs',[x.run,x.variant,x.files+' / '+(x.completed_cases||0),x.rounds||0,x.logged_valid||0]));document.getElementById('secondary').textContent='扫描 '+S.source_files+' 个确认DS模型的结果文件，原始 '+S.raw_rounds+' 轮。其他DS方法还有 '+S.variants.other_DS.unique_safe_answers+' 个按题去重、通过本地验证且无保守重合的候选答案；这批不作为2.2决策训练的首选。全池不同答案数 '+S.shortlisted_unique_answers+' 已包含2.2答案，不能再次相加。';
</script></html>'''
    (out/'report.html').write_text(template.replace('PAYLOAD',payload),encoding='utf-8')
    print(out/'report.html')


if __name__=='__main__':main()
