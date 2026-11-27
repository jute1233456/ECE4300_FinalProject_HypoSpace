# 训练数据整理流水线(training/)

面向后续接手的 agent:这里说明“跑题结果 → 训练样本”的唯一路径、不变量和扩展方法。
**不要绕过本流水线手工拼样本**,否则 ledger 去重和版本追溯会失效。

## 0. 边界

- 本流水线**只读** `results/` 下的运行结果,不启动模型、不访问 API、不操作 DS/Qwen 进程。
- **用户决定(2026-11-26):训练数据稀少,本任务所有已完成题全部进入训练,不做评测重合/等价族的排除;评测阶段对测试集做减法**(即从评测题里去掉已用于训练的题号,或单独报告这些题)。
  当前因此进入训练的、与评测重合的题:`g3_h3_b2_001`、`n2_inputs_00_01_outputs_01`、`n2_inputs_00_10_outputs_01`;等价族重复题:`n3_inputs_00_10_11_outputs_101`(与 `n3_inputs_00_01_11_outputs_101` 同族)。
  27 道中等评测题是另一个任务,不在本流水线范围。
- 交接包 `local-backups/handoffs/DS_MCTS22_completed13_handoff_20261126T063000Z/` 是 **v1 冻结包,只读不改**。

## 1. 目录

```
training/
  canonical.py       同一任务的多次运行 → 一个规范化文件
  stage_sft.py       规范化记录 → 阶段训练样本(输入重建 + 校验)
  ingest.py          ledger 去重(append-only)
  sources/           <task>-canonical.jsonl + MANIFEST.json
  data/<dataset>/    train.jsonl / valid.jsonl / manifest.json
  ingest/ledger.jsonl
```

生成物都是**可重建的**:删除后按下面命令重跑即可,不要手工编辑 JSONL。

## 2. 三步与命令

### ① 规范化(`training/canonical.py`)

规则:**同一任务、同一提示词版本、同一题单的多次运行,是一份数据的多个快照,不是多份数据。**

- 每题取**记录数最多**的快照(并列取最新);
- 其余快照必须是通过签名 `(exploration_round, hypothesis_key, answer)` 校验的**记录级前缀**,否则在 manifest 标记冲突并停止使用;
- 原始运行目录只作存档,不再作为第二数据源;
- 版本不一致的运行写入 `excluded_runs`(含原因),不静默混入。

```powershell
python -m training.canonical --task ds22-sampled19 --out training/sources
```

产出:`training/sources/ds22-sampled19-canonical.jsonl`(每题一行,含全部记录与 provenance)、`MANIFEST.json`。

### ② 重建与出样本(`training/stage_sft.py`)

DS 的 stage 客户端只保存了阶段**输出**,没有保存逐阶段 wire 输入。重建依据(全部来自记录本身):

| 目标 | 来源 |
|---|---|
| 原题 + 独立复盘 | `record.prompt`,按 `SplitTreeToolClient.generate` 同一规则切分(`\n\nChoose the direction most likely` 与 `Independent review for this attempt:` 标记) |
| 当轮完整树文本 | `record.tool_trace` 中最后一次 `read_tree` 的 `result`,用 `json.dumps(..., ensure_ascii=False, separators=(',',':'))` 复现 |
| 可选分支菜单 | 快照节点上的 `branch` 字段,按序号排序;描述用 `ReasoningTreeAgent.describe` 的同一格式(路径 Plan + Mode + 子节点 Plan) |
| 选中的分支 | selection 输出的 `Branch: N` → 菜单第 N 项 |
| 下一步 Plan | expansion 输出首行,并与 `record.plan_path[-1]` 校验一致 |
| 三阶段 assistant 目标 | `api_requests[*].stage_response.text`(完整文本,不是截断后的首行) |
| 前序阶段决策 | 与运行时代码一致,以 `selection: ...` / `expansion: ...` 形式附在提示词末尾 |

重建后的输入用**线上 Qwen 实际使用的模板**包裹:`llm.qwen_stage_protocol.isolate_stage(stage, prompt, instruction, examples=True)`(`isolated_examples_v1`),阶段指令与 `llm/qwen_stage_revision.py` 逐字一致。

样本结构:`messages = [system(阶段契约), user(引用 JSON + 当前阶段契约 + 格式示例), assistant(教师输出)]`。

```powershell
python -m training.stage_sft --canonical training/sources/ds22-sampled19-canonical.jsonl `
  --out training/data/ds22-extended-n1 --cases extended_n1_inputs_00_outputs_1 --append-ledger
```

### ③ 去重(`training/ingest.py`)

`ledger.jsonl` 每行记录 `sample_id`(由 case、轮次、阶段、输入哈希、输出哈希决定)、来源文件 SHA、`builder_version`、时间。

语义:**输出文件始终是该 case 集合的完整结果,可重复生成、字节稳定;ledger 只用于统计增量、并支持 `--new-only` 跳过已整理过的题目。**
重跑②不会因为“已整理”而删掉样本(早期版本这样写会清空数据集,已修正);`--append-ledger` 只追加 `new_samples`。
`--new-only` 若选中的题全部已整理,直接打印提示并退出,不改写任何文件。

```powershell
python -m training.ingest --check training/data/<dataset>/train.jsonl
```

## 3. 不变量(改代码时必须保持)

1. **只收正常轮次**:`reference_hit=true`、非截断(`finish_reason != length`、`final_answer_missing` 为假)、无 `protocol_error`;空记录跳过。
2. **按结构去重**:同一题的候选按 `hypothesis_key` 去重,保留首次出现。
3. **重建输入必须仍能容纳教师输出**,否则丢弃并计入 issues:
   - selection 的 Branch 必须在重建菜单范围内;
   - expansion 的 Plan 必须与 `plan_path` 一致;
   - candidate 必须与 `record.answer` 一致。
4. **每条样本带标记**:`input_reconstructed=true`、`plan_unverified=true`(教师的 Plan/理由未经语义认证)、`reference_verified=true`、`protocol`、`builder_version`。
5. **按题目族切分** train/valid,不按记录切;当前只有 1 个族时不产生 valid。
6. 生成物不可手改;改动逻辑必须提升 `BUILDER_VERSION`。

## 4. 当前状态(2026-11-26 16:30 UTC+8)

- 规范化:`ds22-sampled19-canonical.jsonl`,19 题 / 469 轮,14 题完成、5 题未完成;三个快照(merged / official / ark-resumed)全部通过前缀校验,完成题内容逐字节一致。
- **数据集(记录版)**:`training/data/ds22-all-completed/` = **309 条**(103 个去重有效候选 × selection/expansion/candidate),覆盖全部 14 道已完成题、14 个族;弃置 43 轮(21 未命中、18 重复、6 空记录、1 协议错误、部分截断)。
- 早期子集:`training/data/ds22-extended-n1/`(42 条,只含 v1 之后新完成的 `extended_n1_inputs_00_outputs_1`),已被上面的全集包含,仅作追溯。
- ledger:309 条,`--new-only` 重跑会正确跳过全部 14 题。
- 尚未做:tokenizer / loss mask 实测;train/valid 按族切分(用户暂不要求,评测用减法处理)。

## 5. 新运行接入检查清单

1. 新运行必须是同一任务同一版本(`tree_config.stage22_revision=true`,阶段指令未改);版本变了先升 `BUILDER_VERSION` 并重新映射。
2. 把运行目录加进 `training/canonical.py` 的 `runs` 列表,重跑①;确认 manifest 中无 prefix 冲突、无重复计数。
3. 重跑②(可加 `--cases` 限定),看 issues 是否只有 `not_reference_hit` / `duplicate` / `empty record`;出现 `reconstruction failed` / `plan does not match` / `candidate does not match` 说明日志结构或模板变了,先排查再入库。
4. 追加 ledger,提交 `sources/`、`data/`、`ingest/ledger.jsonl` 与 manifest。

## 6. 已知限制

- `label_mask_checked=false`:loss mask 与长度需在服务器用 Qwen3 真实 chat_template 实测(见 `server/train_qlora.py` 的编码逻辑)。
- **长度(已用 Qwen3 tokenizer 实测,`training/data/ds22-all-completed/lengths-tokenizer.json`)**:309 条,min 1448 / p50 2152 / p90 4386 / p99 12036 / **max 13706** tokens;thinking on/off 结果一致,模板前缀校验 0 失败。selection 是长尾来源(p50 3129、p90 8566、max 13706),expansion/candidate 最大 4332/4386。
  ≤8192 会丢 11 条(全为 selection),≤12288 丢 3 条,≥14336 全覆盖。**建议 `--max-length 14336` + 动态 padding**(bsz=1 时按实际长度,不固定 pad 到上限),峰值显存由最长的几条 selection 决定,必须用 T2 冒烟实测。
- 教师 Plan/理由未认证;不要声称“因为这些答案正确,所以中间步骤正确”。
- 当前只有 3D/布尔两域,因果题仍无已完成运行。
- 多次运行若出现**非前缀**关系(例如中途改了提示词又继续跑),程序会报冲突并拒绝合并,需要人工决定以哪次为准。
