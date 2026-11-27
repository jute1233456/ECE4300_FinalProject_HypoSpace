# B 数据集语义质量审查（Claude Code）

- 日期：2026-11-26
- 输入：`training/data/ds22-new4-20261126/train.jsonl`，507 条 = 169 轮 × 3 阶段（selection 169 / expansion 169 / candidate 169）。
- 覆盖：`n1_inputs_01_outputs_1`(26 轮)、`n1_inputs_10_outputs_1`(23)、`n6_hard_021`(63)、`n6_hard_029`(57)。**因果域首次进入训练数据**。
- 审查方式：模型逐条语义审查 + 离线结构复核；只读本地文件，未训练、未运行答题实验、未发 API/网络请求、未操作进程/服务器/Git。
- 产物：`decisions.jsonl`(507)、`build_filtered.py`、`train.filtered.jsonl`、`rejected.jsonl`、`uncertain.jsonl`、`manifest.json`；另有非命中推理审查：`decisions-reasoning.jsonl`(90)、`build_reasoning.py`、`train.reasoning_only.jsonl`(21)。

## 1. 审查依据

- `docs/TRAINING_DATA_PIPELINE.md`、`training/stage_sft.py`：重建规则、`messages` 结构、Plan/Branch/candidate 不变量。
- `datasets/balanced/boolean.json`（n1 两题，各 27 个标准解）、`datasets/balanced/causal.json`（n6 两题，63 / 57 个参考图）。
- 交接包 `tools/offline_verifier.py`：**169/169 个 candidate 重新打分，全部 valid 且在参考集内**（因果按图 hash、布尔按 mechanistic key）。
- 每条样本的 system 契约、user（重建树/菜单/复盘/前序决策）与 assistant 目标。

## 2. 决策口径

- **keep**：目标与输入一致、结构合法、未发现会教坏模型的硬错误。
- **uncertain**：答案本身可能有效，但推理中存在具体可指出的问题（错误计层、幂等填充、虚构规则、前后冲突、无具体步骤等），隔离、不纳入推荐训练集。
- **reject**：目标的核心主张错误且直接把折叠后的重复当作新结构。本轮 **0 条**：新数据里的问题集中在"错误计层/冲突/填充"，按 uncertain 处理更合适；没有出现 A 审查中那种"把折叠本身称为新结构"的直接主张。

## 3. 主数据集结果

| 决策 | selection | expansion | candidate | 合计 |
|---|---:|---:|---:|---:|
| keep | 164 | 157 | 166 | **487** |
| reject | 0 | 0 | 0 | 0 |
| uncertain | 5 | 12 | 3 | **20** |

| 题目 | keep | uncertain |
|---|---:|---:|
| `n6_hard_021`(因果) | 187 | 2 |
| `n6_hard_029`(因果) | 165 | 6 |
| `n1_inputs_01_outputs_1` | 73 | 5 |
| `n1_inputs_10_outputs_1` | 62 | 7 |

其中 1 条 keep 带 caveat（`keep_valid_minor_caveat`：`n6_hard_029` r67 expansion 把 10 条边说成 9 条，方向与候选均正确）。

## 4. 主要问题

### 4.1 布尔两题的"深度三层"循环（无效搜索的根源）

`n1_inputs_01` 有 **806,532 tokens（40%）**、`n1_inputs_10` 有 **820,270 tokens（46%）** 的记录未命中参考解；因果两题只有 1% / 0%。审查确认这不是偶然：

- 大量 selection/expansion 把 `OR→AND→NOT`、`NOT→AND→NOT` 这类**三层结构**称作"depth 2 / 合法"；
- 同一个三层候选被反复重提（如 `y OR (x AND NOT y)`、`NOT((NOT x) AND (y))`），有的轮次甚至与上一轮完全相同却称为"新子分支"；
- 这类样本全部判 uncertain（`uncertain_wrong_depth` 等），不进入训练集。

### 4.2 幂等填充

`n1_inputs_10` r14/r16/r25：用 `y AND y`、`x AND x` 的重复操作数折叠出"新"结构（`NOT((y) AND (y))`、`x OR (x AND x)`、`NOT((NOT x) OR (y AND y))`）。答案 key 有效，但监督文本在教模型用重复填充制造 novelty → expansion/candidate 判 uncertain。

### 4.3 虚构规则

- `n1_inputs_01` r53/r58：把"深度过滤器应先于 novelty/评分"当成评测流程规则；
- `n6_hard_021` r3：把"≤5 条边的最小骨架"写成必须约束（题目只限制 ≤10 条边），下一轮 review 自己纠正。

### 4.4 前后冲突与误读

- `n6_hard_021` r70、`n6_hard_029` r19/r42/r69/r75：selection 明确要求保留/加入某条边，expansion 却删掉或换成另一条；
- `n6_hard_029` r6/r22：所选分支已经包含某条边，selection 仍把"加入它"当成未尝试动作。

### 4.5 边界样本（保留）

`n6_hard_029` r67 的 expansion 方向正确、候选有效，只是边数口误（10 说成 9）；标 `keep_valid_minor_caveat`，仍保留在训练集。

## 5. 非命中轮次的推理（reasoning-only）

用户指出"没命中参考解不代表思路没价值"。因此对 45 个可重建的非命中轮次单独审查了 90 条 selection/expansion：

| 决策 | 数量 |
|---|---:|
| keep | **21** |
| uncertain | 69 |

uncertain 的主要原因是 `uncertain_wrong_depth`(43)、`uncertain_no_concrete_step`(7)、`uncertain_procedural_guard`(5)、`uncertain_ignored_validity_rule`(3)、`uncertain_selection_expansion_conflict`(3)。

**结论**：确实有 21 条"答案没命中、但选支/计划本身成立"的推理被保留下来（例如布尔题里深度确实合规的三文字 OR、因果题里合法的近上限边集）；但大多数非命中轮次的问题是真实的推理错误，不是"被埋没的好思路"。

这些样本输出到 `train.reasoning_only.jsonl`，每条带 `candidate_verified=false`、`round_outcome=invalid`、`input_reconstructed=true`，**不并入主训练集**，是否使用由使用者决定。

## 6. 文件与校验

| 文件 | 条数 | SHA256 |
|---|---:|---|
| `train.filtered.jsonl` | 487 | `eb200843e1ea8a0dbe5e4706022b765ca2a04c0a201445ebbeca456d74fc04e2` |
| `rejected.jsonl` | 0 | `e3b0c44…`（空文件） |
| `uncertain.jsonl` | 20 | `b724c8de…` |
| `train.reasoning_only.jsonl` | 21 | `b6f02151…` |
| `decisions.jsonl` | 507 | 见 `manifest.json` |
| `decisions-reasoning.jsonl` | 90 | — |

`build_filtered.py` 已验证：507 个 `sample_id` 全覆盖、无重复、无多余；输出按原始顺序、原始 JSON 行字节输出；源文件与 ledger 未被修改。

## 7. 局限与复核建议

1. 这是**模型语义审查**，不是形式证明，也未训练；不能保证 487 条 keep 的每个推理细节都正确。
2. 因果域是首次审查：候选图全部通过离线评分器的参考集核对，但"边集最优性/最小性"的推理论证仍属模型判断。
3. `g3_h3_b4_001` **暂缓处理**（用户明确）：现在不做；等它跑完后，再重新生成 canonical、建新数据集并单独审查（不在本轮 507 条内）。
4. 若之后修改 prompt、`stage_sft.py` 或 `BUILDER_VERSION`，应重新审查。
5. 仍未做 tokenizer / loss mask 实测，这与语义审查是两件事。
