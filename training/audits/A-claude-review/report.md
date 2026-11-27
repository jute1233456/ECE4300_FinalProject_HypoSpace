# A 数据集语义质量审查（Claude Code）

- 日期：2026-11-26
- 输入：`training/data/ds22-all-completed/train.jsonl`
- 输入 SHA256：`23d181e2661a0b6e81b59acdd010757a897952df064c870f5a3c6320c65b58e2`
- 输入条数：309（selection 103 / expansion 103 / candidate 103）
- 审查方式：模型逐条语义审查；只读本地文件，未训练、未运行答题实验、未发 API/网络请求、未操作进程/密钥/服务器/Git。
- 产物：`decisions.jsonl`、`build_filtered.py`、`train.filtered.jsonl`、`rejected.jsonl`、`uncertain.jsonl`、`manifest.json`、本报告。

## 1. 审查依据

实际读取并交叉核对：

- `docs/TRAINING_DATA_PIPELINE.md`、`training/stage_sft.py`：样本如何从 canonical 记录重建、`messages` 三段结构、plan/branch/candidate 校验不变量。
- `datasets/balanced/boolean.json`、`datasets/balanced/boolean_extended.json`、`datasets/balanced/3d.json`：各题的 observations、允许算子、深度上限、标准解集合。
- `experiments/boolean_scoring.py` 与上游 `boolean_benchmark.py` / `boolean_dataset.py`：mechanistic key、NOR 原语计层、真值表评分。
- 上游 3D benchmark：top-view、支撑、层数、最大高度、结构 hash。
- 每条样本的 `messages[0]` system、`messages[1]` user（含重建树、branch menu、独立复盘、前序阶段决策）与 `messages[2]` assistant 目标。

逐条检查的内容：

1. **阶段契约**：selection 只能选分支并说明 legal choice / untried direction / avoid；expansion 只能提出一个具体下一步约束；candidate 只能给完整候选。检查是否在错误阶段答题、是否声明 EXHAUSTED、是否输出 Branch/Plan。
2. **与输入一致**：理由是否引用了实际存在的 branch、tree、review、prior attempt；是否虚构约束或隐藏评分规则；是否依赖未提供的信息。
3. **结构评分**：是否错误计 NOR/嵌套层数；是否把逻辑真值等价自动当成结构重复；是否把“找到一个解”当成“找全”。
4. **计划与候选**：candidate 是否遵守所选 branch 和 expansion 的 next step；expansion 是否与 selection 的明确方向冲突。
5. **重复扩展**：是否用 idempotence/associativity 会折叠的写法冒充新结构。
6. **候选本身**：每条 candidate 都用独立脚本重新解析，核对是否在标准解集合中、是否满足 observations；3D 另核对 top-view、逐层支撑、非空底层和高度上限。**103/103 个 candidate 目标都通过该结构核对**。

说明：候选答案正确不代表 selection/expansion 的推理正确；本轮主要问题恰好集中在 reasoning/context，而不是最终表达式。

## 2. 决策口径

- **keep**：目标本身与输入一致，结构合法，未发现会教坏模型的硬错误。keep 也写了具体理由（branch 是否存在、plan 是否等于 plan_path、candidate 的 key/高度是否在标准解集合）。
- **reject**：目标的核心主张错误或直接把折叠后的重复当成新结构，属于不适合训练的负面数据。
- **uncertain**：目标答案本身有效，但输入/上下文有具体矛盾、虚构规则、重复折叠或计划-候选冲突；按用户要求隔离，**不纳入推荐训练集**。
- 评测题重合、逻辑真值等价、多解题本身都不是删除理由；按项目结构评分和用户“训练全部完成题、评测做减法”的安排处理。

## 3. 数量

| 决策 | selection | expansion | candidate | 合计 |
|---|---:|---:|---:|---:|
| keep | 101 | 99 | 100 | **300** |
| reject | 1 | 1 | 0 | **2** |
| uncertain | 1 | 3 | 3 | **7** |
| 合计 | 103 | 103 | 103 | **309** |

| 题目 | keep | reject | uncertain |
|---|---:|---:|---:|
| extended_n1_inputs_00_outputs_1 | 41 | 0 | 1 |
| g3_h3_b2_001 | 27 | 0 | 0 |
| g3_h3_b3_006 | 81 | 0 | 0 |
| n2_inputs_00_01_outputs_01 | 38 | 2 | 2 |
| n2_inputs_00_10_outputs_01 | 42 | 0 | 0 |
| n3_inputs_00_01_10_outputs_011 | 10 | 0 | 2 |
| n3_inputs_00_01_11_outputs_101 | 11 | 0 | 1 |
| n3_inputs_00_10_11_outputs_101 | 11 | 0 | 1 |
| n3_inputs_01_10_11_outputs_110 | 9 | 0 | 0 |
| n4_inputs_00_01_10_11_outputs_0001 | 9 | 0 | 0 |
| n4_inputs_00_01_10_11_outputs_0110 | 3 | 0 | 0 |
| n4_inputs_00_01_10_11_outputs_0111 | 9 | 0 | 0 |
| n4_inputs_00_01_10_11_outputs_1001 | 3 | 0 | 0 |
| n4_inputs_00_01_10_11_outputs_1110 | 6 | 0 | 0 |

其中 keep 细分：`keep_valid_consistent` 293 条，`keep_valid_minor_caveat` 7 条（见第 5 节）。7 条 uncertain 分属 7 个 reason code：继承的虚构格式规则 1、错误 depth 说法 1、idempotent padding 1、duplicate-collapse plan 1、duplicate-collapse target 1、plan-candidate conflict 1、selection-expansion conflict 1（其中 duplicate-collapse 两项同属 `n3_inputs_00_01_10_outputs_011` round 6 的不同阶段）。2 条 reject 为 idempotent novelty claim 1 与 collapsed step as new 1。

推荐训练集为 `train.filtered.jsonl`（300 条）；`rejected.jsonl` 和 `uncertain.jsonl` 均不纳入。

## 4. 主要问题

### 4.1 n2_inputs_00_01_outputs_01 round 13（2 reject + 1 uncertain）

`sample_id`：selection `cd2787b0e65552116763`，expansion `1bdda431b8ca19a2967c`，candidate `5d9f6dcea0146798387f`。

selection 目标原文：

> “Instantiate the non-y operand as the bare literal y, giving the shallow form y AND y (which idempotence collapses to y); this is a different structural path…”

expansion 目标原文：

> “both root AND operands are the same literal y and idempotence collapses the root AND.”
> “Expected difference: This yields a shallow variable-collapse path…”

问题：目标自己承认 `y AND y` 在 idempotence 下折叠成 `y`，却把折叠写成 “different structural path”。按本项目结构评分，这不是新结构，而是已有字面量 `y` 的填充写法；训练它会教模型用重复操作数伪造 novelty。selection 和 expansion 因此判 **reject**。

candidate `Expression: y AND y` 在评分器里确实得到 key `y`，`y` 是标准解之一，所以答案本身有效；但它是 padding，且输入 plan 是退化的折叠计划。判 **uncertain**，隔离而不推荐。

### 4.2 n3_inputs_00_01_10_outputs_011 round 6（2 uncertain）

`sample_id`：expansion `d67677b63deb19b4fd4b`，candidate `d5f57c7036c750631e1c`。

expansion 说过滤器 “excluding ... duplicate-collapse cases ignored as already normalized”，但 “Expected difference” 又选择 “the both-plain-NOR(x,y) configuration”，candidate 正是：

> `Expression: NOT ((NOT ((x) OR (y))) OR (NOT ((x) OR (y))))`

两个 `NOR(x,y)` 操作数完全相同；评分会把它折叠成 `NOT(NOR(x,y))`，这是标准解中的新 key，所以 candidate 不算错答案；但监督文本是重复填充，且与 expansion 的“排除 duplicate-collapse”措辞相冲突。两者判 **uncertain**。

### 4.3 n3_inputs_00_01_11_outputs_101 round 3 candidate（1 uncertain）

`sample_id`：`a76f84280a525227c8a7`。

expansion 的 next step 要求：

> “allowing NOT variables only as direct conjuncts of the top-level AND or inside a primitive NOR gate.”

但 candidate 是：

> `Expression: (NOT y) OR (x AND y)`

它是 OR-root，`NOT y` 是 OR 的直接操作数，和“top-level AND / NOT 只能作直接合取项”的下一步约束冲突。该表达式本身是标准解，但训练对会教模型无视 plan；判 **uncertain**。selection/expansion 本身保留（见 decisions）。

### 4.4 n3_inputs_00_10_11_outputs_101 round 5 expansion（1 uncertain）

`sample_id`：`adfe8a5a6c957ae00c87`。

selection 与 review 明确说：

> “A depth-2 NOR-root construction … rather than another OR-root arrangement.”

expansion 却选择：

> “Restrict the next child expansion to a two-level OR-root with … NOT x and … y”

这个 plan 本身是合法的新结构，但其理由直接违背它所依据的 review/selection。判 **uncertain**；candidate 与该 plan 一致且是有效新 key，保留。

### 4.5 n2_inputs_00_01_outputs_01 round 12 expansion（1 uncertain）

`sample_id`：`afc9496f7d1cc10f2244`。

目标原文：

> “avoiding the failed depth-3 structure”

输入 review 把失败候选 `NOT (x OR y) AND NOT x` 解释为 “depth 3”。但该题 `datasets/balanced/boolean.json` 的 operators 含 `NOR`，`NOT(x OR y)` 在评分器里是 primitive NOR（depth 1）；`NOT(x OR y) AND NOT x` 的 key 实际在生成空间内、primitive depth 2。它被拒是因为 **key 不在该题 14 个标准解中**，不是因为 depth 3。expansion 的 plan 本身（NOT over NOR(x,A)）合法，但复述了错误的计层理由；判 **uncertain**，避免把错误 NOR 计层教给模型。

### 4.6 extended_n1_inputs_00_outputs_1 round 33 selection（1 uncertain）

`sample_id`：`361642bcb7a2d797d203`。

目标原文：

> “using fully balanced binary grouping”
> “Avoid: … especially unparenthesized flat OR cascades”

这是继承了输入 review 的说法：“formatting rules still require explicit balanced grouping”。项目任务没有这条规则；失败候选 `(x AND y) OR NOT x OR y` 的零奖励是因为其 key 不在 14 个标准解中（我们已在 `boolean_extended.json` 上核对），不是括号格式问题。branch 选择本身合法，但该 Avoid/Reason 虚构了约束；判 **uncertain**。

## 5. 边界样本（保留，但报告 caveat）

这些样本的 candidate 都在标准解集合中，未达到 reject/uncertain 的门槛，但值得复核：

- **n4_inputs_00_01_10_11_outputs_0001 round 0**（expansion `6bfede0a03032af9a931`，candidate `1af854a5788382449714`）：用 `NOR(x,x)`/`NOR(y,y)` 作为 `NOT x`/`NOT y` 的编码；idempotence 会折叠这些子项。但整体 key `NOR(NOT x, NOT y)` 是该题的标准解且此前未出现，review 也明确指出了 padding，下一轮改用直接 AND。保留并标 caveat。
- **n2_inputs_00_01_outputs_01 round 10 expansion**（`e59962d8b321089d4867`）：selection 建议 root-AND，expansion 选择 `NOT (NOT y)`。这是合法新 key，但与 selection 的具体建议不一致；保留并标 caveat。
- **n3_inputs_00_10_11_outputs_101 round 1 expansion**（`6e26a600ff428b71f749`）：selection 建议 AND/NOT-root，expansion 选择 OR-root 的新操作数组合；结构合法，保留并标 caveat。
- **n3_inputs_00_10_11_outputs_101 round 5 candidate**（`3eb5f31a37b9ed697827`）：candidate 本身有效且遵守其直接 next step，但该 expansion 与前面的 selection 冲突；保留 candidate，标 caveat。
- **extended_n1_inputs_00_outputs_1 round 33 expansion/candidate**（`e2f497b99547c8e86987` / `9987434827a5b0318e16`）：selection 含虚构的格式约束；expansion 的 operand set 和 candidate `(y) OR (NOT y)` 本身合法且是标准解，保留并标 caveat。

这 7 条 `keep_valid_minor_caveat` 仍在 `train.filtered.jsonl` 中。如果采取更严格口径，可把它们也移入 uncertain；本轮没有这样做，因为它们的候选目标都有效、结构核对通过，问题只是上下文或启发式措辞。

## 6. 不构成删除理由的情况

- **De Morgan / 真值表等价但结构不同**：例如 `NOT (x AND y)` 与 `NOT x OR NOT y`、`x OR (x AND y)` 与 `x`。项目明确不做 De Morgan/吸收归一化，这些是不同 mechanistic key；未按真值表等价删除。
- **多解题只找到一个**：所有 candidate 都是某一标准解；没有因“没找全”删除。部分 selection 的 “avoid” 语言是启发式，不等于已证明穷尽。
- **3D 高度组合重复**：`g3_h3_b3_006` 的 27 个高度三元组全部不同，且每个候选都通过 top-view/support/height 检查；未按“看起来都是 3 层”删除。
- **与评测题重合**：按用户安排不在本次删除；评测阶段做减法。

## 7. 局限与复核建议

1. 这是**模型语义审查**，不是形式证明，也没有跑训练。不能保证 300 条 keep 的每一个推理细节都正确；能保证的是：309/309 条都被实际读过（没有因预算截断而留下未审查范围），每条 candidate 都做了结构/标准解核对，发现的具体问题都给了可定位的引用。
2. 输入 review 本身可能有错（例如 4.5、4.6 的错误计层/格式说法）。本次以数据集、评分代码和实际目标为准，没有把 review 的说法当作事实；只有当 assistant 目标复述或依赖这些错误时才标 uncertain。没有因为某条记录的 review 有错就删除该记录下所有目标，因为 review 是上下文而不是监督答案。
3. `uncertain` 7 条和 `reject` 2 条建议人工复核后再决定是否永久排除；`train.filtered.jsonl` 只是“按本轮判定可训练”的 300 条，不是对推理正确性的认证。
4. 若之后修改 prompt、`stage_sft.py` 或 `BUILDER_VERSION`，应重新审查；尤其应加两条自动检查：candidate 是否满足 expansion 的 next step、以及 idempotence 折叠后是否与已出现 key 重复。
5. 仍未做 tokenizer/loss mask 实测；这与本次语义审查是两件事。

## 8. 文件与校验

| 文件 | 条数 | SHA256 |
|---|---:|---|
| 输入 `train.jsonl` | 309 | `23d181e2661a0b6e81b59acdd010757a897952df064c870f5a3c6320c65b58e2` |
| `decisions.jsonl` | 309 | `661da6b6a681aff8af5d0c1abb40d6135ed119f00e4154f9b56de772267f89a8` |
| `train.filtered.jsonl` | 300 | `00ce136d3066659e28ae2123265711160546ed68a004babebbbd6f0ab57542f2` |
| `rejected.jsonl` | 2 | `107d58c5beea2b280f56a207532fa4229e57ffe15331e9130c66492240fc1bc7` |
| `uncertain.jsonl` | 7 | `293f0c1fba3a3a0fb80b117e339e050082407d768ac70c2bb165951f3cca7e7d` |
| `manifest.json` | — | `e0fdaa2af5dea1d4b9485516683921643bc510e652fcb3b1369c5b1ebbd1d69f` |

`build_filtered.py` 已验证：309 个 `sample_id` 全覆盖、无重复、无多余；三个输出按原顺序、原 JSON 行字节输出（逐行 byte-identical），源文件未被修改，ledger 未被读写。
