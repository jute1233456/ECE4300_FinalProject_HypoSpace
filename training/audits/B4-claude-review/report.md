# B4 数据集语义质量审查（Claude Code）

- 日期：2026-11-26
- 输入：`training/data/ds22-b4-20261126/train.jsonl`，**243 条** = 81 轮 × 3 阶段（`g3_h3_b4_001`，3D，81 个参考结构）。
- 背景：该题最后完成（`all_recovered`，134 轮，覆盖率 1.0）；用户指示跑完后再处理，本轮补审。
- 方式：离线结构复核 + 逐条语义审查；未训练、未发 API/网络请求、未操作进程/服务器。

## 1. 结构复核

- `datasets/balanced/3d.json` 中 `g3_h3_b4_001` 有 **81 个**参考结构，与运行记录一致；
- 用交接包 `tools/offline_verifier.py` 对 **81/81 个 candidate** 重新打分：全部 `valid` 且在参考集内；
- 题面：3×3 网格、最大高度 3、俯视占用格为 `(1,1),(1,2),(1,3),(2,1)`。

## 2. 结果

| 决策 | selection | expansion | candidate | 合计 |
|---|---:|---:|---:|---:|
| keep | 80 | 77 | 80 | **237** |
| reject | 0 | 0 | 0 | 0 |
| uncertain | 1 | 4 | 1 | **6** |

## 3. uncertain 明细

1. **selection–expansion 冲突（4 条）**
   - r22：selection 要求"高度和 = 10"，expansion 改成"至少三列高 ≥2"；
   - r46：selection 要求精确向量 `(2,1,1,3)`，expansion 只要求 `h(1,3) ≥ 2`；
   - r105：selection 要求评估 `h=(1,1,1,2)`，expansion 改成 `h=(1,1,3,2)`；
   - r127：selection 要求高度和 = 7，expansion 改成 = 8。
2. **虚假完备（1 条）**：r25 expansion 用 `Plan: NONE` 声称分支已唯一固定 `(2,3,1,1)`、无新约束，但 selection 仍认为高度 2 的列未定，二者对树状态判断不一致。
3. **计划–候选冲突（1 条）**：r73 candidate 所在分支固定 height-3 列为 `(2,1)`，候选却是 `(1,1)/(1,2)` 三层、`(2,1)` 仅一层。

其余 237 条保留：Layer 1 固定为俯视占用格、逐层支撑、高度 ≤3，计划与候选一致，未发现把任意高度约束说成"观测强制"的硬错误。

## 4. 与 B（4 题）合并

| 数据集 | keep | uncertain | reject |
|---|---:|---:|---:|
| B（507 条，4 题） | 487 | 20 | 0 |
| **B4（243 条，1 题）** | **237** | **6** | 0 |
| 合计 | **724** | 26 | 0 |

另有非命中推理 `train.reasoning_only.jsonl` 21 条（B 审查产物），不含在 724 内。

## 5. 文件与校验

| 文件 | 条数 | SHA256 |
|---|---:|---|
| 输入 `train.jsonl` | 243 | 见 `manifest.json` |
| `train.filtered.jsonl` | 237 | `13f08bf9…` |
| `uncertain.jsonl` | 6 | `a4e513fb…` |
| `rejected.jsonl` | 0 | `e3b0c44…` |
| `decisions.jsonl` | 243 | 见 `manifest.json` |

`build_filtered.py` 已验证 243 个 `sample_id` 全覆盖、无重复、无多余；输出按原始顺序字节一致；源文件未改；243 条已追加到 `training/ingest/ledger.jsonl`。

## 6. 局限

- 模型语义审查，不是形式证明；未训练。
- 3D 的"高度剖面是否最优/是否真正新颖"仍属模型判断；本轮的判据是结构合法性、计划–候选一致性、是否把任意约束写成规则。
