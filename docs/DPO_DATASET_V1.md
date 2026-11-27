# simple-reflection DPO 数据集 v1（2026-11-27）

只做离线数据构建与评分，不修改采集程序、不暂停任务、不调整 GPU、不启动 DPO 训练。
采集进程与九卡池状态未受影响。

## 产物位置

- 服务器（四号机 `EEC430004-4090`）：
  - 冻结快照：`/opt/data/private/ECE4300-dpo-datasets-20261127/simple-reflection-v1/`
  - 数据集：同目录 `dataset/`（`dpo.jsonl`、`judgments.jsonl`、`pair_manifest.jsonl`、`skipped.jsonl`、`manifest.json`、`README.md`、`checks.json`、`SHA256SUMS`）
  - 打包：`/opt/data/private/ECE4300-dpo-datasets-20261127/simple-reflection-dpo-v1-20261127.tar.gz`
    （sha256 `c9baf1eba5bede2e3135f867a6322cdefefa5c86ce0f2be57507aa2436c63d28`）
- 本地副本：`local-backups/handoffs/simple-reflection-dpo-v1-20261127.tar.gz`（同一 sha256）。

## 快照

- `snapshot_utc = 2026-11-26T23:23:05Z`，`states.jsonl` 521 行（sha256 `b93d1be0…`），262 份候选响应。
- 冻结后不再改动快照内文件；构建脚本与判读文件放在 `build/`，输出放 `dataset/`。
- 每个被评审状态 = 4 份 `sft-pruned-v1-ep1` 完整候选（`response.final_text`）+ 1 份 DS 历史反思回答（`ds_answers[0].text`）。
- 62 个状态满足“四份候选完整”，全部进入评审；`review/index.json` 记录 A–E 与 qwen/DS 来源类型。

## 判读规则

- 标签 A–E 匿名化，离线逐条通读，对照 TASK、HISTORY、观测与约束。
- 每条回答给 `helpful / harmful / neutral / uncertain`、原文证据、简短解释、置信度。
- harmful 的典型依据：建议已出现过的边集/候选、候选违反观测或深度/支撑约束、虚构规则、误读 reward、造成环或删除必需边、复制无关模板示例。
- 每个状态最多一对：chosen = helpful 且 confidence=high，rejected = harmful 且 confidence=high；其余情况跳过。
- 62 个状态中：59 对，3 个跳过（`0aa97f86…`、`eb2a6903…` 无高置信 harmful；`d2daaf38…` 的候选对未达到 high/high）。
- 判读总计：helpful 126、harmful 140、uncertain 44、neutral 0。

## 数据构成（重要）

- 59 对：布尔 22、3D 24、因果 13。
- chosen：DS 50、qwen 9；rejected：qwen 59。
- 因此这套数据主要在学习“DS 反思 > ep1 反思”，而不是“ep1 反思 > ep1 反思”；属于蒸馏式偏好集，不能据此声称 ep1 检查点本身变好。
- DS 回答来自重建输入（`input_reconstructed=true`），已逐条核对 `contexts/<sid>.json` 的 `original_input` 与 `states.jsonl` 的 `messages` 完全一致；但生成时不是在线输入，provenance 弱于 qwen 候选。
- DS 回答是原始 DS 运行中记录下来的那一次续写，不是按 ep1 协议重新采样的候选，可能受该运行分支影响；应视为教师式目标，而不是同策略的等概率样本。

## 输出格式

`dataset/dpo.jsonl`（59 行，训练器可直接用）：

```json
{"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "chosen": [{"role": "assistant", "content": "原始 Reflection 文本"}],
 "rejected": [{"role": "assistant", "content": "原始 Reflection 文本"}]}
```

- `prompt` 与 `states.jsonl` 的 `messages` 逐字相同；chosen/rejected 逐字取自冻结文件，未拼接 thinking、参考解或任何后缀。
- 没有 train/validation/test 划分字段。若下游需要划分，按 `family_id`/`case_id` 分组，避免同题泄漏；评测时仍按既定规则对测试集做减法。

其余文件：

- `judgments.jsonl`：62 行，含每个标签的判读、证据、置信度、解析后的来源（文件名/`sample_id` 或 DS 的 `record_index`）。
- `pair_manifest.jsonl`：59 行，含 chosen/rejected 标签、来源、文本与 prompt 的 sha256、配对理由。
- `skipped.jsonl`：3 行，跳过原因。
- `manifest.json`：快照指纹、统计、格式说明、检查项。
- `README.md`：面向使用者的说明。

## 已执行的检查

- 62 个 packet 状态的 A–E 文本与冻结响应/DS 文件匹配（归一化前缀相似度 ≥0.95），来源类型与 `review/index.json` 全部一致。
- 59 对全部满足 chosen=helpful/high、rejected=harmful/high、文本不同、prompt 为 system+user 且非空。
- dpo 行仅含 `prompt/chosen/rejected`；chosen/rejected 的 sha256 与 pair_manifest 一致；无 `<think>`。
- 50 个 DS-chosen 的 prompt 与冻结 `messages` 完全一致。

## 复现

```bash
cd /opt/data/private/ECE4300-dpo-datasets-20261127/simple-reflection-v1
python3 build/build_dataset.py     # 读取快照 + build/judgments，重建 dataset/
```

`build/judgments/judgments-01..08.json` 是逐包人工判读结果；构建脚本会做全部断言，失败即退出，不写半成品。

## 边界

- 未启动 DPO 训练，未改动任何训练超参或适配器。
- 未修改 `reflection-v2-collection` 采集器、队列、暂停标记或 GPU 角色。
- 本数据集只覆盖 reflection 阶段，不含 selection/expansion/candidate 阶段，也不含最终答案偏好。
