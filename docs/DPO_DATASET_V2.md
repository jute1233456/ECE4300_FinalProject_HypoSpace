# simple-reflection DPO 数据集 v2（2026-11-27，补全版）

v2 是 v1 的**超集**：v1 的 59 对逐字保留，新增 74 对。v1 目录与压缩包不改动，便于复现。
只做离线数据构建与评分，不修改采集程序、不暂停任务、不调整 GPU、不启动 DPO 训练。

## 产物

- 服务器（`EEC430004-4090`）：
  - 增量冻结快照：`/opt/data/private/ECE4300-dpo-datasets-20261127/simple-reflection-v2-increment/`
  - 合并数据集：`/opt/data/private/ECE4300-dpo-datasets-20261127/simple-reflection-dpo-v2/`
  - 打包：`/opt/data/private/ECE4300-dpo-datasets-20261127/simple-reflection-dpo-v2-20261127.tar.gz`
    （sha256 `db7891c934611add149c2d7d52f7ca3fb7b2d44a784e00bfa19c4262bed53882`）
- 本地副本：`local-backups/handoffs/simple-reflection-dpo-v2-20261127.tar.gz`（同一 sha256）。

## 增量范围

采集在 2026-11-27 暂停时共有 **148 个“四份候选完整 + DS 历史回答”状态**；v1 覆盖 62 个，剩余 **86 个**全部纳入本轮：

- 增量 86 状态、344 份 qwen 候选，全部 `finish_reason=stop`、`final_text` 非空、无 thinking。
- 其中 9 个状态属于 v1 已有题族，另 4 个题族首次入库：
  `n3_inputs_00_01_10_outputs_011`、`n3_inputs_00_01_11_outputs_101`、
  `n3_inputs_00_10_11_outputs_101`、`n4_inputs_00_01_10_11_outputs_0001`。
- 评审仍为匿名 A–E：4 份 ep1 候选 + 1 份 DS 历史回答；判读标准与 v1 相同。
- 增量判读：helpful 185、harmful 239、uncertain 6。

## 合并结果

- 148 个状态：**133 对，15 个跳过**。
- 按域：布尔 53、3D 49、因果 31。
- chosen：DS 118、qwen 15；rejected：qwen 133。
- 判读合计：helpful 311、harmful 379、uncertain 50（148 × 5 = 740）。
- 增量 12 个跳过：
  - 8 个“五条里没有高置信 harmful / 只有 medium helpful”；
  - `32e616cae407`、`ace352e07b8e`、`10f69639de9c`、`922620a7b2b7`、`bec75647e64d` 等为 high/high 条件不满足；
  - 其中 `bec75647e64d` 是原判 chosen medium，构建时按 high/high 规则自动降为跳过。

## 格式

与 v1 完全一致：

```json
{"prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
 "chosen": [{"role": "assistant", "content": "原始 Reflection 文本"}],
 "rejected": [{"role": "assistant", "content": "原始 Reflection 文本"}]}
```

- `prompt` 与冻结 `states.jsonl` 的 `messages` 逐字一致；chosen/rejected 逐字取自冻结文件。
- 无 train/validation/test 划分字段；下游如需划分，按 `family_id`/`case_id` 分组。
- 文件：`dpo.jsonl`（133）、`judgments.jsonl`（148）、`pair_manifest.jsonl`（133）、
  `skipped.jsonl`（15）、`manifest.json`、`README.md`、`checks.json`、`SHA256SUMS`。

## 已完成检查

- 148 个状态各 5 条判读；133 对全部 chosen=helpful/high、rejected=harmful/high。
- 133 个 prompt 与 v1/增量冻结 `messages` 完全一致；chosen/rejected 的 sha256 与 manifest 一致。
- 文本互异、非空、无 `<think>`；无 split 字段；v1 的 59 对逐字未变。
- 增量标签映射由 `review/labels.json` 固化，状态前缀唯一解析。

## 复现

```bash
cd /opt/data/private/ECE4300-dpo-datasets-20261127
python3 make_increment.py      # 重建增量快照与评审包（不会改动采集目录）
python3 build_merged.py        # 读取 v1 + 增量判读，重建 simple-reflection-dpo-v2/
```

## 边界

- v1 目录、v1 压缩包、采集器、队列、暂停标记、GPU 角色均未改动。
- 未启动 DPO 训练。
- 这套数据仍以“DS 反思 > ep1 反思”为主（chosen 中 DS 118/133），属于蒸馏式偏好集；不能据此声称 ep1 检查点变好。
