# 服务器数据归档（2026-11-27 21:00 UTC+8 快照）

来源主机：`EEC430004-4090`。本目录是**可提交 git 的紧凑归档**：训练数据/参数/日志、DPO 数据集文本、评测摘要。模型权重、分词器、原始候选响应和大型逐轮结果 JSON 不在本目录。

## 内容

- `training/dpo-v1|v2|v3/`
  - `config.json`、`status.json`、`pipeline.log.txt`、`data/dpo.jsonl`、`data/manifest.json`
  - `adapter/status.json`、`adapter/history.json`、`adapter/adapter_config.json`
  - `smoke/status.json`、`smoke/history.json`、`smoke/reference_logps.json`
- `training/sft/ga16-epoch1-v1|ga16-epoch2-v1/`
  - 运行参数、manifest、status、训练日志、校验代码；`train.jsonl` 未重复（与本地 `training/data/` 同 SHA）。
- `training/sft/data-length-audit-v1/`
  - `lengths.json`、`compact-simulation.json`、长度日志；1024 条训练数据在本地 `training/data/ds22-sft-all-20261126/train.jsonl`（SHA `c50dfcfc…`）。
- `datasets/`
  - v1：59 对；v2：133 对；v3-additive：160 对（130 基线 + 30 新增）。
  - 每个含 `dpo.jsonl`、`judgments.jsonl`、`pair_manifest.jsonl`、`skipped.jsonl`、`manifest.json`、`README.md`、`checks.json`、`SHA256SUMS`。
  - `increments/v2|v3/`：评审包、标签映射、快照 manifest。
- `eval/`
  - `quick3-summary.json`、`quick3-tasks.json`、`quick3-config.json`、`quick3-state.json`。
  - `eval-index.jsonl`：80 个任务的紧凑结果（found/used/call_cap、stop/completed、routing 元数据、结果文件 SHA）。
- `manifest.json`：本地最终校验（服务器导出后重命名日志、加入 README/SHA）的 size/SHA256；排除项说明。

## 本次未纳入（体积/权重）

- 模型权重：`*.safetensors`（DPO adapter、smoke、SFT checkpoint）。
- 分词器文件：`tokenizer.json`、`vocab.json`、`merges.txt` 等。
- 原始采集响应/`states.jsonl`/`contexts/`、quick3 `*.tree.json` 和完整逐轮 result JSON。

完整原始备份（不进 git）在本地：

- `local-backups/server-full-20261127/ECE4300-quick3-results-full-20261127.tar.gz`（22 MB，sha `ebee90ca…`）
- `local-backups/server-full-20261127/ECE4300-dpo-datasets-full-20261127.tar.gz`（7.6 MB，sha `d2257863…`）
- `local-backups/server-full-20261127/ECE4300-archive-20261127.tar.gz`（紧凑包，sha `51545061…`）

## 这是快照，不是最终数据

导出时仍在运行：

- SFT ep1 只剩 `n1_002`（混合题库）；
- DPO v3 混合组 25 题（原 3 + mixed100 22）在跑，快照时仅少量完成；
- DPO v3 全阶段对照组 3 题在跑；
- 采集器 `stopped`（PAUSE），1107/2084。

**这些任务完成后需要重新导出**：更新 `eval/`、`datasets/`（如有新评审）、`training/dpo-v3/` 的 history/status；采集若恢复并新增完整状态，还要做下一批 DPO 数据。导出脚本与步骤见 `docs/SERVER_DATA_ARCHIVE.md`。
