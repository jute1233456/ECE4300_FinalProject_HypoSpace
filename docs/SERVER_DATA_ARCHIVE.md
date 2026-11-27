# 服务器数据本地归档（2026-11-27）

目标：把要进 git 的微调数据/参数/评测摘要从 `EEC430004-4090` 汇总到本地；模型权重不下载。当前是**快照**，任务仍在跑，完成后再更新。

## 本地位置

- 可提交 git 的紧凑归档：`training/server-archive-20261127/`（13 MB，121 文件，无权重/分词器/原始大结果）
- 原始数据本地备份（`.gitignore` 已忽略，不进 git）：`local-backups/server-full-20261127/`
  - quick3 完整非 tree 结果 22 MB
  - DPO 数据集全量（含 responses/states/contexts）7.6 MB
  - 紧凑包 tar 1.4 MB
- 导出脚本：`scripts/export_server_archive.py`（在服务器上执行，生成 `/tmp/ECE4300-archive-20261127`）

## 紧凑归档内容

- `training/dpo-v1|v2|v3/`：config、data/dpo.jsonl+manifest、status、pipeline 日志（`.txt`）、adapter/smoke 的 status/history/adapter_config（不含权重）。
- `training/sft/ga16-epoch1|epoch2/`：运行参数、manifest、status、日志、校验代码；训练数据不重复，本地已有同 SHA 文件。
- `training/sft/data-length-audit-v1/`：长度统计、压缩模拟、日志。
- `datasets/v1|v2|v3-additive/`：DPO 数据、judgments、pair_manifest、skipped、manifest、README、checks、SHA。
- `datasets/increments/v2|v3/`：评审包、标签映射、快照 manifest。
- `eval/`：quick3 summary/tasks/config/state + 80 个任务的紧凑 `eval-index.jsonl`（found/used/call_cap、stop、routing 元数据、结果 SHA）。

## 明确排除

- 所有 `*.safetensors`（adapter、smoke、checkpoint）和分词器文件（`tokenizer.json`、`vocab.json`、`merges.txt` 等）。
- 采集原始 `responses/`、`states.jsonl`、`contexts/`。
- quick3 `*.tree.json` 与完整逐轮 result JSON（只在 `local-backups/server-full-20261127/` 保留压缩包）。

## 更新方法（任务跑完后）

1. 在服务器执行导出脚本（仓库内 `scripts/export_server_archive.py` 可 scp 到 `/tmp/`）：
   ```bash
   python3 /tmp/export_server_archive.py
   cd /tmp && tar czf ECE4300-archive-20261127.tar.gz ECE4300-archive-20261127
   ```
2. 本地重新下载并解压到 `training/server-archive-20261127/`（先清空旧解压内容，保留目录）。
3. 同步原始备份：
   ```bash
   tar czf /tmp/ECE4300-quick3-results-full-20261127.tar.gz --exclude='*.tree.json' \
     -C /opt/data/private/ECE4300-controller-20261127/quick3 results
   tar czf /tmp/ECE4300-dpo-datasets-full-20261127.tar.gz \
     -C /opt/data/private ECE4300-dpo-datasets-20261127
   ```
   下载到 `local-backups/server-full-20261127/` 覆盖旧包。
4. 更新 `training/server-archive-20261127/README.md` 的快照时间/统计，并同步修改 `AGENTS.md` 的归档小节。

## 触发更新的任务

- SFT ep1：仅剩混合题库 `n1_002`（在途）。
- DPO v3 混合组：25 题（原 quick3 3 + mixed100 22）在跑。
- DPO v3 全阶段对照组：3 题在跑。
- DPO 原始采集：`stopped`（PAUSE），1107/2084；若恢复并新增完整状态，需要先做下一批离线评审，再更新 `datasets/`。

在以上任务结束前，本归档只代表 2026-11-27 21:00 UTC+8 的快照。
