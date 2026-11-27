# training/ — 跑题结果 → 训练样本

设计、不变量、扩展步骤见 **[`docs/TRAINING_DATA_PIPELINE.md`](../docs/TRAINING_DATA_PIPELINE.md)**。改任何逻辑前先读它。

```
canonical.py   同一任务的多次运行 → sources/<task>-canonical.jsonl(每题取最长快照,校验前缀)
stage_sft.py   规范化记录 → data/<dataset>/{train,valid}.jsonl(阶段输入重建 + 校验 + 标记)
ingest.py      ingest/ledger.jsonl(append-only,按 sample_id 去重)
```

常用命令(仓库根目录):

```powershell
python -m training.canonical --task ds22-sampled19 --out training/sources
python -m training.stage_sft --canonical training/sources/ds22-sampled19-canonical.jsonl `
  --out training/data/ds22-extended-n1 --cases extended_n1_inputs_00_outputs_1 --append-ledger
python -m training.ingest --check training/data/ds22-extended-n1/train.jsonl
python -m unittest tests.test_training_pipeline -v
```

红线:

- `results/ds-stage22-*` 只作存档,不是数据源;新增运行要进 `canonical.py` 的 `runs`。
- 评测题(27 中等、19 评测运行)不得进入训练。
- `sources/`、`data/`、`ledger.jsonl` 是生成物,不可手改;逻辑变更必须提升 `BUILDER_VERSION`。
- 样本带 `input_reconstructed=true` / `plan_unverified=true`:教师中间步骤未认证。
