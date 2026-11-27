# B 审查进度（已完成）

- 输入：`training/data/ds22-new4-20261126/train.jsonl`，507 条（169 轮 × 3 阶段），4 题：`n1_inputs_01_outputs_1`、`n1_inputs_10_outputs_1`、`n6_hard_021`、`n6_hard_029`。
- 结构复核：169/169 candidate 经离线评分器通过（布尔标准解集合；因果 63/57 参考图）。
- 主数据集：**487 keep / 0 reject / 20 uncertain**，见 `decisions.jsonl`、`train.filtered.jsonl`、`manifest.json`。
- 非命中轮次：45 轮可重建 → 90 条 selection/expansion，**21 keep / 69 uncertain**，见 `decisions-reasoning.jsonl`、`train.reasoning_only.jsonl`（带 `candidate_verified=false`，不并入主集）。
- 无效搜索量化：n1_01 806,532 tokens(40%)、n1_10 820,270(46%) 未命中；因果 1%/0%。主要原因是布尔两题反复提出三层深度结构。
- `g3_h3_b4_001` 已完成并补审：243 条 → 237 keep / 6 uncertain / 0 reject，见 `training/audits/B4-claude-review/`。DS 19 题全部完成，等权覆盖率 0.9903。
