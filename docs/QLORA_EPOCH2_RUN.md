# 上半数据 A：独立 2 epoch 训练（2026-11-26）

用户已批准：查看原版未微调 MCTS 2.2 进度，然后开始当前 A 数据的 2 epoch 训练。本次只启动训练，不自动开始效果评测或继续第 3 轮。

## 训练配置

- SSH：`EEC430004-4090`，双 RTX 4090；只使用此前释放的训练卡。
- 持久目录：`/opt/data/private/ECE4300-train-20261126/ga16-epoch2-v1/`。
- 后台控制进程启动 PID：25775；入口 `run.py`。
- 使用上一次验证通过的冻结脚本与同一批 309 条 A 数据，启动前核对 SHA256；数据 SHA256 `23d181e2661a0b6e81b59acdd010757a897952df064c870f5a3c6320c65b58e2`。
- 从原始 Qwen3-8B 初始化，**不加载上一次 LoRA，不恢复上一次 checkpoint**。
- 2 epoch，预计 40 次参数更新；batch 1，梯度累积 16，lr 1e-4，r 16 / alpha 32，NF4 双重量化，BF16，SDPA，梯度检查点，max_length 14336。
- 保留已经修复的 `trainer.model_accepts_loss_kwargs=False`；不重复运行已通过的冒烟。
- 每轮保存 checkpoint；旧独立 1 epoch 目录 `ga16-epoch1-v1/` 完整保留。

## 状态与产物

- 训练日志：`train.log`；控制日志：`runner.log`；启动参数与文件 SHA256：`run_manifest.json`。
- `status.json` 的 `training` 表示正在训练；`validating_saved_adapter` 表示正在检查保存参数；`completed` 才表示 epoch=2 / global_step=40 且适配器全部有限、LoRA B 均非零；`failed` 保留错误原因。
- 最终适配器：`adapter-epoch2/`；检查点预计为 `checkpoint-20/`、`checkpoint-40/`。
- 权重和环境保留服务器；本地只备份轻量启动资料与日志。
- 注意：两轮训练计划的第一轮学习率轨迹不同于此前独立单轮计划；`checkpoint-20` 与独立 `ga16-epoch1-v1` 不是同一权重。

## 原版未微调 MCTS 2.2 启动前快照

口径：Qwen3-8B、自部署、MCTS 2.2 / isolated_examples_v1、思考开启，19题共358参考解。结果目录为一号机 `/opt/data/private/ECE4300 Final Project/results/qwen-stage22-protocol-examples19-20261126/`。

- 完成 14/19；另 5 题进行中。
- 已找到参考解 135/358，按解加权覆盖率 37.71%；按题平均覆盖率 52.42%。包含未完成题当前找到的解，不代表最终成绩。
- 未完成题：因果 `n6_hard_029` 14/57、`n6_hard_021` 13/63；3D `g3_h3_b4_001` 29/81；布尔 `n1_inputs_01_outputs_1` 9/27、`n1_inputs_10_outputs_1` 11/27。
- 完成14题合计59/103参考解。原版推理任务未暂停、未更换模型或后端。
