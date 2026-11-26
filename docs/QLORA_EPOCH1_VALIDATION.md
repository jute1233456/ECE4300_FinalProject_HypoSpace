# QLoRA 首轮运行验证（2026-11-26）

目的：修正梯度累积归一化，验证一个 epoch 能正常训练和保存。不是微调效果评测；不自动运行后续 epoch，不操作其他推理任务。

## 修复与验证

- 实际训练环境：Transformers 4.57.6，四号机双 RTX 4090，SSH `EEC430004-4090`。
- 自定义 `compute_loss` 返回每个微批的 assistant token 平均交叉熵，没有使用 `num_items_in_batch`。Qwen 的 `forward(**kwargs)` 使 Trainer 自动认为损失已做跨累积批次归一化，旧实现因而漏除累积次数。
- 修复：创建 Trainer 后设置 `trainer.model_accepts_loss_kwargs = False`，由 Trainer 根据实际 `current_gradient_accumulation_steps` 归一化，包括最后不足 16 条的尾批。
- `server/validate_qlora_loss.py` 提取生产脚本中的真实 AssistantOnlyTrainer 类，在 CPU 上数值检查：裁剪 logits 前后的损失和参数梯度一致；旧累积 16 的梯度为预期 16 倍；修复后累积 16 和尾批 5 都与参考梯度一致。
- 双卡真实模型冒烟：309 条原始训练数据，累积 16，2 次参数更新，成功退出并保存适配器。loss 2.6062 → 1.5711；运行 56.64 秒；峰值已分配显存 GPU0 6.07 GiB、GPU1 9.57 GiB。该短测试不代表全数据峰值，也不证明泛化提升。
- 经用户授权，Claude Code 对这两份脚本进行了只读复核；Codex 检查其结论并补充了完整 `Trainer.train()` 的 CPU 集成测试。21 条样本自动形成 16 + 5 两个窗口，检查到 2 次更新和 epoch=1，已通过。日志 `tail-integration.log`。
- Claude 提到未来长答案仍可能造成 CE 显存占用。当前数据的 assistant 最长长度是 candidate 69、expansion 281、selection 238 token；本次无需增加截断或修改损失算法。以后扩充长答案数据时重新测试。

## 首个 epoch

- **已完成**：北京时间 2026-11-26 17:13:27，`epoch=1.0`、`global_step=20`。训练计时 596.45 秒（约 9 分 56 秒），平均训练 loss 1.20968。完整数据峰值已分配显存 GPU0 10.76 GiB、GPU1 16.80 GiB；无 OOM。仅证明流程运行成功，不代表解题覆盖率提高。
- 保存后重载验证通过：504 个 adapter tensor、43,646,976 个参数全部有限，252 个 LoRA B tensor 均已更新为非零；重新加载到 Qwen3-8B 后，一次前向计算输出形状 `[1, 1, 151936]` 且全部有限。记录 `adapter-validation.json`。未运行第二个 epoch。
- 持久根目录：`/opt/data/private/ECE4300-train-20261126/ga16-epoch1-v1/`。
- 后台控制进程 PID：24846；入口 `run.py`。顺序执行 CPU 校验、累积 16 冒烟、独立从基础模型训练一个 epoch；任何阶段非零退出均停止后续步骤。
- 参数：309 条，batch 1，累积 16，lr 1e-4，LoRA r 16 / alpha 32，七种线性层，NF4 双重量化，BF16，SDPA，梯度检查点，max_length 14336，无样本截断。
- `device_map=auto` 是单进程模型分片，不是两份数据并行；有效批大小 16，预计 20 次参数更新。
- 当前状态以服务器 `status.json` 与 `epoch1.log` 为准。`phase=epoch1` 代表正在运行；`phase=completed` 才代表训练成功并检查到 epoch=1 / global_step=20。`phase=failed` 保留失败阶段与错误信息。
- 训练日志：`epoch1.log`；启动日志：`runner.log`；数据和脚本 SHA256：`run_manifest.json`。
- 输出：`adapter-epoch1/`，包含适配器、tokenizer 与 epoch 检查点；权重留在服务器。

## 本地记录

轻量启动脚本、SHA256 清单、CPU 校验与冒烟日志位于 `local-backups/training/ga16-epoch1-v1/`。此目录不包含模型或虚拟环境。首次启动时冻结的服务器验证脚本尚未包含后来增加的“检查生产代码确实设置 False”静态断言；数值检查内容与本地一致。

训练数据按用户决定保留所有已完成题；评测时从测试集排除重合和等价题。此次不修改数据与评测规则。
