# Reflection DPO v1（2026-11-27）

用户已授权删去一个有错误的正样本，立即开始 DPO，训练结束自动跑既有三题快速验证；训练和推理跑题优先于候选采集。

## 数据和模型

- 原始 simple-reflection-v1 的 59 对完整保留；独立训练副本 58 对。
- 删除 state_id `0258751fb1219ad03851023c7487dc7588493116c40ed48b8cc7ff3cfcf71516`，原因是 chosen 把 `y OR NOT x` 错判为深度 1。
- 训练数据 SHA256：`206b0458acb4e853164a9aea00eb439f9c1593479df420ebf46b12aa4932cee5`。
- 起点与 reference 均为 `sft-pruned-v1-ep1`，检查点 SHA256 `8fa112bf26333cd24ea3c2c3759cd5584453a2cce83e75ba8781f569e4bac63e`。不使用关闭 SFT 适配器后的 base 作 reference。
- 模型名 `dpo-reflection-v1-ep1`，Qwen3-8B + 原有 r16 LoRA 的 QLoRA 继续训练。
- 1 epoch，微批 1 对，累积 8 对，8 次更新，最后一批 2 对按 2 归一化；学习率 5e-6，warmup 1 次更新后反比例下降，beta=0.1，gradient clip=1，优化器 paged AdamW 8bit。
- 最大总序列 8192，不截断；仅计算 assistant completion 的序列 log probability，使用 sigmoid DPO。
- reference log probability 在任何更新前通过同一冻结 SFT 起点计算并缓存，无需同时加载另一个 8B reference 模型。关闭 dropout。首次策略与参考一致时 loss 应为 log(2)，程序检查误差。

## 自动衔接

SSH `EEC430004-4090`；持久根目录：

`/opt/data/private/ECE4300-train-20261127/dpo-reflection-v1-ep1/`

- `config.json`：配置和数据/起点指纹。
- `data/`：58 对训练数据、配对来源和删除说明。
- `pipeline.log` / `status.json`：预约、训练、归还、发布阶段。
- `smoke/status.json`：最长样本一次真实更新，独立进程、独立产物，不污染正式起点。
- `adapter/status.json`：正式训练 phase、global_step、loss、lr、实际 batch、梯度范数，完成后含显存峰值和权重 SHA。
- `adapter/reference_logps.json` / `history.json`：冻结 reference 与更新日志。
- `training-request.json`：针对指定 GPU 的采集让卡请求。

先预约 node4-4090-g0/g1，排空后由现有 run_reserved_training 停对应推理引擎、运行最长样本检查及正式训练，结束后 finally 自动归还推理。冒烟失败则不开始正式训练。训练和发布失败会记录 failed，不会假报 ready。

训练完成后向 node4 双4090和 node4 单5090发布同一个不可变适配器，检查 SHA 和真实 API 生成，再发布 ready 回执。已有 quick3 控制器从 future_models 自动加入同样三题（布尔12解、3D9解、因果24解，共45解），使用既有 MCTS 2.2 阶段隔离示例版、thinking、32768 请求额度。三个任务按原队列顺序接入，不重启既有跑题。

三题进度在：

`/opt/data/private/ECE4300-controller-20261127/quick3/`

检查 `tasks.json` / `summary.json` 中 `dpo-reflection-v1-ep1`，结果在 `results/dpo-reflection-v1-ep1/`。训练根的 ready 仅指发布成功，不代表三题完成。

## 采集优先级

采集器在每次候选请求前检查更高优先级的 pending 跑题任务，以及训练指定 GPU 的让卡请求。当前 API 完成先保存，随后归还卡，其余候选保持 pending，下次只补缺失部分；不打断在途 API。训练中尚未可用的 future model 不阻塞其他空闲卡采集。三题入队后优先于采集；如有已经开始的采集请求，等待该请求结束后归还卡。

旧采集进程收到 SIGTERM 后合作退出，已保存的数据保留，新版从原目录恢复。新 PID 读采集目录 state.json，不使用历史固定 PID。

## 验证

8 项采集测试通过，包含训练仅让指定卡、跑题 pending 时两次候选之间归还、已有候选不重复生成。CPU 数值检查通过：初始 DPO loss=log(2)、chosen/rejected 梯度方向、按真实组大小积累与组均值梯度一致。正式训练仍须以服务器实际 status.json 和日志为准。

权重和虚拟环境留服务器；本地仅保留代码、说明及轻量日志。

## 首次连续训练 OOM 与重启

首次最长单样本更新成功，但正式训练在完成第1次更新后、下一组 backward 时 OOM：GPU1 已分配16.98GiB、保留未分配4GiB，申请2.44GiB失败。双卡由封装自动归还，未发布不完整模型。

失败产物保存在 `smoke-failed-attempt1/` 和 `adapter-failed-attempt1/`。第二次从原 SFT ep1 重新开始，不接着不完整的DPO参数训练。改为先无梯度计算当前偏好差及损失对 chosen/rejected logprob 的系数，再分别重算、反传两个回答；关闭dropout且两次反传间不更新参数，因此链式法则梯度与原DPO相同。CPU双精度对照一致，同时启用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。新GPU检查及训练结果仍以本轮实际日志为准。
