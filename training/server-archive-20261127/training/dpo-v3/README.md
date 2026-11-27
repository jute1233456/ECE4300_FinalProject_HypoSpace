# DPO v3：只做反思，SFT ep1 完成搜索

## 新增：全流程 V3 对照（2026-11-27）

用户随后授权将全流程 DPO v3 也加入推理队列，对照原有混合实验。只新增同样 quick3 三题、45 个参考解，不重新训练，不修改或覆盖正在跑的混合结果。

- 混合组队列名：`dpo-reflection-v3-ep1`，仅反思用 V3，其余步骤 SFT ep1。
- 全流程组队列名：`dpo-v3-full-ep1`，反思、读树、选择、拓展、候选生成均使用实际 API 模型 `dpo-reflection-v3-ep1`；这是队列别名，不是另一份模型权重。
- 全流程组清单：`/opt/data/private/ECE4300-controller-20261127/quick3/full-v3-manifest.json`，与原 quick3 题目和参考解数量一致。
- 全流程组原始结果和日志：`/opt/data/private/ECE4300-controller-20261127/quick3/results/dpo-v3-full-ep1/quick3-v3-full/medium/`。
- CC 查统一 quick3 的 `tasks.json`、`summary.json` 时，用 `model=dpo-v3-full-ep1` 筛选全流程组。结果里的 `requested_model`、反思与搜索响应 model 都应为 `dpo-reflection-v3-ep1`，不能出现混合组的 `stage_models`。

通过现有 additional_suites 动态入队，控制器和在途任务未重启。数据生成继续服从现有训练/推理优先规则，在当前请求结束后让卡。部署前 9 项路由/协议测试通过，服务器 CLI 计划检查通过；这不代表三题已完成。

最初的混合组按用户要求指定 V3 负责反思、其他步骤使用 SFT；随后新增的全流程 V3 对照见上方。DPO v2 未新增混合实验。

## 冻结配置

- 实验：MCTS 2.2 / isolated_examples_v1，hybrid-reflection-v3。
- 反思模型：`dpo-reflection-v3-ep1`。
- 读树、节点选择、拓展和完整候选生成：`sft-pruned-v1-ep1`。
- V3 从原 SFT ep1 重新训练，reference 也为同一冻结 SFT ep1，不接着 DPO v2 训练。
- 数据：独立原始审查的 185 对；SHA256 `08a59b05bd2ee9293ec9dad15481caecf71d45845d9e43d044b1f44a558a2da9`。来源/手写记录见 `DPO_INDEPENDENT_RAW_REVIEW_V1.md`。
- 1 epoch，微批 1，累积 16，12 次更新，最后一批 9 对按实际数量归一化；峰值 lr=1e-6，warmup=2 更新，沿用反比例下降，beta=0.1，最长总序列 8192，无截断；r16/NF4/BF16 与旧版一致。
- 保持标准 DPO 目标。数据量和阶段分工已变化，本轮不再同时加入新正则损失。现有数据仍有长度/来源偏差，不声称已消除偏差。
- 正式训练每 3 次更新保存检查点，增加 chosen/rejected 相对 reference 的 logprob 变化、每 token NLL 与批次偏好准确率。它们不是独立验证集成绩，也不是完整 KL。

## 部署与自动衔接

SSH `EEC430004-4090`。

训练根：`/opt/data/private/ECE4300-train-20261127/dpo-reflection-v3-ep1/`。

冻结训练代码：`/opt/data/private/ECE4300-controller-20261127/dpo-v3-code/`。其 `server/train_dpo.py` 对应本地保留的 `server/train_dpo_v3.py`。

冻结测评代码：`/opt/data/private/ECE4300-controller-20261127/eval-hybrid-v3-release/`。从实际旧测评 release 复制后，仅增加反思模型路由及结果元数据，保留原协议、题库、额度和提示词。

两张四号机 4090 等待当前采集/API 请求结束后，按现有动态池封装预约、排空、训练，完成后自动归还推理。先最长样本独立冒烟，再从原 SFT 重新加载正式训练。发布至四号机双 4090 与单 5090，ready 服务为 localhost:28287。

采集器启动时固定了训练优先级文件列表。为避免中断采集，V3 暂复用已结束 V2 的 inactive `training-request.json` 作为兼容让卡标记，payload 的 owner 明确为 `dpo-reflection-v3-ep1`。这不表示重启 V2。原 V2 已 ready，未再执行训练；V3 三题入队后标记自动清除。

唯一验证队列仍为 `/opt/data/private/ECE4300-controller-20261127/quick3/`。控制器未重启，在途题进程保留。`future_models` 等 V3 ready 后自动加入原 quick3 三题、45 个参考解。

队列的 Python 启动入口改为独立 `hybrid_worker_python.py`，它只拦截模型名 `dpo-reflection-v3-ep1`：将主模型设成 SFT ep1，并附加 `--reflection-model dpo-reflection-v3-ep1`，使用冻结混合测评 release。其他模型原参数、原测评目录透传，不改变旧实验。

结果目录仍按队列任务名保存：`quick3/results/dpo-reflection-v3-ep1/`。**该目录是混合模型结果，不是全阶段 DPO v3。** 文件内 `tree_config.model_strategy=hybrid-reflection-v3`，`stage_models` 列出各阶段模型；`requested_model` 为实际搜索主模型 SFT ep1。反思响应内 model 应为 DPO v3，搜索各阶段响应应为 SFT ep1。不允许缺适配器时回退 base。

## 监控给 CC

连接：`ssh EEC430004-4090`。以下均为这台服务器上的绝对路径：

- 总流程状态：`/opt/data/private/ECE4300-train-20261127/dpo-reflection-v3-ep1/status.json`
- 连续日志和参考概率计算进度：`/opt/data/private/ECE4300-train-20261127/dpo-reflection-v3-ep1/pipeline.log`
- 冒烟结果：`/opt/data/private/ECE4300-train-20261127/dpo-reflection-v3-ep1/smoke/status.json`
- 正式训练进度、loss、概率变化：`/opt/data/private/ECE4300-train-20261127/dpo-reflection-v3-ep1/adapter/status.json`
- 正式训练结束后完整更新历史：`/opt/data/private/ECE4300-train-20261127/dpo-reflection-v3-ep1/adapter/history.json`
- 三题任务列表：`/opt/data/private/ECE4300-controller-20261127/quick3/tasks.json`，筛选 `model=dpo-reflection-v3-ep1`。
- 三题汇总：`/opt/data/private/ECE4300-controller-20261127/quick3/summary.json`，读取 `dpo-reflection-v3-ep1` 项。
- 三题原始结果与逐题日志：`/opt/data/private/ECE4300-controller-20261127/quick3/results/dpo-reflection-v3-ep1/`

`adapter/status.json` 的 phase=reference 表示正式更新前的冻结参考概率计算，完成数见 pipeline.log 的 reference/completed/total；不能把它报成卡住或已完成参数更新。phase=training 才报告 global_step/12。顶层 phase=ready 仅表示发布成功，不表示三题验证完成。smoke 的一步完成不代表正式训练完成。

只读检查训练根的 `status.json` 与 `pipeline.log`。`waiting_for_collection_to_yield` 表示等指定卡当前 API 完成；`training` 阶段细分见 `smoke/status.json`、`adapter/status.json` 和 pipeline 中的 reference 计算进度。不要用旧固定 PID 判断存活。

训练 ready 后查 quick3 的 `tasks.json` 中 `model=dpo-reflection-v3-ep1` 三项，再读对应结果：`reflection_response.model` 必须是 V3，记录的搜索 `model` 必须是 SFT ep1，同时核对 `stage_models`。覆盖率用三个题各自 recovered/references；三题未完成不报最终整体覆盖率。原 SFT 和全阶段 DPO v1/v2 的既有结果保留作对照，不把混合结果混进旧分组。

## 已完成验证

三项本地测试通过：反思改路由不改变主模型、错误返回模型拒绝、队列只重写登记的 V3。服务器冻结版通过协议整合模拟，确认原 isolated examples 提示词与 V3 目标模型都到达反思调用。真实 CLI `--plan` 能识别混合参数与原 quick3 题。新代码编译通过。V3 发布后的真实接口检查由发布流程负责，不能把上述模拟检查说成真实 V3 已完成生成。
