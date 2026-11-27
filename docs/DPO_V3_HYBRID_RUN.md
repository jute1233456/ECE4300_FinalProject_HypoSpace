# DPO v3：只做反思，SFT ep1 完成搜索

用户 2026-11-27 明确指定 V3 负责反思，其他步骤使用 SFT。本轮不新增 DPO v2 混合实验，也不把 V3 用于全部搜索阶段。

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

只读检查训练根的 `status.json` 与 `pipeline.log`。`waiting_for_collection_to_yield` 表示等指定卡当前 API 完成；`training` 阶段细分见 `smoke/status.json`、`adapter/status.json` 和 pipeline 中的 reference 计算进度。不要用旧固定 PID 判断存活。

训练 ready 后查 quick3 的 `tasks.json` 中 `model=dpo-reflection-v3-ep1` 三项，再读对应结果：`reflection_response.model` 必须是 V3，记录的搜索 `model` 必须是 SFT ep1，同时核对 `stage_models`。覆盖率用三个题各自 recovered/references；三题未完成不报最终整体覆盖率。原 SFT 和全阶段 DPO v1/v2 的既有结果保留作对照，不把混合结果混进旧分组。

## 已完成验证

三项本地测试通过：反思改路由不改变主模型、错误返回模型拒绝、队列只重写登记的 V3。服务器冻结版通过协议整合模拟，确认原 isolated examples 提示词与 V3 目标模型都到达反思调用。真实 CLI `--plan` 能识别混合参数与原 quick3 题。新代码编译通过。V3 发布后的真实接口检查由发布流程负责，不能把上述模拟检查说成真实 V3 已完成生成。
