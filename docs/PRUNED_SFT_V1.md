# 原始格式删减版 SFT v1

2026-11-27 用户收窄验证范围：仅保留 linear-ep1 和 linear-ep2，其他模型暂停并保留结果与工作进程；最新调度为按模型分组分配三题，允许跨模型并发，不设模型完成屏障。全量删减版训练完成且发布检查成功后，再接在这两组后面验证。原19题基准已经正常完成，不再占卡。

## 数据与训练

完整已审查教师数据1024条，按真实Qwen模板长度删除80条超过14336 token的样本；保留944条，覆盖19题，不改原始消息内容。selection 266、expansion 332、candidate 346，最长14302 token。排除文件独立留存，不将超长样本截断后混入训练。

输入SHA256：`6f3e178e83d0004b6e2e84ec35ea1c790faac09db208ebf784112eaeaab8579e`。过滤实现为 `server/filter_sft_lengths.py`；本地数据及详细manifest在 `training/data/ds22-sft-pruned-v1/`，原数据、权重和令牌均不随本次代码提交。

模型名 `sft-pruned-v1-ep2`，从原始Qwen3-8B独立初始化。NF4 QLoRA，r16/alpha32/dropout0.05，七个线性投影，batch1、梯度累积16，有效batch16；双4090模型分片，不把batch额外乘2。2 epoch共118次更新，lr5e-5，warmup12次更新，随后反比例衰减shape2。assistant-only loss和梯度累积修复沿用已验证实现。与原309条线性对照的数据量及学习率策略均不同，不能归因为单一因素。

最长样本两步冒烟通过：约24.9秒，峰值显存11.06/17.18 GiB，没有OOM。随后才启动944条正式训练，冒烟权重不作为初始化，也不进入验证池。

## 自动衔接

四号机双4090的持久目录为 `/opt/data/private/ECE4300-train-20261126/sft-pruned-v1-ep2/`，其中 `config.json` 保存实际命令、`handoff.log` 保存日志、`status.json` 保存阶段、`adapter/` 保存正式产物。代码冻结到该目录的 `code/`。入口为 `server.run_sft_handoff`。

1. 通过既有 `run_reserved_training` 预约并排空两卡，停推理服务后运行最长样本冒烟和正式训练。无论训练成功失败，封装器均尝试归还两卡。
2. 验证checkpoint-118、epoch2、全部权重有限以及LoRA B非零。成功后复制两份适配器必需文件至服务器项目资产目录，计算SHA，逐卡排空并运行时注册新名字，最后用真实API请求确认可生成。
3. 四号机内网28281提供只读 `/status`，使用既有池令牌鉴权，仅返回固定JSON收据。状态服务不占GPU；失败保留明确failed及原因。
4. 主机quick3队列热读enabled_models与future_models。只启用两个线性组，future_models轮询上述收据；只有ready才追加新模型三题，分配优先级排在两个线性组之后，有空闲名额即可运行。训练未结束时控制器保持等待，失败不冒充验证完成。已完成题跳过，暂停任务保留。

已做三项隔离调度验证：禁用base后只启动linear-ep1；新模型ready才入队；训练未结束时不提前退出。动态池、原评测代码、MCTS2.2协议均不变。
