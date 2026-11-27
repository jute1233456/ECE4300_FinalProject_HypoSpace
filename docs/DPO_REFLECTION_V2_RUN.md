# 保守 DPO v2 与三题验证（2026-11-27）

用户在商讨后明确批准：降低学习率、增大有效batch、两步warmup、反比例下降；训练完成只跑quick3。候选采集获重新授权，但仅使用训练和跑题之外的空余卡，不恢复DPO v1已取消的混合题库任务。

## 数据复核

原始simple-reflection-dpo-v2归档SHA256：db7891c934611add149c2d7d52f7ca3fb7b2d44a784e00bfa19c4262bed53882。

独立检查133对：状态与输入唯一、prompt与冻结messages相同、266个回答与冻结原文及哈希一致、标签high/high、Qwen来源为预期SFT ep1且finish_reason=stop、模板前缀匹配。最长prompt4458、完整序列4738 token，全部低于8192。对新增正样本进行了定向语义抽查，不代表全部标签经穷尽数学证明。

原始133对保留，另建130对训练版，排除：

- 0258751fb1219ad03851023c7487dc7588493116c40ed48b8cc7ff3cfcf71516：v1已知错误，把y OR NOT x说成深度1。
- 41ccbd793a5c476736ab753466487bb650a0e513bb33a94a2f086259c663ec2e：正样本声称尚无Layer2为足迹真子集的候选，输入历史已有多个反例。
- ca14adbc265e2a4757f62e4d6c31cea54021c5b2fa288cd29441649423c3997f：正样本把NOR展开记号机械计作NOT和OR两层并禁止复合操作数，而同一历史已有复合NOR被接受的反例。

训练文件SHA256：6e06f4e925255c4df305d72ab89f6ef1aecdcefc5f195dadfe724bb01423c2c9。

## 配置

- 名称：dpo-reflection-v2-ep1。
- policy起点和冻结reference均为sft-pruned-v1-ep1，SHA256为8fa112bf26333cd24ea3c2c3759cd5584453a2cce83e75ba8781f569e4bac63e。不接着DPO v1训练。
- 130对、1epoch、微批1、累积16，共9次更新；尾批2对按实际2归一化。
- 峰值lr=1e-6；更新1/2分别为5e-7/1e-6，之后反比例下降，更新9约4.545e-8。beta=0.1，clip=1。
- 沿用已跑通的NF4/BF16、r16 LoRA、关闭dropout、completion-only logprob、精确分开反传两个回答；开启expandable_segments避免碎片。最大序列8192，不截断。
- 先最长样本独立真实更新检查，再从原SFT重新加载进行正式训练。

## 服务器与监控

SSH：EEC430004-4090。

持久根：`/opt/data/private/ECE4300-train-20261127/dpo-reflection-v2-ep1/`。

`status.json`/`pipeline.log`是预约、训练、归还和发布状态；`smoke/status.json`为最长样本检查；`adapter/status.json`为正式训练更新步数与损失，参考计算进度见pipeline.log；`data/`保存训练数据与排除理由，`config.json`保存配置。

双4090按现有动态封装排空后训练，不打断在途API；结束后归还推理。适配器发布至node4双4090及node4单5090，ready回执端口28286。主队列quick3/config.json的future_models自动接入同样三题（共45解），不会加入mixed100 additional_suites。

验证任务与结果：`/opt/data/private/ECE4300-controller-20261127/quick3/`，筛选model=dpo-reflection-v2-ep1，共3题。训练ready不代表评测完成。

原reflection-v2-collection采集器从605份回答断点恢复，原始配方不变，训练请求标记包含v1/v2，按现有pending和running任务占用计算剩余名额；没有余卡时等待，不争抢。

与v1同时改变数据量、学习率和batch，属于组合对照，不能把效果变化全部归因于数据增加。
