# 当前运行与代码入口

本说明记录2026-11-25的配置，实时进度需读取服务器结果，不能用本文替代状态查询。

## 最新调度覆盖下方旧流程描述

优先运行分阶段反思MCTS 2.2（Stage-MCTS）四题小样本，结束后自动恢复反思MCTS 2.1未完成的评测题及其后续训练题流程；普通MCTS 2.1继续暂停。2.2只跑causal/n1_003、n1_002、n2_006和3d/g3_h3_b2_002，2.1对应平均覆盖率33.68%。保留2.1全部旧结果，新方法从空树开始，不共用断点。

- 快照：`releases/mcts-2.2-stage-pool6-32k`；入口：`server/run_stage22_pilot.py`。
- 状态：`logs/mcts-2.2-stage-medium-pilot-32k/state.json`；输出：`results/server-mcts-2.2-stage-medium-pilot-32k`。
- 每轮复盘、工具读树、选节点、提出步骤、生成候选各为独立API调用；完整树只送入选节点阶段，后续使用选中路径/子步骤，保留答案历史。
- 四题并行，统一走六卡池8003；相同模型、思考模式、题目、搜索轮数、请求32768。实际生成额度仍由总上下文减输入决定。
- 不能把相同搜索轮数称为相同API或token预算；结果需报告完整阶段开销。本样本偏向2.1低分题，仅作探索判断。
- 54项离线测试在服务器全部通过；本地54项通过（其中两项POSIX测试跳过）。实时完成数与覆盖率以结果文件为准。
- 实际首轮验证：causal/n1_002完成读树、选择、扩展和候选四阶段，候选有效，无协议错误。用户要求程序能跑即可，不持续监控。
- 2.2每轮实际最多5次API（复盘1次、搜索4次），工具最多1次。旧通用入口的 `api_call_cap` 和启动日志仍套用两步公式，不代表2.2真实上限；本试验manifest的 `cost_accounting` 给出正确上限，实际消耗按 `api_requests` 与 `reflection_response` 汇总。
- 新版本代码、首轮结果及服务器状态已增量备份到本地 `local-backups/stage22-deploy/server-state.tar.gz`，不含权重、环境或凭据，不提交Git。

## 方法与配置

- MCTS 1.0：提示词分支菜单；2.0：工具读树；2.1：工具读树并允许重复有效解回退、严格限制 EXHAUSTED。
- 普通MCTS与两步反思是分别报告的变体；模型、难度、token额度是独立的配置维度。
- 当前Qwen3-8B开启思考，请求生成额度32768；总上下文32768，实际额度为 `32768 - input_tokens - 8`。输入不静默裁剪。
- 客户端和网关等待3600秒；网关通过 `ECE_BACKENDS` 使用已配置的三个推理后端。
- 当前仅两步反思MCTS 2.1，6题并行；普通MCTS 2.1已暂停。旧16K记录保留，迁移结果通过 `generation_limit_transitions` 明确标记，不能宣称旧轮次也用了32K。

## 当前服务器流程

项目目录为 `/opt/data/private/ECE4300 Final Project`，SSH别名 `EEC4300`。私钥在本机SSH目录中，不存放在仓库。

活动快照：`releases/mcts-2.1-temp-pool6-32k`。状态：`logs/mcts-2.1-temp-pool6/state.json`。

顺序：

1. 在中等评测27题上仅继续两步反思MCTS 2.1，保留普通组已有断点。
2. 生成独立训练题库 `datasets/medium_training_v1`。
3. 仅两步反思MCTS 2.1运行训练题，导出交互轨迹，尚不启动微调。

输出：

- 评测：`results/server-mcts-2.1-medium-evaluation-32k`。
- 训练题轨迹：`results/server-mcts-2.1-medium-training-32k`。
- 训练交互导出：训练输出目录下的 `training_exports`，仍需验证tokenizer模板、思考隔离和assistant loss mask后才能用于微调。

## 源码分工

- `agents/`：树状态、分支策略、读树工具和版本策略。
- `llm/`：模型调用、思考隔离、工具调用循环、上下文额度。
- `experiments/direct_evaluation.py`：评分、断点恢复、版本检查与记录。
- `experiments/generate_medium_training.py`：训练题生成、等价去重和完整参考解核验。
- `server/run_evaluations.py`：按环境配置执行题目。
- `server/run_medium21_pipeline.py`：当前中等评测到训练轨迹的顺序流程。
- `server/backup_project_state.py`：排除模型、环境和凭据的信息归档。
- `tests/`：离线检查；不调用真实模型。

`queue_mcts21.py` 是已取消的旧“等待2.0再跑简单题”队列；`switch_medium21_32k.py` 是已经执行过的一次性迁移工具。二者保留作历史复现，不能当作当前流程重复执行。

## 训练题边界

预检可生成28题：因果12题、布尔12题、3D四题。参考解已独立枚举核验；排除评测题的节点重命名、变量交换和3D旋转/反射/平移等价情况。

3D训练使用4×4棋盘、两个占用位置、高度上限3，每题9个参考解；这是与3×3评测题不同的训练空间变体，需在报告中注明。27道评测题不得混进训练交互导出。

汇报进度时，同时给出版本、变体、题数和按正常完成题等权平均的覆盖率。比较效果要对齐共同题目，并分别报告截断、网络错误和工具错误。

当前临时六卡池127.0.0.1:8003包括新增lab4090双卡与lab5090单卡；节点目录 /opt/data/private/ECE4300-temp-api-20261125，其他工作区不操作。凭据不得提交。
