# DPO未评分数据生成 v1

**已被用户修订的v2采集方案取代**，当前以 `DPO_COLLECTION_V2.md` 为准：只保存输入、DS历史回答和目标候选，不划分、不评分、不配对；生成保持停止。下文仅为旧方案历史，不可据此开启程序。

## 当前状态：用户要求停止，等待检查

生成器及DPO专用模型扩展发布进程均已停止，正式响应0份、占卡0；其他做题队列继续。state.json标为user_stopped，保留PAUSE和stop_requested.json。未经用户明确要求不得恢复。现有输入及train/validation标签只保留供用户检查，尚未获最终认可；以下原计划不构成继续执行授权。

## 范围

用户最新要求：只生成并保存原始数据，不调用Claude Code，不评分，不自动形成chosen/rejected偏好对，不训练DPO。反思分析是首要阶段；采样器支持reflection、selection、expansion、candidate，由准备输入时的参数指定，不需要重写GPU采样逻辑。

当前模型是944条SFT数据训练的 `sft-pruned-v1-ep1`，checkpoint-59，权重SHA256 `8fa112bf26333cd24ea3c2c3759cd5584453a2cce83e75ba8781f569e4bac63e`。不是旧309条数据的linear-ep1。

## 已冻结的第一批输入

源为19道训练题的规范化DS2.2轨迹，581轮；reflection_prompt缺失的20轮不重建猜测，完全相同的反思输入合并40项，得到521个固定状态。覆盖因果136、3D162、布尔223个状态。每个状态采样4份，计划2084份回答。包括有重复、无有效新解等历史，不只取成功轮次；旧DS的本轮答案不作为生成目标，也不展示给模型。

反思输入复用线上 `isolated_examples_v1` 的system/user隔离、格式示例及三行Review/Constraint check/Next direction契约；既不让反思阶段提交最终候选，也不向它添加线上未提供的树或未来结果。其他阶段直接使用已经审查的SFT输入，删除assistant教师答案。

按规范化等价题族划分：训练415状态、验证106状态（20.35%），每类至少保留一个完整验证族。验证族为extended_n1_inputs_00_outputs_1、g3_h3_b2_001、n6_hard_029对应的整个等价族；不是随机拆同题的回答。因果只有两族，所以无法在各类都有验证的同时精确做到10%。这些题在SFT阶段已使用，故这是DPO阶段holdout，不是未见题泛化测试。独立混合测试题库没有用于生成。

## 程序

- `dpo/prepare.py`：只读规范化数据和已审查SFT，冻结阶段输入、原题参考和族划分。
- `dpo/generate.py`：空闲GPU采样、原始响应落盘、断点恢复、自动导出。没有评分依赖或Claude调用。
- `dpo/export.py`：把同一输入的回答分组，导出未评分train/validation文件和压缩包。

本地重建输入示例（输出目录必须全新）：

```powershell
python -m dpo.prepare --output local-backups/dpo/reflection-v2 --stages reflection --samples-per-state 4
```

要增加其他阶段，在新目录使用 `--stages reflection candidate` 等参数。reflection取规范化轨迹中的反思前输入；其余阶段取已审查SFT输入，不直接混用不同状态输出。

## 服务器入口与监控

SSH `EEC430004-4090`，数据根目录：

`/opt/data/private/ECE4300-dpo-candidates-20261127/reflection-v1/`

- `state.json`：动态PID、总数、received、errors、running、当前等待原因和心跳。
- `generation.log`：异常日志。
- `manifest.json`：冻结输入清单、来源SHA、划分、计划数量。
- `states.jsonl`、`contexts/`、`questions/`：逐状态固定输入、来源、原题规则与参考解。
- `responses/`：每次尝试独立JSON文件，含完整原始API响应、思考文本、最终文本、模型、GPU、seed、token使用、截断状态及错误。重复回答和截断输出也原样保留，留待CC判断。
- `review/train.unscored.jsonl`、`review/validation.unscored.jsonl`：累计每20个成功响应刷新；同一输入的多份回答放在outputs数组。
- `cc-review.tar.gz`：结束或正常停止时自动生成的统一交接包，不含配置令牌、模型权重或虚拟环境。没有评分、chosen、rejected字段。

后台控制入口：四号机 `/opt/data/private/ECE4300-controller-20261127/code` 下 `python -m dpo.generate --config <数据根目录>/config.json`，解释器为控制根目录 `venv/bin/python`。进程已部署，等空闲模型GPU自动生成，不需要人工盯住。

## 算力和恢复

优先给quick3验证及混合题库补基线队列；有更高优先级pending任务时不开始新生成。按这些队列的存活worker数计算剩余槽位，并要求GPU空闲、支持ep1、没有其他owner。通过网关原子预约空卡，409冲突就跳过，不抢断在途调用；每张预约卡一次只生成一个回答，完成后归还原角色。

参数沿用Qwen线上随机采样temperature0.7/top_p0.8/top_k20，开启思考，输出请求上限32768，实际按32768上下文减完整输入和8token余量；不截输入。不同样本使用不同可复现seed，同一状态messages不变。API异常最多两次尝试，全部原始记录保留；超时后等metrics证明引擎空闲再归还，不盲目重放或杀进程。

同目录PAUSE文件只阻止新采样；已有调用自然结束。SIGTERM同样停止领任务并保存当前输出后退出。重启通过响应文件恢复，成功样本不重跑；状态在断电或异常退出时遗留的本任务预约卡，仅在metrics证明空闲后回收。配置、提示词或模型变更必须使用新的输出目录。

其他六张卡的ep1能力已安排逐卡后台补齐，目录在控制根目录 `publish-dpo-ep1-pool/`；不重启推理服务，不触发评分。

## 验证

六项单元检查通过：线上反思契约一致、题族隔离、空卡准入与模型匹配、已结束队列心跳处理、先落盘再还卡、保留截断且不产生偏好标签。服务器模拟端到端生成两份输出，验证自动导出、归还GPU以及第二次运行不重复生成。正式生成只在空闲卡出现后执行；没有用测试题冒充生成数据。
