# 混合100解题库：SFT ep1 与 DPO v1 队列（2026-11-27）

用户授权两个模型跑 mixed100_v1，并要求已存在的三题验证不重复；随后要求暂停全部候选采集。

- 题库25题293解：因果7题100解、布尔11题100解、3D7题93解。
- 模型：`sft-pruned-v1-ep1`、`dpo-reflection-v1-ep1`；方法均为新版MCTS 2.2阶段隔离示例版、thinking、32768请求额度。
- 每个模型复用已有quick3的三个题族（45解），包括其已完成结果和在途进程；仅追加22题248解。因此共44个新任务，最终每个模型25题293解。
- 任务按model+题族去重；不同难度下相同case_id仍是不同题，新增任务和文件均含难度目录。
- 现有quick3任务派发优先，新增题库排在后面；空槽可接新题，不等待所有quick3任务结束。一任务一worker，动态总并发上限9。

## 唯一主队列

SSH `EEC430004-4090`。

`/opt/data/private/ECE4300-controller-20261127/quick3/`

`config.json` 的 additional_suites 固定 manifest SHA 和两个模型。`tasks.json` 为持久任务表，`state.json` 为控制器及在途PID，`summary.json` 为每模型全题库汇总。两个目标模型的total现为25，不再是3；若只看快速三题，应筛选没有suite字段的原始任务。

原三题结果目录不变。新增结果路径：

`results/<model>/mixed100-v1/<difficulty>/<dataset>-<case_id>.json`

控制器更新时保留了全部5个在途做题PID，并核实命令指纹；新控制器接管后已派发4个新题，共9个并行worker。原配置与状态快照在 `before-mixed100-20261127/`。不要启动第二个主队列写同一组结果。

`tasks.json` 还保留未启用recip模型的历史pending任务；它们不是本次授权测试，统计在跑/待跑时需按enabled_models过滤。

## 数据采集已暂停

采集根：`/opt/data/private/ECE4300-dpo-candidates-20261127/reflection-v2-collection/`。

已保留PAUSE，合作停止控制器，在途回答已保存并全部归还卡。停止时605/2084份回答、149/521个完整状态、0错误、0运行任务。未经用户重新明确开启，不移除PAUSE或重启采集。

空闲的lab4090-g1已加载已有SFT ep1适配器，可接测试。DPO目前发布在node4双4090和node4单5090，网关按模型能力路由，不回退base。

## 检查

两项队列测试通过：已在途题族不会重复追加、重复扫描幂等、跨难度同名题输出互不覆盖、quick3派发先于新增任务。实机断言每模型25题293参考解，新增各22题；4个新增题已读取各自难度的正确元数据并进入首次搜索。

## 后续用户调整：DPO只跑三题（2026-11-27）

已取消并归档DPO v1的22个尚未启动的mixed100任务，additional_suites仅保留SFT ep1。DPO仅保留原quick3三题及已有进度，SFT仍为25题293解，采集继续暂停。此调整覆盖本文前述双模型完整题库安排。
