# V3 混合路由 mixed100 队尾任务

用户 2026-11-27 要求停止 DPO 数据生成，将 V3 混合路由的 mixed100 任务排在现有队列尾部。未经用户重新明确授权，不恢复采集。

采集目录：`/opt/data/private/ECE4300-dpo-candidates-20261127/reflection-v2-collection/`。已写入 `PAUSE` 与 `stop_requested.json`，核对 PID 对应本采集器后发送合作退出信号。停止时 1105/2084 份响应、2 个状态在途；在途 API 完成后保存响应并归还卡，不再开始下一份候选。停止后的最终数以该目录 `state.json` 为准，不把发送停止信号等同于已全部退出。

推理仍使用四号机统一队列 `/opt/data/private/ECE4300-controller-20261127/quick3/`，现有任务未中断，控制器未重启。

模型组名：`dpo-reflection-v3-ep1`，仍为 V3 反思、SFT pruned v1 ep1 完成读树/选择/拓展/候选。它不是全流程 V3 对照组 `dpo-v3-full-ep1`。

复用 `datasets/mixed100_v1/manifest.json`：25 题、293 参考解；现有 quick3 的 3 题45解以模型＋题族去重，不重复启动。新增 **22 题、248 参考解**，覆盖简单、中等及补充 3D 难度。

已有 additional_suites 的 `mixed100-v1.models` 加入该混合组，热更新自动生成任务。新增任务 priority=1，将混合组放到 model_order 最后，保证现有 quick3 priority=0 和其他模型原有追加任务排在前面。这里的队尾是派发顺序；有空余槽位且前面的待派发任务已启动时，允许新任务启动，无须等所有在途题完成。

CC 监控：在统一 `tasks.json` 中筛选 `model=dpo-reflection-v3-ep1` 且 `suite=mixed100-v1`，应有22项。整个混合组 `summary.json` 的 total 将是25，包含原quick3。新增结果位于：

`/opt/data/private/ECE4300-controller-20261127/quick3/results/dpo-reflection-v3-ep1/mixed100-v1/<difficulty>/`

核对结果 `tree_config.stage_models` 和每轮反思、搜索响应模型名；汇报覆盖率时同时统计旧quick3与新增22题，并注明在途结果。不要将全流程 V3 的记录并入该混合组。
