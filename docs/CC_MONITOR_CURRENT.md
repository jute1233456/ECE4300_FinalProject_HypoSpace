# 给 CC：当前任务监控说明（2026-11-27）

你的身份是负责监控的 Claude Code。仅检查和汇报，不自行调整配置、重启、暂停、杀进程、扩充题库或启动训练；发现异常先说明证据。历史固定 PID/端口可能过期，以本说明中的配置、state.json 和当前进程为准。

## 连接与任务范围

本地连接：`ssh -o BatchMode=yes EEC430004-4090`。

1. SFT v1 ep1（`sft-pruned-v1-ep1`）：混合题库25题293参考解，已包含原quick3三题，不重复跑。
2. DPO v1（`dpo-reflection-v1-ep1`）：仅原quick3三题45解，后续22题已取消，不恢复。
3. DPO v2（`dpo-reflection-v2-ep1`）：130对、从SFT ep1重新训练；lr1e-6，微批1/累积16，1epoch共9更新，warmup2更新后反比例下降，beta0.1。训练与发布成功后自动quick3三题45解，不跑完整混合题库。
4. DPO原始候选采集：重新获用户授权，但仅用训练/推理之外的余卡。目标仍是SFT ep1，521状态×4份=2084回答，不评分、不划分数据。等待余卡不是故障。

## 推理队列

根目录：`/opt/data/private/ECE4300-controller-20261127/quick3/`。

- `config.json`：enabled_models、additional_suites、future_models是当前授权范围。
- `state.json`：控制器PID、running列表、容量、future_models状态。
- `tasks.json`：逐题status、pid、references、result、result_key。
- `summary.json`：每模型**已完成题**汇总，不含在途题找到的全部解。
- 原三题：`results/<model>/<dataset>-<case_id>.json`。
- 新增题：`results/<model>/mixed100-v1/<difficulty>/<dataset>-<case_id>.json`；同名.log查看阶段、请求错误和截断。
- `controller-mixed100.log`：控制器异常。

统计时只计算enabled_models，过滤历史未启用recip模型；state.total可能包含它们，不是当前有效总任务数。DPO v2在ready前尚未插入tasks.json，应报告“等待训练完成入队”，不能据此判定任务丢失。

汇报每模型：完成/总题数、运行/待跑数、找到解数/全部参考解数、覆盖率，并注明是含在途进度还是仅已完成题。**每道在途题必须同时给出已用/搜索上限与剩余次数**（例如 `used 60/106，剩 46`），否则无法判断任务何时结束；used 取结果 JSON 的 records 数，cap 取 `call_cap`。

计算含在途覆盖率：读取每题JSON的cases和interrupted_case中records，按reference_hit=true且hypothesis_key去重计数；key可能是列表，先转稳定JSON字符串再去重。可优先采用已完成task.result.recovered；不要直接把records数量当成找到解数。

总解覆盖率 = 各题找到解数之和 / 全题库参考解数之和。按题平均覆盖率是另一指标，必须标明，不能混用。尚未做的题不宜悄悄从总题库分母删除。同模型原三题与混合题库按任务来源去重。

## DPO v2 训练

根目录：`/opt/data/private/ECE4300-train-20261127/dpo-reflection-v2-ep1/`。

- `status.json` / `pipeline.log`：预约排空、训练封装、归还、发布和失败。
- `smoke/status.json`：最长单样本真实更新检查；completed只表示冒烟完成，不代表正式训练结束。
- `adapter/status.json`：正式阶段、global_step/total_updates、epoch、loss、lr、effective_batch、梯度范数。reference阶段的逐对进度见pipeline.log。
- `adapter/history.json`：正式更新日志，完成后可查看。
- `config.json` / `data/manifest.json`：真实配置、130对来源、排除3条记录及指纹。

注意外层status.phase=training可能仍处于GPU排空/冒烟/参考概率计算；用adapter/status.json和pipeline.log判断实际阶段。global_step=9且adapter.phase=completed才表示正式训练完成；root.phase=ready表示发布就绪，**不表示三题验证完成**。

首次DPO v1曾OOM，已改为精确分开反传、expandable_segments；v2使用此修复。若报CUDA OOM、非有限损失或梯度、wrapper退出非0、root.failed，应及时汇报，不自行重训。

## 候选采集

根目录：`/opt/data/private/ECE4300-dpo-candidates-20261127/reflection-v2-collection/`。

- `state.json`：received/2084、completed_states/521、running、errors、pid、phase。
- `generation-priority-v3.log`：当前恢复后的控制器日志。
- `responses/`：每次原始回答，收到即保存；`records.jsonl`是周期导出，可能略滞后。
- `config.json`：priority_queues和training_requests定义让卡规则。

从605份回答恢复。605份里原始“完整状态计数”和实际可评审状态数可能因候选空文本/完整性标准不同而不同；采集计数不能直接叫“可训练状态数”。训练集v2离线评审的是148个完整可评审状态，原始133对，实际训练剔除3条后130对。

phase=waiting_for_idle_model_gpu且running=0，在推理占满时是预期。不要为让采集继续而暂停跑题或训练；原605份不会重复生成。

## GPU与健康检查

在服务器执行：

```bash
curl -s http://127.0.0.1:8005/health
```

查看backends的id、role、models、owner、inflight、observed。训练卡没有推理metrics是正常的；running题进程数不等于正在占用GPU数，训练期间原9个题进程可以排队等待剩余推理卡。

lab4090-g0曾因启动时保存空命令行SHA而被代理误判“No running managed service”。现已核验身份并修复，SFT ep1真实生成验证成功，网关已观察到inflight=1。已部署等待真实argv的启动修复，轻量代理已重启；GPU0/1模型进程均保留。不要把它继续当成未加载适配器的空闲卡。

若需要检查管理服务，节点代理健康路径是`/health`，不是`/status`。lab4090为http://10.106.254.59:28280/health；涉及管理token的读取或请求不能输出token，不在聊天/日志/提交中粘贴完整含凭据配置。

异常判断：先比对当前时间、文件更新时间、/proc/<pid>/cmdline和日志最新活动。长推理期间搜索轮数不增加不必然卡死；metrics有请求或日志仍有token活动时，先报告“在途长调用”。控制器心跳长期不更新且进程不匹配、连续Request failed、任务needs_inspection时汇报证据。

## 简短汇报模板

“SFT ep1：完成X/25，含在途Y/293解、Z%；DPO v1：完成X/3，Y/45解；DPO v2：参考概率X/130或更新X/9，验证X/3、Y/45解；采集X/2084、完整状态Y/521、运行N、错误E；GPU：训练N张、实际推理N张、采集预约N张、空闲N张及原因。”

只在完成、失败、明显异常或需要用户决策时主动通知；普通轮询不反复发送无变化消息。
