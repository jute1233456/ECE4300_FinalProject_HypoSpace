# 回答采集程序 v2：只记录，不做数据处理决策

最新用户范围覆盖DPO_GENERATION_V1.md的旧设计。当前程序保持停止，未经用户明确要求不得开启。旧v1数据和分组留作历史，v2没有训练/验证标签。

## 一条采集记录

`records.jsonl`每行对应一个固定输入，包含messages、阶段、题号、来源、已有DS回答ds_answers、目标模型候选target_responses。每个目标响应保留完整原始API结果、思考文本、最终文本、模型/权重指纹、seed、token、结束原因和请求失败信息。DS回答不进入目标模型输入，也不被标成好答案。

历史DS原始wire/system消息没有完整保存；程序同时保存原始reflection_prompt、原始响应、记录索引以及该限制。Qwen输入使用当前线上阶段包装。两者来自相同历史任务上下文，但不能声称原始wire逐字一致；是否能形成偏好对由后续工作判断，本程序不判断。

重复的固定输入归为一个状态，但对应的多份DS回答全部保留。失败尝试、截断回答和重复回答也全部记录，不按内容质量删除。

程序不评分，不调用CC，不筛最好/最差，不构造chosen/rejected，不训练DPO，也不划分训练集/验证集。

## 按状态分布式调度

每个状态是一个任务，当前每任务需4份候选。一张空闲GPU领取一个状态，依次对同样的messages采样4次；每次使用不同的可复现seed，不更新题目、树和答案历史。每份回答收到后立即落盘；完成状态后归还该卡、再领取下一个状态。

中央持久队列位于四号机，各虚拟机GPU通过已有推理接口执行。空闲几张支持目标模型的卡，就同时处理几个不同状态；没有按状态数量固定为一个或两个worker。最大并发9，实际跟随空闲可用GPU数变化。同一状态不会同时交给两个GPU；意外中断后依据已保存响应，只补缺失候选。

现有验证/补基线优先，有旧pending任务时不领取新状态。通过网关原子预约已经空闲的卡，不停止任何在途请求。状态完成或正常停止后归还GPU；API异常先确认引擎空闲再归还，不杀进程。用户停止要求仍生效；PAUSE和stop_requested.json保留，启动保护已验证。

## 当前准备结果与位置

- 目标模型：sft-pruned-v1-ep1，944条SFT数据的checkpoint-59。
- 阶段：reflection；通过prepare参数可换selection、expansion、candidate，采样器本身不变。
- 19道训练题、521个固定输入，记录561份关联DS历史响应；计划2084份目标回答。
- 正式生成仍为0，未启动新生成进程。
- SSH：EEC430004-4090。
- 数据目录：`/opt/data/private/ECE4300-dpo-candidates-20261127/reflection-v2-collection/`。
- `states.jsonl`：固定输入及DS回答；`records.jsonl`：统一记录文件，初始target_responses为空，生成后刷新。
- `responses/`：逐次目标模型原始响应；`questions/`和`contexts/`：原题及来源。
- `state.json`：状态、总状态数、已完成状态数、候选收到数和运行GPU数。
- 完成或正常停止后自动导出`collected-responses.tar.gz`。不含令牌、权重、虚拟环境。

## 代码与检查

prepare只读历史输入与DS记录，generate按状态调度采样，export只把已有记录序列化；没有评分器或划分器。准备新批次时输出目录必须全新：

```powershell
python -m dpo.prepare --output local-backups/dpo/new-collection --stages reflection --samples-per-state 4
```

六项单元检查覆盖阶段契约、空卡准入、先保存再归还、保留DS/截断/失败记录且没有划分或偏好标签，以及一个状态补齐缺失候选而不重跑已有候选。
