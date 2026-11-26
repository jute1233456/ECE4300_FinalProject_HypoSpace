# ECE4300 Final Project 交接说明

## 当前真实状态：九卡动态池已部署，三题验证已续跑

2026-11-26用户要求将验证从6题缩为3题，并实测训练/推理卡无全局中断切换。**已部署并验证，不是下文“离线尚未部署”阶段。** 一号机 `EEC4300` `/opt/data/private/ECE4300-dynamic-pool-20261126/`，新网关8005，旧8002/8003/8004全部兼容转发至8005；六台节点代理内网28280，新模型端口28000/28001。九卡均支持Qwen3-8B及五个原适配器。逐卡role及worker上限可动态修改，排空只等当前API，不再全局停题。

实机证据：`swap-validation.json` passed=true；两张4090在有2048token诊断API时进入draining，两请求均HTTP200完成后释放显存，执行两次真实QLoRA更新（11.64秒），训练退出0后自动重启推理并归还shared；期间旧19题继续、无全局暂停。诊断请求和训练测试不计正式分数。

新唯一验证为该根目录 `quick3/`，控制器PID154578，清单 `datasets/quick_validation3_v1.json`：因果n2_006(24解)、3D g3_h3_b2_002(9解)、布尔n2_inputs_00_01_outputs_00(12解)，共45解。6模型×3题=18任务，当前6并行，旧19题3并行，已确认九后端都有请求。随旧题完成可自动增加验证并发。base17轮和linear-ep1的12轮旧记录签名前缀完整保留并--resume续跑。旧四号机quick6调度器和2个题目进程已停止归档，不得重启，避免重复写入。原19题进程已通过8004兼容入口恢复。

动态池实现修复NFS时间戳缓存、凭据切换；19项离线测试通过。训练应使用 `server.run_reserved_training` 封装预约/排空/释放/运行/finally归还，不能裸启动去抢卡。新训练模型另发布校验并注册，不自动替换旧模型名。详见 `docs/DYNAMIC_POOL.md`。凭据只留私有配置，不提交；权重只在服务器间同步，不下载本地。

## 动态算力池：离线验证完成，尚未部署（2026-11-26 21:05）

用户方案：每张卡有可动态修改的任务状态（baseline/validation/shared/draining/reserved/training/disabled）；统一网关只把请求分给“可接任务且模型匹配”的卡，**绝不把适配器请求发给原版卡**；切换前先停接新请求并等排空；vLLM 0.11 运行时加载/卸载 LoRA；首次迁移做一次受控切换，先在一张空闲卡验证模型路由、排空、训练释放。

- 代码：`server/dynamic_api_pool.py`（模型感知网关、管理接口、排空守卫、404 不回退）、`server/gpu_node_agent.py`（仅管理登记进程，停止前必须证明引擎空闲，适配器 SHA256 校验后运行时加载）、`server/poolctl.py`（status/assign/drain/reserve/release/workers/load）、`server/dynamic_pool_proxy.py`（8004→8005 兼容）、`server/run_dynamic_validation.py`（并发跟随池容量的持久队列）。
- **尚未部署**：没有 pool/node 配置与 token，8005 未启动，8004 仍是旧静态网关（7 张卡）。
- 本轮离线改动：①baseline 入口请求适配器名立即 404，不再空等 30 分钟；②新配置先校验后替换，坏配置不再覆盖好配置；③`drain` 在引擎 metrics 不可用时立即报错并保留卡；④去掉 vLLM 0.11 已弃用的 `--disable-log-requests`。
- 测试：`tests/test_dynamic_pool.py` 15 项全绿（路由/无回退/排空/鉴权/配置校验/节点代理占用与 checksum 拒绝/示例配置）。示例配置 `config/dynamic-pool.example.json`（7 baseline + 2 validation）、`config/gpu-node.example.json`。
- 只读核对：vLLM 0.11.0 的 LoRA 参数与运行时加载路由存在；节点代理完整 argv 经 vLLM 真实 parser 验证通过。
- 未做（需要空闲卡与迁移窗口）：节点代理实机、8005 启动、真实排空与显存释放、LoRA 卸载接口、训练结束自动归还、quick6 改为池任务。建议测试卡用 node4-5090，等 Qwen 19 题（本轮 16/19）跑完后受控切换。

## 当前授权：新1/2/4epoch对照＋自动quick6验证

用户要求全程自动衔接后可离开，不需持续监控。已设计三组从基础Qwen独立训练：recip-ep1/2/4，峰值lr分别1e-4/5e-5/2.5e-5（lr×epoch均1e-4）；warmup10%（2/4/8次更新），之后反比例先快后慢下降。固定六题清单 `datasets/quick_validation6_v1.json`，中等27题中每类两个不同族，排除完整教师19题等价族，114参考解。六模型（base、旧linear-ep1/ep2、新recip-ep1/ep2/ep4）各跑相同MCTS2.2阶段隔离示例版六题，共36任务。控制器 `server/run_finetune_quick6.py`，专用目录四号机双4090 `/opt/data/private/ECE4300-train-20261126/quick6-controls-v1/`；先等旧2epoch结束，串行完成三组新训练，再用本双卡各一独立LoRA服务、两个worker自动验证；不操作原7卡池及19题。详见 `docs/QUICK6_AUTOMATION.md`。以下“尚未授权立即运行”仅为早期设计历史，已被此次明确授权覆盖。

已启动控制进程PID28327，确认进入recip-ep1，20更新/warmup2。真实调度器3步与六题入口、源文件哈希、LoRA API生成通过；完整32K单轮MCTS诊断主动中止，不作为成功评测。只需后续读取state.json/pipeline.log/summary.json，不持续人工监控。若训练失败不覆盖目录，某组失败留日志且跳过该模型验证，其他组继续。

## 待执行设计：低学习率＋warmup对照

用户要求另记一组：同309条A数据、同基础Qwen、2epoch，峰值lr5e-5、warmup4次参数更新（40步的10%）；用户随后明确不要线性衰减，改为先快后慢的归一化反比例曲线：衰减进度t∈[0,1]时 `lr=5e-5*(1-t)/(1+2*t)`，终点0。其余参数与当前2epoch基准一致。微批1/累积16/有效batch16，双卡为模型分片不乘2；4次更新约64条训练样本。仅记录设计，尚未授权立即运行或加入队列，不改变当前基准；详见 `docs/QLORA_LR_WARMUP_CONTROL.md`。这是降低lr、warmup与衰减曲线的组合对照，不能声称单因素因果。

## 当前：上半 A 独立 2 epoch QLoRA（2026-11-26）

用户批准新的2epoch对照训练，已从基础Qwen3-8B重新初始化，未接着单轮适配器累训。四号机双4090 `EEC430004-4090`，持久目录 `/opt/data/private/ECE4300-train-20261126/ga16-epoch2-v1/`，控制进程启动PID25775，309条冻结数据/预计40更新，已包含梯度累积修复；旧单轮目录完整保留。日志 `train.log`、状态 `status.json`、参数哈希 `run_manifest.json`；只以completed标志确认结束，不自动跑第3轮。详见 `docs/QLORA_EPOCH2_RUN.md`。原版Qwen MCTS2.2启动前14/19，含未完成题找到135/358解（按解37.71%、按题均值52.42%），5题仍继续，未操作推理池。

## QLoRA 首轮验证已完成（2026-11-26）

用户已授权修正梯度累积并只运行一个 epoch 验证流程。`server/train_qlora.py` 设置 `trainer.model_accepts_loss_kwargs=False`，由 Transformers 4.57.6 对自定义微批 mean loss 正确归一化；CPU 损失/梯度对照、真实 Trainer 尾批 16+5、双卡累积16冒烟均通过。四号机双4090 `EEC430004-4090` 的 `/opt/data/private/ECE4300-train-20261126/ga16-epoch1-v1/` 已完成309条/1epoch/20更新，训练596.45秒，峰值已分配10.76/16.80GiB；适配器保存和重载前向验证通过。仅验证流程，不声称效果提升，不自动继续训练，不改推理池。详见 `docs/QLORA_EPOCH1_VALIDATION.md`；轻量日志在本地 `local-backups/training/ga16-epoch1-v1/`，权重留服务器。下文历史“不启动训练”不覆盖本次明确授权，但不代表有后续epoch授权。

## 训练数据整理流水线（`training/`，2026-11-26）

用户明确规则：**同一任务跑了多次，始终是同一个任务，必须合并成一份规范化数据，不能按运行次数分成几个文件或几份数据源。** 该规则已落实为 `training/` 流水线，设计说明见 `docs/TRAINING_DATA_PIPELINE.md`；改动前先读该文档。

- `training/canonical.py`：把同一任务的多次运行合并为 `training/sources/ds22-sampled19-canonical.jsonl`。每题取**记录数最多**的快照（并列取最新），其余快照必须是通过 `(exploration_round, hypothesis_key, answer)` 签名校验的**记录级前缀**，否则报冲突。原始 `results/ds-stage22-*` 目录从此**只作存档**，不再是数据源；版本不一致的运行（旧版16题、错误provider、pilot、空壳目录）写在 `MANIFEST.json` 的 `excluded_runs`。
- `training/stage_sft.py`：按记录里的 `prompt` / `tool_trace` / `branch` 序号 / `plan_path` 重建 selection、expansion、candidate 三阶段输入（DS 当时未保存 wire 输入），套用线上 Qwen 的 `isolated_examples_v1` 模板，输出 `messages=[system,user,assistant]`。只收 `reference_hit` 且非截断、无协议错误的轮次，按 `hypothesis_key` 去重；重建输入必须仍容纳教师输出（Branch 在菜单内、Plan 等于 `plan_path`、candidate 等于 `record.answer`），否则丢弃并计入 issues。所有样本标 `input_reconstructed=true`、`plan_unverified=true`。
- `training/ingest.py`：append-only ledger（`training/ingest/ledger.jsonl`），按 `sample_id` 去重，保证以后新跑的数据只产出增量、不与已整理样本混淆。
- 当前产出：规范化 19 题/469 轮（14 完成、5 未完成，三个快照全部通过前缀校验）；数据集 `training/data/ds22-all-completed/` 共 **309 条**（103 个去重有效候选 × 三阶段），覆盖全部 14 道已完成题；`training/data/ds22-extended-n1/` 42 条为早期子集，已被包含。ledger 309 条。
- 用户口径（2026-11-26）：**训练数据稀少，已完成题全部入库，不做评测重合/等价族排除；评测阶段对测试集做减法**。因此 `g3_h3_b2_001`、`n2_inputs_00_01_outputs_01`、`n2_inputs_00_10_outputs_01` 及等价族 `n3_inputs_00_10_11_outputs_101` 已进入训练，评测时必须从测试集剔除或单独报告。
- 边界：不改动 v1 交接包；不操作 DS/Qwen 进程；27 道中等评测题属于另一任务、不在本流水线；`label_mask_checked=false`，tokenizer/loss mask 仍需在服务器实测后才能训练。
- 新运行接入：把目录加进 `canonical.py` 的 `runs` → 重跑① → 重跑② → `--append-ledger`，并检查 manifest 里没有 `reconstruction failed` / `plan does not match` / `candidate does not match`。

## 数据工作已改为交接：仅当前新版DS完成13题

用户停止本对话继续制作训练集，要求打包给别人接手。唯一交接源为`results/ds-stage22-merged/revision`中当前新版2.2、stage22_revision=true且completed=true正常结束的13题；不是旧版stage22-sampled的16题，也不是两者并集。另6题仅在pending清单列出，无部分作答记录。交接包`local-backups/handoffs/DS_MCTS22_completed13_handoff_20261126T063000Z.zip`（v1），含原始13份112轮、来源与SHA、过滤后的评分参考、27题测评隔离参考、离线重评分脚本、详细SFT处理及后续接入说明。3题测评重合隔离、1题内部变量交换等价去重后，9个题目族，51轮基本筛选通过、48个不同有效候选。候选不是SFT-ready，Plan/理由/复盘未因答案正确自动认证。因果两题尚未完成，本包训练候选没有因果领域。已解压独立重跑并验证摘要一致。后续6题由接手人按规则生成v2，不修改v1。本次未操作DS进程、未启动训练，也未影响Qwen19题运行。

## 当前活动任务：Qwen 2.2阶段隔离＋示例版19题

用户已授权先跑19题。当前自部署Qwen3-8B运行`2.2 / isolated_examples_v1`，采用已测试的复盘/选节点/扩展阶段隔离与示例，九题并行、开启思考、生成请求32768，九卡池8004。活动快照`releases/qwen-stage22-protocol-examples19-20261126`，结果与状态`results/qwen-stage22-protocol-examples19-20261126/state.json`，manifest.json含19题358参考解，release_manifest.json锁定源码。入口`server/run_qwen_revision19.py --run-name qwen-stage22-protocol-examples19-20261126 --protocol-label isolated_examples_v1`。已经核实9题启动、九个引擎各1请求。不要复用旧协议19题断点，不操作DeepSeek、不启动训练；原诊断已结束，以下停止说明只指旧协议运行。用户不要求持续监控。

## 最新任务：只处理Qwen协议诊断，正式19题已停止

用户明确本对话只处理Qwen，DeepSeek由另一对话负责，不操作其进程或结果。Qwen19题调度器及9个题目进程均已停止，断点保留，结果目录PAUSE保留，不自动恢复正式题库。诊断日志 `logs/qwen-protocol-diagnosis-20261126`。31次协议失败全部为布尔题选节点时输出Expression而非Branch。原题的最终回答指令与阶段指令混淆；候选方案把原题完整包成引用JSON数据，并在user消息末尾重申当前阶段契约，新增选节点、扩展与复盘的格式示例，保持严格解析器，不伪造Branch或静默接收错误格式。

三个失败输入各重复两次：原提示0/6格式通过，隔离6/6，隔离+示例6/6（包括Reason/Untried direction/Avoid字段）。另对两道布尔题和一道3D题各做两方案单轮完整测试，6/6无协议错误，但各方案仅2/3候选有效，不能声称示例提高解题覆盖率。候选实现`llm/qwen_stage_protocol.py`，只作用Qwen；正式发布快照保持不变。复盘也发现曾直接输出Structure，已按同样原则隔离并加示例，另做一道联合测试，结果见`e2e-review-isolated/summary.json`。不要直接重启原19题，候选方案尚未替换正式版本。

## 原19题启动信息（现已停止）

用户明确要求新版2.2使用本地自部署模型接口，不是DeepSeek。误启动的`experiments.run_stage22_revision19`及其子进程已全部停止，结果`results/stage22-revision19-20261126T042923Z`标记cancelled_wrong_provider，不计入正式比较，禁止重启该入口。已将阶段理由传递和复盘反馈适配到`llm/qwen_stage_revision.py`，服务器入口`server/run_qwen_revision19.py`。活动快照`releases/qwen-stage22-revision19-20261126`，输出`results/qwen-stage22-revision19-20261126`（state.json/manifest.json），自部署Qwen3-8B、8004九卡池、九题并行、思考开启、生成请求32768，原困难抽样19题358参考解。前九题均已确认新版标记与Qwen接口生效，九卡各有一个请求。不能将19道困难题总覆盖率与旧27道中等题总覆盖率直接比较。旧2.1正式27/27=51.16%；旧2.2正式27/27=37.19%，布尔题全零并有499次协议失败。训练仍取消，用户不要求持续监控。

## 最新优先级：Stage-MCTS 2.2 四题试验

最新用户确认总数为9卡：原6卡加四号机单5090和双4090，环境与API生成测试全部通过，已接入。最新明确顺序为：先“分阶段反思MCTS 2.2（Stage-MCTS）”全部27道中等评测题，再恢复两步反思MCTS 2.1未完成评测。用户已暂时取消2.1训练题生成、运行和导出，training_enabled=false；不要加--with-training，除非用户重新授权。不是先2.1再2.2，也不再用2.1填当前空位。2.2原四题不重跑，其余23题已入队；所有断点保留，普通MCTS 2.1继续暂停。最后验证active=9，全部为2.2，pending=18。

唯一活动调度器为 `server/run_mixed_mcts_queue.py --workers 9 --api-base http://127.0.0.1:8004/v1`，状态仍为历史目录 `logs/mixed-mcts-six-question-queue/state.json`，以workers字段为准。`priority_order`和`resume21_gate`明确2.2先全部完成。旧控制器已替换，禁止重新启动。2.2按结果文件完整路径接管，不中断；过渡期的2.1任务标记`drain21`，当前请求结束、日志出现Paused后才终止该进程，空位补2.2。这些drain任务暂占并发位置，避免过渡时实际超出九题。PAUSE_2.1在全部27道2.2正常结束后由程序移除，2.1按旧断点恢复。网关读取真实引擎负载，旧8003网关请求也计入忙闲判断；新任务走8004。节点及部署详情见 `docs/NODE4_DEPLOYMENT.md`，接入状态 `logs/node4-provision/activation-state.json`。

2.2快照 `releases/mcts-2.2-stage-pool6-32k`，入口 `server/run_stage22_pilot.py`，状态 `logs/mcts-2.2-stage-medium-pilot-32k/state.json`，结果 `results/server-mcts-2.2-stage-medium-pilot-32k`。四题：causal/n1_003、n1_002、n2_006，3d/g3_h3_b2_002。2.1对应覆盖率14.58%、22.92%、41.67%、55.56%，等权平均33.68%。这是偏向差题的探索样本，不能代表整个中等题库。

每轮单独调用：复盘、请求读树、选分支、提出一步、生成候选。选分支看完整本地树，后两阶段看选中路径和已有子步骤；历史答案保留。阶段输出与token/耗时分别记录。2.2继承2.1重复提交与理论穷尽规则，穷尽时须单独输出Proof，属于模型论证而非形式验证。各阶段请求32768并受总上下文32768约束。每题2.2从空树开始，与2.1相同搜索轮数；并非相同API/token预算。2.2前四题保留在pilot目录，后23题在 `results/server-mcts-2.2-stage-medium-followup-32k`；合起来27题，不重复计数。统计时同时给出总计和四题先导样本，不能仅按pilot_total=4汇报整个2.2。

## MCTS 方法版本与进度汇报

用户指定公开版本：1.0=提示词分支选择，2.0=工具读取完整树，2.1=2.0 加上允许重复解提交及基于理论分析的严格 EXHAUSTED 规则。定义在 `config/mcts_versions.json`，策略实现见 `agents/mcts_versions.py`。模型版本、普通/两步反思变体、题库难度与方法版本分别记录。目录中的 v1/v2 不是方法版本；`server-easy-tree-tools-v2` 属于方法 2.0。用户已取消旧2.0及简单题接续序列。当前2.1中等流程状态见 logs/mcts-2.1-temp-pool6/state.json：先评测27题，再生成独立训练28题并运行两个方法。输出分别为 results/server-mcts-2.1-medium-evaluation-32k 和 results/server-mcts-2.1-medium-training-32k；评测题不得用于训练导出。

新运行可用 `--mcts-version 1.0|2.0|2.1`，2.x 自动启用读树工具；结果记录版本与变体。2.1 另保存提示词原文和哈希。不同方法版本禁止共用断点；修改已发布提示词必须升版本。既有结果缺少版本时仅依据原配置推断并明确来源，不把历史结果改标为新方法。

进度汇报必须同时给出方法版本、完成题数/总题数、覆盖率及统计题数；未完成组的覆盖率按正常完成题等权平均。比较优劣时先对齐共同题目，并区分生成截断、网络错误、工具读取错误。

## 项目目标

用 HypoSpace 衡量答题 Agent 能否为同一组观测找到尽可能多的不同解释。每次提交的是一个候选解，评分器用标准解集合计算覆盖率。这里的“继续训练”指恢复评测进程、历史答案和搜索树，实验没有更新模型权重。

- `agents/` 实现直接回答、历史记忆和推理树等策略。普通 MCTS 由模型选择分支。两步反思先调用模型总结最近最多 8 条候选，再把总结附在下一次搜索提示词后。
- `experiments/direct_evaluation.py` 负责选择题目、执行实验、评分、保存断点和恢复状态。
- `experiments/boolean_scoring.py` 负责布尔表达式解析和结构判分。表达式需要满足观测和结构约束。同一真值表可以对应多个不同的逻辑结构。参考解只供评分，复盘使用历史输出。
- `HypoSpace/_HypoSpace-main/` 是上游基准代码。项目使用的固定题库在 `datasets/`。续跑时要使用相同的题库文件，加载时会校验文件哈希。

## 环境与模型

所有示例从 `Final Project` 根目录执行。Windows 可以用现有的 `.venv/Scripts/python.exe`，或先创建虚拟环境并用它安装依赖。

```powershell
python -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install -r requirements.txt
& $py -m unittest discover -s tests -v
```

首次配置时，把 `config/llm.json.example` 复制为 `config/llm.json`，从本机配置读取方舟 Coding Plan 密钥，或设置 `ARK_API_KEY`。现有配置应保留密钥，别用示例文件覆盖。长跑沿用 `timeout: 600` 秒。`config/llm.json` 已被 Git 忽略。

答题评测入口固定使用 **`deepseek-v4.1-flash-economy`**。请求走方舟 Coding Plan 的 `https://ark.cn-beijing.volces.com/api/coding/v3`。评测结果记录请求模型和每次响应返回的模型名。通用的 `python -m llm` 可以单独检查接口。上游套件与历史答题记录中的模型字段用于溯源。

[方舟 Coding Plan 支持模型与接入说明](https://docs.volcengine.com/docs/ark/coding-plan-personal-ai-other-tools?lang=zh)

## 选择实验入口

- 正式题库用 `experiments.test_hard_balanced`，共 **136 题、3,600 个标准解**。因果图 8 题 1,203 解，3D 结构 31 题 1,197 解，布尔表达式 97 题 1,200 解（含扩展集 17 题）。首题有 638 个标准解，启动后可能长时间显示 0 道已处理。
- 验证题库用 `experiments.test_hard_boolean`，固定一道 40 解布尔题。旧的固定三道高难题用 `experiments.test_hard_quick`。
- 历史实验入口包括 `test_full_baseline`（105 道基础题）、`test_medium`（30 道题）和 `test_hard_full`（原高难 90 题）。

```powershell
$py = ".\.venv\Scripts\python.exe"
$runner = @("-u", "-m", "experiments.test_hard_balanced", "--provider-token-limit")
$tree = @("--method", "mcts-tree", "--tree-policy", "llm", "--duplicate-reward", "-1", "--exploration-list", "off")
$two = @("--two-step-reflection", "--unbounded-tree")
& $py @runner @tree @two --plan
& $py -m experiments.test_hard_boolean @tree @two --provider-token-limit --search-cap 120 --plan
```

`--plan` 只在本地装载题库和显示预算。移除它会使用 API 并生成独立的时间戳结果文件。要做同一模型的正式比较，可以分别运行 `@runner --method direct`、`@runner --method memory`、`@runner @tree --child-limit 7` 和 `@runner @tree @two`。每行通过 `& $py` 执行，并行时分别放在不同终端。每个进程使用唯一的结果文件，同一个断点只能有一个写入进程。建议同时为每组指定各自的 `--pause-file`，便于单独暂停。

直接法和记忆法每题最多提交“标准解数 + 5”次。普通 MCTS 的内部搜索预算是其两倍，且沿用每节点最多 7 个子节点的历史配置。当前两步反思组也用该内部搜索预算，树深度、子节点数和提交数均无独立上限。一次内部搜索最多使用两次 API 请求，复盘失败时这一轮可能没有进入搜索。有效的重复解仍提交。找全标准解会提前结束。正式组遇到连续 3 次请求失败会记录已有结果并跳到下一题，汇总时应注明这些题。

## 现有断点与恢复

2026-11-17 18:06 保存的正式进度，以下旧运行请求模型均为 `ark-code-latest`。以本地 JSON 的最新内容为准。

- 直接法 `results/direct_hard_balanced_checkpoint.json` 已处理 136 题，其中 25 题因请求失败跳过。按题等权覆盖率（含跳过题已有结果）15.56%。
- 记忆法 `results/memory_hard_balanced_checkpoint.json` 已处理 49 题，其中 9 题因请求失败跳过。当前布尔题 `n2_inputs_00_01_outputs_10` 已找到 9/12 解。
- 普通 MCTS `results/mcts-tree_hard_balanced_checkpoint.json` 已处理 40 题，其中 13 题因请求失败跳过。当前布尔题 `n1_inputs_00_outputs_1` 已找到 21/25 解。
- 两步反思 `results/mcts-tree-list-off-branch-examples-two-step-unbounded_hard_balanced_20261117T052744652867Z.json` 正在做第 1 题。146 次内部搜索找到 144/638 解。文件名没有 `unbounded` 的两步反思正式断点是更早的配置。

三组旧进程已设置暂停。暂停文件为 `results/PAUSE_HARD_BALANCED`（记忆法、普通 MCTS 共用）和 `results/PAUSE-hard-balanced-two-step`（两步反思）。进程在当前请求结束后暂停，旧进程已经载入原模型。切换到固定模型时保留这两个暂停文件，以免误唤醒旧进程。

如果要用新模型接着已有进度跑，使用 `--continue-from`。以下三条分别运行，可并行。每条会创建独立的新结果文件并记录 `model_transition`，保留旧断点、已处理题和当前题状态。当前题可能包含两种模型生成的答案。之前因失败跳过的题也会保留，需要补测时单独安排。要求整个结果都出自固定模型时，应从新结果文件跑完整题库。

```powershell
& $py @runner --method memory --pause-file results/PAUSE_DEEPSEEK_MEMORY --continue-from results/memory_hard_balanced_checkpoint.json
& $py @runner @tree --child-limit 7 --pause-file results/PAUSE_DEEPSEEK_MCTS --continue-from results/mcts-tree_hard_balanced_checkpoint.json
& $py @runner @tree @two --pause-file results/PAUSE_DEEPSEEK_TWO_STEP --continue-from results/mcts-tree-list-off-branch-examples-two-step-unbounded_hard_balanced_20261117T052744652867Z.json
```

新进程的输出会显示 `Saving results to ...`。用同一套入口和参数，在对应的 `--pause-file` 路径创建空文件以暂停，删除该空文件可恢复仍存活的进程。若进程已退出，改用 `--resume results/<新断点文件名>.json`，并先确认原进程已退出。恢复会跳过已处理题并重放当前题的历史以重建搜索树。断点模型、题库文件和方法配置必须匹配。历史结果如果位于另一台机器，先把 JSON 断点复制到 `results/`，并复用相同版本的题库和评测代码。

## 查看结果

- `cases` 是已处理的题，`interrupted_case` 是正在做的题。用 `stop_reason=request_errors` 识别因请求失败跳过的题，`completed=true` 表示已遍历题库。
- 单题覆盖率是找到的不同标准解数除以该题标准解数，对题取等权平均得到归一化覆盖率（`summary.all.recovery`）。汇总只对正常结束的题（`stop_reason != request_errors`）取平均，因请求失败跳过的题不计入覆盖率、单独列出待补测。`summary.*.normal_cases` 记录正常结束题数，`python -m experiments.rescore_summary <结果文件...>` 用该口径重算并导出 `skipped_cases_to_rerun` 清单。
- 比较方法时优先对齐同一批题、同一个模型及相同停止规则。分别报告内部搜索轮数、答案提交数、API 请求数与 token。两步反思的 `reflection_response` 记录额外 API 调用，整轮结束时汇总到 `api_calls`。

## 服务器与本地评测启动流程（2026-11-24）

### SSH 连接

服务器 IP 固定 `10.1.3.100`，每个任务实例一个端口（存活 3-4 天，端口会变），配置在 `~/.ssh/config`：

- `EEC4300` = `10.1.3.100:25910`（自己，主机 `interactive24187`，密钥 `~/.ssh/lab4090_codex`）
- `EEC430002` = `10.1.3.100:25640`（同学，主机 `interactive86262`，密钥 `~/.ssh/classmate_ed25519`）

### 服务器端（Qwen3-8B，简单题库 105 题）

服务器三个 API（都是回环地址，不对外暴露）：

- `8000`：GPU0 的 vLLM 模型副本
- `8001`：GPU1 的 vLLM 模型副本
- `8002`：FastAPI 网关（`server/api_gateway.py`），按在途 token 负载把请求分到 8000/8001，是评测的唯一入口

**启动 vLLM + 网关**（`cd` 到项目根目录后）：

```bash
setsid bash server/start_api.sh 0 8000 > logs/api_8000.log 2>&1 < /dev/null &
setsid bash server/start_api.sh 1 8001 > logs/api_8001.log 2>&1 < /dev/null &
setsid .venv-inference/bin/python -m uvicorn server.api_gateway:app --host 127.0.0.1 --port 8002 > logs/gateway.log 2>&1 < /dev/null &
```

> 用 `setsid ... &` 让进程脱离 SSH 会话，否则 SSH 命令会卡住直到超时（进程本身仍会启动）。

**跑四方法评测（分两个环境，因为树工具依赖独立的 `.venv-tree-tools`）**：

direct / memory 用 `.venv-inference`（无树工具）：

```bash
export ECE_THINKING=1 ECE_MAX_TOKENS=16384 ECE_WORKERS_PER_METHOD=2 ECE_DATASET_SUITE=easy \
       ECE_METHODS=direct,memory ECE_RESULT_DIR="$PWD/results/server-easy-dm" ECE_LOG_DIR="$PWD/logs/easy-dm"
setsid .venv-inference/bin/python -u server/run_evaluations.py > "$ECE_LOG_DIR/runner.log" 2>&1 < /dev/null &
```

mcts / two-step 用 `.venv-tree-tools`（树工具，需要 `qwen_agent`）：

```bash
export ECE_THINKING=1 ECE_TREE_TOOLS=1 ECE_MAX_TOKENS=16384 ECE_WORKERS_PER_METHOD=1 ECE_DATASET_SUITE=easy \
       ECE_METHODS=mcts,two-step ECE_RESULT_DIR="$PWD/results/server-easy-tree-tools-v2" ECE_LOG_DIR="$PWD/logs/easy-tree-tools-v2"
setsid .venv-tree-tools/bin/python -u server/run_evaluations.py > "$ECE_LOG_DIR/runner.log" 2>&1 < /dev/null &
```

关键坑：**树工具必须用 `.venv-tree-tools/bin/python`**（`qwen_agent` 只装在这个环境里），`run_evaluations.py` 里 `sys.executable` 会传给 smoke 子进程，用错环境会导致 tree-tools 报 `ModuleNotFoundError: qwen_agent`。启动脚本模板见 `server/run_easy_tools.sh`。结果目录要换新的，别复用旧目录（旧 smoke 文件的 `final_answer_missing=True` 会让 check_smoke 误判失败）。

### 本地（deepseek，困难题库抽选的小题库）

- 小题库清单：`datasets/sampled.json`（19 题 358 解，每类约 120 解，手动挑选、跳过解数过大的题如 causal 638 解）。
- 树工具：`llm/deepseek_tree_tools.py` 的 `DeepseekTreeToolClient`，用方舟 Coding Plan 的**原生 function calling**（`tools` 参数 + `tool_calls`），方舟 deepseek 已确认支持；思考在 `reasoning_content` 字段。

跑四方法（direct/memory 无树工具，mcts/two-step 用 `--tree-tools`）：

```bash
python -m experiments.run_sampled --workers 4
# 或只跑部分方法：--methods mcts,two-step
```

结果落到 `results/sampled-local/<dataset>-<case_id>.json`，断点续跑（文件存在会自动 `--resume`）。

### 树工具两套实现对照

| | 服务器 Qwen | 本地 deepseek |
|---|---|---|
| 文件 | `llm/tree_tools.py` | `llm/deepseek_tree_tools.py` |
| 工具解析 | Qwen-Agent Hermes 风格（`<tool_call>` prompt 注入） | OpenAI 原生 function calling |
| 环境/服务 | `.venv-tree-tools` + vLLM | 方舟 Coding Plan |
| 触发参数 | `--tree-tools`（需 `--enable-thinking --base-url`） | `--tree-tools`（无 `--base-url`，用方舟） |

两者共用 `agents/tree_file_tools.py`（`read_tree`/`read_node`/`read_subtree`，每次从磁盘重读树快照）和 `llm/tool_response.py`（`tree_candidate` 分离思考并校验 `Branch:` 编号）。流程都是：写树快照到 `.tree.json` → 模型先 `read_tree` 至少一次 → 按 `Branch / Plan / candidate` 格式作答 → 评分更新树。

### 跨机器 LLM 分派接口（2026-11-24）

一号机 8002 的 `api_gateway.py` 是统一的分派接口：后端池通过环境变量 `ECE_BACKENDS` 配置（逗号分隔的 OpenAI 兼容 base URL，不含 `/v1`），按每个后端的**在途 token 负载**把非流式 chat completion 分派给最空闲的后端，空闲的卡自动接活。

```
ECE_BACKENDS=http://127.0.0.1:8000,http://127.0.0.1:8001,http://10.106.31.1:8000
```

- 一号机双 4090 → 8000（GPU0）、8001（GPU1）；三号机 5090 → `10.106.31.1:8000`。
- 评测进程统一连一号机 `http://127.0.0.1:8002/v1`，不指定具体卡；`/health` 返回 `backends`、`estimated_load`、`inflight_requests` 可看分派情况。
- 三号机 vLLM 要绑定 `0.0.0.0`（`server/start_api.sh` 里 `--host 0.0.0.0`），否则一号机无法访问。

### 三号机（EEC430003，单 RTX 5090）部署

`EEC430003` = `10.1.3.100:25871`（主机 `interactive28913`，密钥 `~/.ssh/lab4090_codex`）。部署步骤：

1. 同步代码 + tests（从本地 `tar czf - agents experiments llm datasets server tests HypoSpace requirements*.txt | ssh EEC430003 'tar xzf -'`）。
2. `bash server/bootstrap.sh`（建 `.venv-inference` + 装 vLLM + ModelScope 下载 Qwen3-8B）。
3. 装树工具环境并复用推理库：`.venv-tree-tools` 装 `qwen-agent` 后，写 `.pth` 指向 `.venv-inference` 的 site-packages（见一号机 `ece_inference_runtime.pth`）。
4. 单卡跑评测要把 `server/run_evaluations.py` 里 smoke 端口改成可配置（`ECE_API_PORTS=8000`），并把 `api()` 改为读 `ECE_API_BASE`。
5. 启动 vLLM：`setsid bash server/start_api.sh 0 8000`（单卡一个后端）。

三号机只跑 vLLM（纯计算节点），评测进程都在一号机、统一走 8002 分派。

当前生成配置：用户要求32768请求额度，按32768-input_tokens-8使用剩余上下文，不再额外限制为16K。旧16K记录保留，迁移后应汇报generation_limit_transitions，不能把旧记录说成32K生成。

最新用户选择：普通MCTS 2.1暂停，只继续两步反思MCTS 2.1，每组6题并行，包含后续训练题流程。不要移除PAUSE_mcts或重新启动普通组。

当前临时六卡池127.0.0.1:8003包括新增lab4090双卡与lab5090单卡；节点目录 /opt/data/private/ECE4300-temp-api-20261125，其他工作区不操作。凭据不得提交。
