# ECE4300 Final Project 交接说明

## 最新优先级：Stage-MCTS 2.2 四题试验

最新用户要求优先测试“分阶段反思MCTS 2.2（Stage-MCTS）”，只跑2.1已完成题目的一部分，优先低覆盖率题。2.1暂时暂停，保留所有断点；2.2四题结束后必须自动恢复两步反思MCTS 2.1原评测/训练流程，普通MCTS 2.1继续暂停。

2.2快照 `releases/mcts-2.2-stage-pool6-32k`，入口 `server/run_stage22_pilot.py`，状态 `logs/mcts-2.2-stage-medium-pilot-32k/state.json`，结果 `results/server-mcts-2.2-stage-medium-pilot-32k`。四题：causal/n1_003、n1_002、n2_006，3d/g3_h3_b2_002。2.1对应覆盖率14.58%、22.92%、41.67%、55.56%，等权平均33.68%。这是偏向差题的探索样本，不能代表整个中等题库。

每轮单独调用：复盘、请求读树、选分支、提出一步、生成候选。选分支看完整本地树，后两阶段看选中路径和已有子步骤；历史答案保留。阶段输出与token/耗时分别记录。2.2继承2.1重复提交与理论穷尽规则，穷尽时须单独输出Proof，属于模型论证而非形式验证。各阶段请求32768并受总上下文32768约束。四题从空树开始，与2.1相同搜索轮数；并非相同API/token预算。四题并行、六卡池可用，未占满六卡属于串行阶段依赖。2.2没有全题库或训练题扩展。

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
