# ECE4300 期末项目

本项目探索提高大语言模型在 HypoSpace 基准测试上的表现。

## 当前工作入口（2026-11-27）

当前主线是 Qwen3-8B + Stage-MCTS 2.2（`isolated_examples_v1`），在 quick3（3 题 45 解）和 mixed100_v1（25 题 293 解）上评测。已完成 QLoRA SFT（`sft-pruned-v1-ep1`/`ep2`）与 DPO v1/v2/v3；V3 混合反思（反思用 V3、其余步骤用 SFT ep1）和全阶段 V3 对照正在验证。

- 当前结论：**SFT 相对 base 有明确提升**（quick3 60.0%→80.0%）；**DPO 尚未显示稳定的额外收益**。V3 结果未完成，不提前下结论。
- 实验与监控：[监控口径](docs/CC_MONITOR_CURRENT.md)、[V3 混合实验](docs/DPO_V3_HYBRID_RUN.md)、[V3 mixed100 队列](docs/DPO_V3_MIXED100_QUEUE.md)。
- 数据与训练：[数据流水线](docs/TRAINING_DATA_PIPELINE.md)、[DPO 数据集 v3](docs/DPO_DATASET_V3.md)、[QLoRA 首轮验证](docs/QLORA_EPOCH1_VALIDATION.md)。
- 服务器数据归档：[紧凑归档](training/server-archive-20261127/)（不含权重）、[归档说明](docs/SERVER_DATA_ARCHIVE.md)。
- 报告大纲：[docs/PROJECT_REPORT_OUTLINE.md](docs/PROJECT_REPORT_OUTLINE.md)。
- 方法版本与交接：[config/mcts_versions.json](config/mcts_versions.json)、[AGENTS.md](AGENTS.md)。

早期 2026-11-25 的 MCTS 2.1 说明保留在 [docs/CURRENT_RUN.md](docs/CURRENT_RUN.md)，仅作历史，不代表当前配置。

### 训练数据说明

`training/sources/ds22-sampled19-canonical.jsonl` 有 112 MB，超过 GitHub 单文件限制，因此**不提交原文件**，提交同目录的 `.jsonl.gz`。复现前先解压：

```bash
gzip -dk training/sources/ds22-sampled19-canonical.jsonl.gz
```

其余已审计训练数据、DPO 数据与运行参数在 `training/data/`、`training/audits/`、`training/server-archive-20261127/`。

## 官方代码来源

`HypoSpace/_HypoSpace-main/` 为下载解压的官方代码，来源：
https://github.com/CTT-Pavilion/_HypoSpace

该目录保留作者的说明和署名。本项目复用其中的题库生成器和评分代码。

## 项目记录

后续在本仓库记录环境配置、改进方法、实验设置和可复现结果。
请勿提交模型接口密钥；本地密钥应通过环境变量或未纳入版本记录的配置保存。

## 调用火山方舟 Coding Plan

在本文件所在目录运行，要求 Python 3.10 或更新版本，模型调用模块无需安装额外依赖：

```powershell
python -m llm "请用 Python 编写一个计算列表平均值的函数"
```

先将 `config/llm.json.example` 复制为 `config/llm.json`，在本地文件的
`ark_coding_plan.api_key` 中填写密钥，可同时设置 `model` 和 `timeout`（秒）。
本地文件已加入 `.gitignore`，只有不含密钥的示例文件可以上传。
客户端自动定位项目根目录下的配置，不依赖当前工作目录；也支持 `--config 文件路径`。
优先级：显式参数 > 本地配置 > 环境变量 > 默认值。空密钥会回退读取 `ARK_API_KEY`。
不会自动读取 `.env` 文件。自定义配置路径必须自行加入忽略规则。
通用接口检查命令可通过 `--model 模型名称` 覆盖配置。答题评测固定使用
`deepseek-v4.1-flash-economy`，并在结果中记录返回的模型名称。
可用 `--max-tokens 2048` 调整输出上限；返回的 `finish_reason=length` 表示可能被截断。

### 接口设计

- `LLMClient` 是抽象父类，业务代码只依赖其 `generate()` 方法。
- `OpenAICompatibleClient` 实现兼容协议的 HTTP 通信。
- `ArkCodingPlanClient` 固定方舟 Coding Plan 地址并读取配置。
- API key 是实例配置，不是每个密钥都创建一个子类；不同协议可新增父类的实现。
- 返回值包含文本、实际模型名、用量和停止原因；请求失败抛出异常，不作为模型答案计分。
- 不自动重试，避免重复消耗额度；目前支持同步文本调用。

```python
from llm import LLMClient, ArkCodingPlanClient

def solve(client: LLMClient, question: str) -> str:
    return client.generate(question).text

# 先在 config/llm.json 中填写密钥
client = ArkCodingPlanClient()
```

离线测试（不调用真实模型、不消耗额度）：

```powershell
python -m unittest discover -s tests -v
```

官方参考：
- https://docs.volcengine.com/docs/ark/coding-plan-personal-get-started
- https://docs.volcengine.com/docs/ark/coding-plan-personal-ai-opencode

Coding Plan 专用地址为 `https://ark.cn-beijing.volces.com/api/coding/v3`。
不要替换为普通方舟 `/api/v3` 地址，官方说明该地址可能产生额外费用。
使用范围和额度以你的套餐说明为准；本模块的接入完成不代表已验证批量评测可用。

## DirectAgent 快速测试与正式基线

两个入口使用同一套 HypoSpace 提示词、解析器和验证器。DirectAgent 每次独立回答，不传历史回答，参考答案只用于评分。

```powershell
.venv/Scripts/python.exe -m experiments.test_direct_agent
.venv/Scripts/python.exe -m experiments.test_full_baseline --plan
.venv/Scripts/python.exe -u -m experiments.test_full_baseline
```

快速测试固定三题，每题默认调用次数等于不同参考答案的数量，共 18 次。因果题干预 A 后 B 和 C 均受影响，体素题只观察到中心列，布尔题给出 (0,0)→0 和 (1,1)→1。题号在快速测试入口中列出。历史基础基线遍历 `datasets/` 中的全部 105 题，每题默认调用次数等于不同参考答案的数量，共 352 次。两个入口均可通过 `--queries` 指定每题固定调用次数，通过 `--plan` 离线查看工作量。

题库采用官方 README 的示例设置，种子为 33550336。因果图为 3 节点，共 61 题。体素为 3×3、最高 3 层、俯视图固定一格占用，共 9 题。布尔逻辑为双变量、basic 运算符、最大深度 2，共 35 题。后续比较应复用这些题库文件和调用预算。

结果分别保存到 `results/direct_quick_*.json` 和 `results/direct_baseline_*.json`。文件记录题库哈希、题号、提示词、模型与原始响应、用量、调用错误、逐题分数及按领域和全题等权平均的汇总。有效率与独特性以计划调用数为分母，覆盖率以不同参考答案数为分母。连续三次请求错误会终止，已完成题目保留在文件中，完整运行标记为 `completed: true`。

### 固定三题的难度

这里按题目结构和找全答案的难度分层，正式题库仍包含全部 105 题。

- **简单 · 体素 `g3_h3_b1_005`**。俯视图只占中心一格，符合重力约束的柱高为 1、2 或 3，共 3 个答案。
- **中等 · 因果 `n1_004`**。干预 A 后 B、C 均受影响。需区分直接边和经另一节点传递的影响，并保持无环，共 5 个图。
- **困难 · 布尔 `n2_inputs_00_11_outputs_01`**。找到一个可行表达式很容易。找全 10 种结构不同的机制需要考虑嵌套式和与简单表达式真值相同的机制。

三题默认预算分别是体素 3 次、因果 5 次、布尔 10 次。每种方法使用同一题目的同一预算。此前使用每题两次或六次的快速结果属于旧预算。

### 30 题方法比较

`datasets/medium_cases.json` 固定 11 道因果、9 道体素和 10 道布尔题。体素题库共 9 道，全部纳入。其余两类按参考答案数、观测数和题号排序后等间隔选取。

运行 `python -m experiments.test_medium --method METHOD`。METHOD 可取 direct、memory、mcts、mcts-breadth、mcts-recent、mcts-llm。追加 `--plan` 可查看最大预算。每题提交上限为答案数 + 5，找齐即停。MCTS 的内部调用上限是提交上限的两倍，每个有效候选都立即提交，包括重复解。无效候选和请求错误消耗调用预算。中等测试遇到空模型响应时在当前预算内继续，连续连接或服务错误仍会中止并保留已完成结果。LLM 选择分支和沿分支作答合并到一次请求，记录分支编号。比较时同时报告覆盖率、提交数和实际调用及 token 消耗。

### 高难题库

运行 `python -m experiments.generate_hard_datasets` 生成 `datasets/hard/` 下三类各 30 题。体素是 3×3、最高三层、固定三个占用列，从官方生成器的 84 种俯视图用种子 33550336 抽取。布尔使用官方 full 运算符、最大深度 2，从 80 道候选中按参考解数量取前 30。因果使用六个节点和六次单节点干预，沿用官方大图生成器的 10 边上限。枚举所有合法 DAG 后，按参考解数量选取 30 种观测结构，排除只重命名节点的重复题。参数是本项目对论文最高主实验档次的实现。

高难题测试入口：

```powershell
.venv/Scripts/python.exe -m experiments.test_hard_quick --method memory --plan
.venv/Scripts/python.exe -m experiments.test_hard_full --method memory --plan
```

快速入口固定选 57 解因果题、27 解体素题和 40 解布尔题。完整入口遍历三类各 30 道，共 90 道。移除 `--plan` 即发起真实模型调用。方法选项沿用 30 题实验。默认每题最多提交参考解数加 5 次，找齐即停。直接法和记忆法的调用上限分别是快速入口 139、完整入口 6258。MCTS 的内部调用上限为提交上限的两倍，重复有效解仍计一次提交。

### 暂停和继续测试

在项目根目录创建暂停文件。默认使用同一路径的测试会在当前 API 请求结束后暂停，保留当前题的历史和搜索状态。

```powershell
New-Item -ItemType File -Force results/PAUSE
Remove-Item -LiteralPath results/PAUSE
```

第一条暂停，第二条继续。可用 `--pause-file results/PAUSE_memory` 给单个进程指定独立的暂停文件。暂停期间保持测试进程运行。

### 高难三题第二轮

`python -m experiments.diagnose_hard_quick <结果文件...>` 按题统计新解、重复、无效候选和接口错误。第二轮的 `memory-structured` 仅记录有效历史，并给出与题型匹配的结构修改提示。运行命令为 `python -m experiments.test_hard_quick --method memory-structured --max-tokens 4096`。沿用第一轮的题目、提交上限和停止条件。生成提示和最大输出长度同时改变，因此此轮衡量两项改动的合并效果。空响应记录服务返回的结束原因及 token 用量（如果服务提供）。

### MCTS 推理树排列实验

高难三题入口使用 `--method mcts-tree --provider-token-limit`，组合 `--tree-policy`（uct、breadth、recent、llm）、`--child-limit`（5、7）、`--duplicate-reward`（0、-1）。保留两层推理步骤和原有预算。扩展时展示已有子步骤，去除规范化后重复的文字。模型在扩展模式下返回 EXHAUSTED 会关闭节点，达到子节点上限的节点进入直接作答模式。有效候选立即提交，重复也计一次；无效候选计内部调用。运行 `python -m experiments.summarize_tree_sweep` 查看每次提交后的累计新解数。批量运行使用独立暂停文件 `results/PAUSE_TREE_SWEEP`。

排列实验保留 14 组。最少访问优先只按访问次数选节点，因此只保留 5 和 7 个子节点各一组，沿用已有进度的重复奖励 −1 配置。汇总跳过其余两个最少访问优先结果。

### 待处理：LLM 分支编号

等探索列表开关对照结束后统一修改提示词。把 `Branch: <编号>` 明确写为多候选时的必填首行，并给出两个短示例：EXPAND 依次输出 Branch、Plan 和完整答案，SOLVE 输出 Branch 和完整答案。示例使用占位符，避免给出当前题的解。修改后比较多候选场景的分支未确认率，以及不同有效解数量和覆盖率。

### 实验口径

正式题库使用 `experiments.test_hard_balanced`，每类约 1200 个参考解。单题验证题库使用 `experiments.test_hard_boolean`，用于验证算法改动。两步复盘使用 `--two-step-reflection`，每轮先独立复盘再搜索，比较时记录搜索轮数与实际 API 调用成本。
