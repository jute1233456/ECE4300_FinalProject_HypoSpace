# ECE4300 服务器操作

服务器根目录：`/opt/data/private/ECE4300 Final Project`。所有命令从该目录执行。

## API

- GPU 0：`http://127.0.0.1:8000/v1`，GPU 1：`http://127.0.0.1:8001/v1`。
- 正式评测统一入口：`http://127.0.0.1:8002/v1`，根据在途请求的估计 token 负载分配到相同的两份 BF16 模型。它只支持本次使用的非流式聊天请求，健康状态为 `http://127.0.0.1:8002/health`。
- 模型名称均为 `Qwen3-8B`。仅绑定回环地址，不对外暴露端口。
- 明确指定 `chat_template_kwargs: {enable_thinking: false}`；评测使用 temperature 0.7、top_p 0.8、top_k 20、max_tokens 4096。
- 不对每次请求固定相同 seed，以免直接法相同提示词产生完全相同候选。连续批处理的采样并非逐次位级可复现，结果需后续多次重复验证。
- 启动：`bash server/start_api.sh 0 8000`。后台启动脚本见 `server/pipeline.sh`，已有进程运行时不要重复启动。
- 健康检查：`curl -f http://127.0.0.1:8000/health`。

Windows 本机访问：

```powershell
ssh -N -L 127.0.0.1:18000:127.0.0.1:8000 ECE4300
```

保持终端打开，本机 API 地址即 `http://127.0.0.1:18000/v1`。客户端需要 API key 字段时填 `local`，它是占位符。

2026-11-20 已在 Windows 后台建立该转发，并实际生成 `WINDOWS_API_OK`。本地 PID 记录在 `server/windows-tunnel.pid`，验证响应在 `server/windows-api-smoke.json`。本机重启或连接断开后，重新执行上面的命令；端口已被本项目隧道占用时不要重复启动。

## 评测

`.venv-inference/bin/python server/status.py` 查看进度。

正式题库每方法 136 题，逐题保存于 `results/server/{direct,memory,mcts,two-step}/`。每种方法至多 4 个题目进程；题目内搜索顺序不变。原模型和量化加载的微调检查彼此隔离，正式 API 使用 BF16 原模型。

运行中曾观察到 GPU 1 的 KV 缓存接近 97% 且发生抢占，因此在当前请求完成后保存断点、停止旧评测进程，并通过 8002 统一调度恢复。已有候选与搜索树保留，端点变化记录在每题 `api_routes`；模型与采样设置不变。

暂停某方法：`touch results/server/PAUSE_direct`（或 memory、mcts、two-step）；移除对应文件恢复。停止协调程序前先暂停并等待当前调用完成，避免同一断点多个写者。

日志：`logs/pipeline.log`、`logs/api_8000.log`、`logs/api_8001.log`，每题日志与结果同目录。

汇总：`.venv-inference/bin/python server/report_results.py`，生成 `results/server/comparison.json` 与 CSV。失败重试的旧结果以 `.failed-*` 保留。

## 微调

当前没有使用正式测试答案训练。`server/check_trainability.py` 只验证 QLoRA 前向反向，不更新或保存原模型。`server/train_qlora.py` 是后续 SFT 入口，输入 JSONL，每行有 prompt、response。

实际训练前停止所选 GPU 上本项目的 API，准备独立训练集并检查与测试集的结构重叠；不能同时占用同一 GPU 的全部显存。

## 文件与环境

没有上传本地密钥配置、历史结果、Windows 虚拟环境或 Git 数据。保留上游代码 README 与署名。

精确依赖见 `logs/environment.lock.txt`。为匹配 vLLM 0.11.0，实际 GPU PyTorch 由其依赖固定为 2.8.0 / CUDA 12.8；使用系统 Python 3.10 的独立虚拟环境，不改系统 Python。

模型源见 `logs/model_source.json`；配置校验和与文件大小见 `logs/model_manifest.json`。

## 思考模式新轮

2026-11-20 用户要求开启思考模式重跑四种方法。旧非思考任务已保存断点并停止，旧暂停文件仍保留。

启动脚本：`bash server/run_thinking.sh`。已有新轮运行时不要重复启动。

新结果：`results/server-thinking-v1/`；新日志：`logs/thinking-v1/runner.log`；完成标记：`logs/thinking-v1/evaluations.complete`。

```bash
.venv-inference/bin/python server/status.py --run-dir results/server-thinking-v1 --log-dir logs/thinking-v1
```

新轮每次输出上限 16384（含思考），上下文 32768，输出按剩余上下文自动缩小且记录实际上限；提示内容不会被自动截断。生成 text 仅含最终回答，思考保存在 reasoning_text，避免把中间尝试送入评分器和记忆。

新轮暂停文件位于 `results/server-thinking-v1/PAUSE_direct`、`PAUSE_memory`、`PAUSE_mcts`、`PAUSE_two-step`，不要操作旧轮暂停文件来控制新轮。
