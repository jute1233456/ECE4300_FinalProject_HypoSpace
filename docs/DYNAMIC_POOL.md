# 九卡动态池与三题验证：已部署

2026-11-26完成首次迁移。动态池不是仅离线验证：已在九张GPU上部署服务、恢复旧19题，并启动三题验证。首次迁移使用已有搜索轮次边界保存和暂停；后续单卡变更只排空该卡的当前API，不需要全局暂停题目队列。

## 当前运行位置

一号机 `EEC4300`：`/opt/data/private/ECE4300-dynamic-pool-20261126/`。

- 统一网关8005，代码 `code/server/dynamic_api_pool.py`，私有配置 `pool.json`。
- 旧8002/8003/8004已改为兼容转发，全部进入同一网关，避免旧入口绕过排空管理。
- 原任务入口`/v1`使用baseline/shared卡，验证入口`/validation/v1`使用validation/shared卡；两类请求都必须匹配实际模型名，shared卡可承接两类。
- 每台VM的管理代理监听自身内网IP:28280，只管理本项目登记的服务；新推理端口28000/28001。
- 服务及管理令牌仅在私有文件，权限600，目录700；管理接口和远程推理请求校验令牌。本机旧客户端继续使用原入口。
- 原19题仍在原结果目录和进程中继续，不换模型、不重新做题。

GPU ID：`main-g0/g1`、`node3-g0`、`lab5090-g0`、`lab4090-g0/g1`、`node4-5090-g0`、`node4-4090-g0/g1`，共9张。每张都已校验并加载相同基础模型与五个原微调适配器。

## 三题验证

用户将6题减少到3题，每类一题，均来自原先已排除教师19题等价族的冻结清单：

- 因果 `n2_006`：24参考解。
- 3D `g3_h3_b2_002`：9参考解。
- 布尔 `n2_inputs_00_01_outputs_00`：12参考解。

共45参考解，清单 `datasets/quick_validation3_v1.json`。原版、原线性1/2epoch、新反比例1/2/4epoch，共6模型×3题=18任务，评分与MCTS2.2阶段隔离示例协议、搜索预算、32K请求额度保持原样。

新唯一验证目录：一号机动态池根目录的 `quick3/`，控制器 `code/server/run_dynamic_validation.py`，启动PID154578。`tasks.json`为持久任务状态，`state.json`为运行状态，`summary.json`为汇总，`results/<模型>/`为逐题历史与搜索树。旧四号机 `quick6-controls-v1` 的调度器和题目进程已在保存暂停后停止，旧文件完整保留归档，不得重启该旧调度器。

迁移校验：base旧3D题17轮、linear-ep1旧3D题12轮，逐条记录签名前缀一致；新任务使用`--resume`接续，已确认恢复后进入实际API调用。只有三题全部完成才报告最终平均覆盖率。

并发不是固定两题：根据健康GPU数、人工worker上限和旧19题剩余数量动态计算。默认worker上限9，为最多3道旧题保留请求容量；现阶段6道验证＋3道旧题可同时运行，已实测九个后端都有请求。旧题减少后，验证可自动增加到7/8/9并行。减少卡数不会杀掉已启动的题目进程，它们后续API会由剩余卡承接。

## 实机训练／推理切换验证

一号机 `swap-validation.json`、`swap-events.json` 保存实际过程；四号机4090同根目录 `training-swap.log`、`training-swap-result.json` 保存真实训练记录。

验证过程：

1. 两张4090各有一个2048token诊断请求在生成。它们属于排空诊断，不是正式题库，也不作为覆盖率数据；正式评测仍请求32768。
2. 卡角色变为draining，当前API继续，新请求停止派往这两张卡。
3. 两个请求均完整返回HTTP200后，停止相应模型服务、确认GPU没有计算进程，卡角色转training。
4. 同一VM双卡执行两次真实QLoRA参数更新，耗时11.64秒，无OOM，正常退出。该适配器是基础设施测试产物，不加入正式模型对照。
5. finally重新启动推理服务，等待模型就绪并自动改回shared。
6. 测试期间旧19题没有全局暂停，其他卡始终有旧题请求。最终两个请求均成功，训练退出码0，两卡归还；`swap-validation.json`的passed=true。

## 常用操作

在一号机动态池的 `code/` 目录，以该机原推理Python运行：

```bash
python -m server.poolctl --config ../pool.json status
python -m server.poolctl --config ../pool.json assign node3-g0 --role validation
python -m server.poolctl --config ../pool.json workers --count 6
python -m server.poolctl --config ../pool.json reserve node4-4090-g0 node4-4090-g1 --owner my-training
python -m server.poolctl --config ../pool.json release node4-4090-g0 node4-4090-g1 --owner my-training --role shared
```

`assign`只改任务路由，不停止正在生成的请求。`reserve`先排空、再释放显存；指标不可读、进程身份不符或有未知GPU进程时拒绝继续，不会强杀其他作业。`release`等待推理服务就绪才重新接任务。owner不同的预约不能互相覆盖。

需要训练结束自动归还时，应在实际训练GPU所在VM使用封装入口，不能直接裸启动训练去争抢显存：

```bash
python -m server.run_reserved_training \
 --config ../training-client.json \
 --gpus node4-4090-g0 node4-4090-g1 \
 --job ../my-training-job.json --owner my-training
```

job文件包含`cwd`和`command`参数数组，shell=False。训练卡须属于当前VM，训练正常或报错退出后均尝试归还；不跨VM拼显存。长任务应后台运行并保存日志，封装进程需持续运行。归还的是推理资源；新增训练适配器另以不可变新名字发布、同步并校验后，使用`poolctl load`注册，不会自动冒充旧模型。

`poolctl load --name ... --path ... --sha256 ...`只允许节点项目资产目录内的适配器，运行时加载后登记模型。旧名字不能替换成不同权重；所有请求按模型匹配路由，未知模型不回退到基础模型。

## 改动、测试与保存

- 修复实机NFS配置mtime缓存：改用内容指纹识别配置变更，连续修改角色立即生效。
- 模型服务重建时同步更新网关认证信息，避免沿用旧服务凭据。
- 19项离线测试通过；六台节点代理、九卡服务、真实基础/适配器请求、排空训练归还和旧断点续跑均完成实机验证。
- 权重仅在服务器间通过SSH同步，不保存在本地。轻量配置的脱敏副本、测试记录、代码与日志保存在 `local-backups/dynamic-pool/`，不提交令牌、权重、虚拟环境和原始运行输出。
- 不再使用旧静态池激活脚本或旧quick6控制器重启这些卡；后续通过动态网关和节点代理管理。
