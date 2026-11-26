# SFT v1第一轮检查点队尾验证

用户授权把原始格式删减版SFT v1的第一轮也加入同一三题验证，并放在队尾。独立名字 `sft-pruned-v1-ep1`，与旧309条数据的 `linear-ep1` 不同，也不替换 `sft-pruned-v1-ep2`。

源：四号机 `/opt/data/private/ECE4300-train-20261126/sft-pruned-v1-ep2/adapter/checkpoint-59/`，944条数据，epoch1.0/global_step59，504张量全部有限且LoRA B非零。权重SHA256为 `8fa112bf26333cd24ea3c2c3759cd5584453a2cce83e75ba8781f569e4bac63e`。这是既有两轮学习率计划中间的检查点，不是独立把学习率在一轮内衰减到零的新训练。

两份必需适配器文件已复制到服务器不可变资产目录 `.../ECE4300-dynamic-pool-20261126/adapters/sft-pruned-v1-ep1/`，不下载本地。四号机双4090与单5090共享该资产路径。

后台入口 `server.publish_queued_adapter`；控制根目录 `/opt/data/private/ECE4300-controller-20261127/` 下 `publish-sft-pruned-v1-ep1/` 保存config、status、publish.log。一次只排空一张卡，等当前API自然结束再加载，逐卡做真实生成测试并登记模型，不重启GPU服务；HTTP错误体写入收据。

主队列config的future_models已添加该模型，收据接口在控制机本地28283/status，使用既有池令牌鉴权。发布完成后，必须先等现有mixed100-baselines-v1中所有pending任务取得运行名额，才将收据置ready。主控制器自动把新模型三题按组追加到model_order尾部，使用同样MCTS2.2/isolated_examples_v1及32K生成请求，独立结果目录保存。

队尾指任务分配顺序，不要求前面已分配的题全部结束；没有空卡则自然等待。原版、线性1/2轮、SFT v1第二轮继续运行，其他被禁用模型不启用。已完成隔离验证：只有加载及API检查通过后才ready，先等待旧pending任务，不调用stop/serve。
