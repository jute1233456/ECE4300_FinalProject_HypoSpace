# simple-reflection DPO 数据集 v3（2026-11-27，130 基线加法更新）

按用户要求，后续筛选以 `dpo-reflection-v2-ep1` 实际训练用的 **130 对**为基线做加法；
本次把采集恢复后新完成的 35 个状态评审后追加进去。v1/v2 原目录与压缩包不改动。

## 基线

- 来源：`/opt/data/private/ECE4300-train-20261127/dpo-reflection-v2-ep1/data/dpo.jsonl`
- 130 对，sha256 `6e06f4e925255c4df305d72ab89f6ef1aecdcefc5f195dadfe724bb01423c2c9`
- 从 v2 的 133 对中剔除 3 条（原因见其 `manifest.json`）：
  - `0258751f…`：chosen 对 `y OR NOT x` 的深度声称错误（实为 2 层）；
  - `41ccbd79…`：chosen 否认存在严格子集 Layer2，与历史多个有效掩码矛盾；
  - `ca14adbc…`：chosen 把 NOT+OR 分开计层、否定 NOR 复合操作数，与同输入已接受候选矛盾。
- 这 3 条在 v3 中**保持排除**，不会重新加回。

## 本次增量

- 采集恢复后新完成、且不在 v2 已评审 148 状态中的 **35 个状态**；140 份 qwen 候选全部 `finish_reason=stop`、`final_text` 非空、无 thinking，且均有 DS 历史回答。
- 按同一匿名 A–E、逐条 helpful/harmful/uncertain、high/high 才配对的原则评审：
  - **30 对 + 5 跳过**；
  - 判读：helpful 83、harmful 91、uncertain 1；
  - 域分布：布尔 14、3D 10、因果 6；
  - chosen：DS 24、qwen 6；rejected：qwen 30。
- 冻结时采集仍在生成；冻结点为 183 个完整状态。处理期间又出现 1 个新完整状态（当前 184），**未纳入本次**，留待下一增量。

## 修正记录（f2b591e3）

- 原 chosen（包内 packet-04 的 C）建议 `NOT ((NOT x) OR y)`，并声称它不同于历史中的
  `NOT ((NOT x) OR (y AND y))`；但 `y AND y` 按幂等折叠为 `y`，两者是同一结构，而后者已被接受（+1）。
- 该答案改判为 harmful/high，并在 line 150 作为 **rejected**。
- chosen 改为同状态的 A：`NOT (x AND y)`，`(1,0)->1`、深度 2、历史未出现，helpful/high。
- 同状态 B 从 harmful/high 降为 uncertain/medium（误把 #7 判成 NOT y，但其建议的
  `NOT (NOR(x,y))` 本身是合法未出现候选）。
- 因此仍是 **160 对**，不需要 159 对版本；其余 159 对未改动。

## 合并结果

- `dpo.jsonl`：**160 对** = 130 基线（逐字未变）+ 30 新增。
- `pair_manifest.jsonl`：160 行（基线 provenance + 新增来源/哈希）。
- `judgments.jsonl`：183 行（v2 148 + v3 35）。
- `skipped.jsonl`：20 行（v2 15 + v3 5）。
- 全库域分布：布尔 65、3D 58、因果 37。
- 全库 chosen：DS 141、qwen 19；rejected：全部 qwen（160）。
- 3 条 v2 排除项仍在 `manifest.base.excluded` 中记录。

## 产物

- 服务器：`/opt/data/private/ECE4300-dpo-datasets-20261127/simple-reflection-dpo-v3-additive/`
- 增量快照：`/opt/data/private/ECE4300-dpo-datasets-20261127/simple-reflection-v3-increment/`
- 打包：`simple-reflection-dpo-v3-additive-20261127.tar.gz`
  - sha256 `16ad03fe0e0429e316a545c0aefb64a761f3b73260d0316d6fd1ae4954d2a4d8`
- 本地副本：`local-backups/handoffs/simple-reflection-dpo-v3-additive-20261127.tar.gz`

## 已执行检查

- 前 130 行与基线 `dpo.jsonl` **逐字节一致**。
- 160 对全部 chosen=helpful/high、rejected=harmful/high；文本互异、非空、无 `<think>`。
- 160 个 prompt 与对应冻结 `messages` 完全一致；文本 sha256 与 pair_manifest 一致。
- 新增 state_id 与基线、3 条排除项均无重叠。
- 无 train/validation/test 划分字段。

## 边界

- 未启动训练、未改采集器/队列/GPU。
- 采集仍在 `generating`（743/2084、184/521），后续新完成状态按同样流程做下一个增量。
- 这套数据仍以 DS 反思为主要 chosen（141/160），属于蒸馏式偏好集。
