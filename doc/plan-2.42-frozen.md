# Plan 2.42 — 冻结判据（脚本里只留指针）

| 项 | 冻结值 |
|---|---|
| `phi` | 路径版 J2，单位 nat：正槽位 token 的 log p 均值 − 负槽位 token 的 log p 均值，teacher-forced 在模型**自己的续写**上 |
| score 项 | 逐 rollout、**未加权**、整序列密度（continuation token 的 log p **求和**）；每上下文 **1 次** batched backward |
| pathwise 项 | 2K 个 (rollout, 槽位符号) 对一个 batch，**分块 8**；对槽位 token 取**均值**（只除一次） |
| 上下文平均 | 每个累加都是 1/(K·C) 一次，不重复 |
| baseline | leave-one-out `b_{-i}`，使 `E[(φ−b_{-i})s_i] = g`；同时算 in-sample 作对照 |
| 可靠性 | ICC + F 区间，来自 Gram；`r(K) ~ K/(K+2κ)`，样本需求是 **K = 2κ** |
| 判据（跑前写死） | C0：φ 有动态范围；R：ICC 区间含 0 即"可靠性未分辨"；不因结果回调 |
