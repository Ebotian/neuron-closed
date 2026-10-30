# 重构中暴露并修掉的 bug（task-7）

每条：**怎么发现的** / **修法** / **回归证据**。按发现顺序。这张表是这次重构的主要产出之一——
其中 4 条只有在"逐值回归"或"暴力参照"下才会暴露，自洽性检查全部会放过。

| # | bug | 发现方式 | 修法 | 回归证据 |
|---|---|---|---|---|
| 1 | `detach().clone().requires_grad_` 抓叶子，切断残差载体路径（早层偏 20–30%） | bf16 FD 不可用 ⇒ 两法精确对照（同前向、`\|Δloss\|=0`）逐层 cos 0.074→1.000 ⇒ **fp32 FD 裁判** | `mlp_capture` 改零叶子加法 | `fd_gate`：新法最差 0.006433，旧法 0.1392/0.2419/0.0041/0.2071 |
| 2 | pathwise 对 `\|slot\|` **除两次**（L⁻² 而 φ 是 L⁻¹） | 审计指出 + `\|pw\|`=12.35 与单序列参考 134.11 不符 | 权重里去掉 `/len(ids_slot)` | `evidence`：`\|pw\|` 134.16、`cos(pw,teach)` 0.6548 |
| 3 | 同一 bug 在张量化时**被写回** | 对单序列参考做逐值回归（`\|pw\|` 12.35 vs 134.11） | 同上 | `verify_pathwise.py`：cos 1.000000、范数差 0.00e+00 |
| 4 | Gram **单边加权**（`G_ij=w_i⟨x_i,x_j⟩`，不是范数平方）⇒ "余弦" 1.82、4.71 | 暴力参照（显式构造两半向量） | `Mw @ Mw.T` | `verify_split_half.py`：n=150/150，逐 split 差 1.11e-16 |
| 5 | 同一 bug 在按批重写后**复发** | 同上，暴力参照再次抓到（n 只 38/150） | 同上 | 同上 |
| 6 | `g_pw` 在上下文循环内初始化 ⇒ 只剩最后一个上下文 | 审计指出 | 移到循环外 + 累计快照 | `PW-ADDITIVITY`：`\|final−Σinc\|`=0.00e+00，`\|final−last_inc\|`=3.75 |
| 7 | device 不匹配 ×4（CPU scale × GPU 梯度） | 运行时报错 | `core/design.py` 在边界检查 + `scale_for` 只发同设备张量 | `verify_design.py` A/B/C 全过 |
| 8 | `str.replace` 子串命中（无缩进锚点）→ `IndentationError` | 编译期 | 锚点带缩进 | 提交说明记录 |
| 9 | 闸门本身太窄：只匹配 `p50[l]`，注入 `p50[0]` 通过 | **故意注入违规做失败测试** | 改正则 | 注入 → `[FAIL] noregress`；恢复 → 绿 |
| 10 | Lean 定理陈述错：ICC 下界写成严格 `<`，`MSb=0` 时取等 | Lean 反驳 | 改 `≤` | `icc_bounds` PROVED |
| 11 | Lean 两次卡住是**重写方向**（`smul_smul`、`Finset.sum_sub_distrib`） | 报错"pattern not found" | 反向重写 | `sswu_shift_invariant`、`ssb_is_between_scatter` PROVED |
| 12 | 假"独立样本组"表述（换 rollout 种子 ≠ 独立 context） | 自审 | 文档改为"同上下文内独立 rollout" | metadata / README |
| 13 | `fd_gate` 证书**未绑定仪器哈希** | 审计指出 | `instrument_sha256` + `make check` 校验 | 两次自动拦下旧证书 |
| 14 | 旧 Lean sha 失效（改注释后未重审） | 新闸门抓到 | 重审并更新记录 | `instruments` 闸门绿 |
| 15 | **两份旧记录不可复现**（plan 250 的 MS_w 0.1009、plan 253 的 trΣ 39.8884） | 逐值回归时参考对不上；随后用 HEAD 代码复跑证明代码自洽 | 记录在案，待单独一轮追溯（候选：plan-263 的批量 score 路径、plan-249 的 LOO baseline） | 见 `metadata.plan276/plan277` |
