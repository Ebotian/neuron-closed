# 计划 06 — 特异性与剂量：`L14N8949` 到底是不是「敌意方向」

状态：**待手动执行**（脚本已写好，命令见 §5）。上层目的见 `05_plan-gradient-discovery.md`，判据 ID 与数字的唯一来源是 `metadata.json`。

## 1. 要回答的问题

前十二轮建立的结论是：**语言质量读数（J2）可复现**（G1 19 条 0.524、G2 0.471–0.662），**而且效应在未参与筛选的 probe 上复现两次**（8/8，p=0.0078），榜首 `L14 N8949 +1` 在 6/9/19 条 prompt 下都是 rank 1。

但**没有回答**：这个效应是**敌意特异**的，还是**通用抬似然**？

三种竞争假设：

| 代号 | 假设 | 若成立，观测到的应该是 |
|---|---|---|
| **H1 特异** | 它抬"敌意槽位语言"的质量，不抬别的 | 剂量单调；敌意 > 其他 4 个特质的字符串对比；安慰剂 ≈ 0；参考文本似然几乎不动 |
| **H2 通用** | 它只是让任何续写都更"像话" | 所有特质的对比一起升；参考文本似然显著上升；安慰剂也动 |
| **H3 无效应** | 之前 8/8 是 6 条 heldout 的偶然 | 剂量曲线平；对照与候选不可分 |

## 2. 设计（预先冻结）

- **候选**：`L14 N8949`，极性 `+1`（v6/v7 扫描的 signed_rank 1）。
- **剂量**：`0.005 / 0.01 / 0.02 / 0.05`（p50 归一化，与既有剂量标尺一致）。
- **镜像臂（`--mirror`，描述性，不进 S1–S4）**：同一坐标的**反号**剂量。预期不具对称性（一阶灵敏度不是对称杠杆），所以它**不是判据**，而是诊断：镜像若与正向**同等强度**，说明这是"任何扰动都抬强度"的通用效应（H2），而不是方向。
- **对照**：6 个**同层（L14）随机**神经元，各自用 `all_gradients.pt` 的**自身预测符号**；在 `0.02` 与 `0.05` 两个剂量上测（固定种子 1729）。
- **读数**（四个，互不替代）：
  1. **J2**（语言质量，adj+verb 两个槽位族，来自扫描 manifest 的池与模板）——主读数，只看 6 条 heldout。
  2. **J1 跨特质**（字符串对比，5 个特质 × 6 条 heldout）——**只做特异性的相对比较**，不要求它可复现。
  3. **安慰剂**（`probes/specificity.jsonl` 的 neutral_1 vs neutral_2，3 个共享上下文）。
  4. **通用似然**（`engine.REFERENCE_TEXT` 的逐 token 平均对数概率）——抓 H2。
- **提示词**：`qwentest/probes/contrasts_v3.jsonl`（19 discovery + 6 heldout）；只读 heldout 6 条。
- **剂量口径**：`alpha = dose · p50[layer] / ‖W[:,n]‖`，施加到该层 `down_proj` 输入的**所有位置**（与既有实现一致）。

## 3. 判据（先冻结，跑完不许改）

| ID | 判据 | 通过条件 | 失败的含义 |
|---|---|---|---|
| **S1** 剂量单调 | 候选的 J2 随剂量单调上升（允许 1 次并列/微降） | 4 个剂量里 ≥3 段为正 | 剂量曲线不单调 ⇒ 效应不可复现（H3） |
| **S2** 特异性 | 在 0.02 上：敌意的 J1 Δ 是 5 个特质里最大，且 ≥ 第 2 名的 2 倍 | 单次比较，只看排序与倍数 | 不满足 ⇒ H2（通用）或 H3 |
| **S3** 通用性 | `|ΔlogP_ref| / |Δmargin_hostility| < 0.5` | 相对量 | 不满足 ⇒ 干预只是抬全体似然（H2） |
| **S4** 安慰剂 | 安慰剂对（neutral vs neutral）的 Δ 绝对值 < 敌意 Δ 的 1/3 | 相对量 | 不满足 ⇒ 读数被抓的是"任何两段文本的差" |

**镜像臂的读法（不当判据）**：正向强、镜像≈0 ⇒ 与 H1 相容；镜像同等强 ⇒ H2（通用扰动）；镜像反号且幅度相近 ⇒ 一阶线性杠杆（与已知曲率不符，需复核）。

**决策**：S1+S2+S3+S4 全过 ⇒ 写成"存在敌意特异的权重方向"，进入行为端（计划 §2 的下一步）。
**S2 或 S3 失败** ⇒ 写成负结果"权重层在这个模型上没有特质特异方向"，收线，把预算还给 a0/harness。
**S1 失败** ⇒ 8/8 判为小样本偶然，重做 verify（更多 heldout）。

## 4. 成本

约 **1.5 分钟 GPU**（模型加载 ≈ 15 s + 权重 SHA256 ≈ 25 s + **167** 个测量 ≈ 35 s；`--mirror` 占 24 个）。测量数在执行前打印，`[i/N] … eta` 逐项显示，末项后 ETA 归零。

## 5. 执行（手动）

```bash
# 0) 脚本与输入（已在远端就位；本地改动后用 scp 覆盖）
#    /root/autodl-tmp/scripts/dose_specificity.py
#    /root/autodl-tmp/probes/contrasts_v3.jsonl
#    /root/autodl-tmp/qwen-gradient-discovery-v6/   ← 冻结包（engine/semantic/progress/report/design）

ssh -p 11168 root@connect.westd.seetacloud.com

cd /root/autodl-tmp/qwen-gradient-discovery-v6
/root/miniconda3/bin/python -u /root/autodl-tmp/scripts/dose_specificity.py \
  --scan-dir /root/autodl-tmp/runs/v7/j2_19p_full \
  --probes   /root/autodl-tmp/probes/contrasts_v3.jsonl \
  --candidate 14:8949:+1 \
  --doses 0.005,0.01,0.02,0.05 --primary 0.02 --controls 6 --mirror \
  --out /root/autodl-tmp/runs/spec/dose_l14n8949
```

先看计划行（`Plan: … = 167`）再决定是否继续；`--dry-run` 只打印计划、不加载模型（本地也能跑）。
运行后看 `out/dose.md` 的判定表与末行 `VERDICT`。

## 6. 产物与可复现

- `manifest.json`：模型指纹（全量 SHA256）、runtime、扫描 manifest 哈希、probe 文件哈希、剂量、候选与对照、脚本 SHA256。
- `dose.json` / `dose.md`：逐剂量数值、跨特质矩阵、安慰剂、通用似然、S1–S4 判定。
- 数字定稿后：先进 `metadata.json`，再 `make -C qwentest meta && make -C qwentest check`。

## 7. 已知限制

- 5 个特质的 J1 对比是**人手写**的字符串（不可复现性已知），这里只用作**相对**比较；若 S2 的边缘很窄（例如 2.1 倍），不得据此下强结论。
- heldout 6 条此前已被使用两次 ⇒ 本计划的 S1/S4 属**复现级**证据，不是确认级。
- 剂量 0.05 已进入 J2 非线性区（同输入 FD 的幅度比 0.33–1.27），因此 S1 只看**方向与单调性**，不用幅度做预测。
