# 11 — 外部证据总账：把「语义 → 神经元闭式逆映射」的每条主张放到同一张网里

**这份文件不同在哪**：`08` / `09` / `10` 是**逐源**的记录（一个源一篇）。这份是**逐主张**的汇总，而且**不是手写的**：

- 数据在 `qwentest/evidence/ledger.json`（8 个来源 × 14 条主张 × 36 条判定）；
- 校验与渲染在 `qwentest/tools/evidence_ledger.py`；
- 下面的矩阵由 `python3 tools/evidence_ledger.py --write` 生成，**手改会被 `--check` 抓**。

**为什么这样搭**：散文式的比较会腐坏。一条"增强"判定可能悄悄丢掉数值依据，一条主张可能被后来的源推翻却没人回去改，
一个来源可能在被引用时没人写出它的前提差异。所以这里把三件事变成机械的：

1. **每条 S/W/X 判定必须含至少一个数字**（否则 `--check` 失败）。"增强"没有数字，正是我们要防的失效模式。
2. **每条主张必须有判定；每条判定必须指向已知的主张与来源**（防孤立主张与幽灵来源）。
3. **来源前提表由 ledger 的 `sources` 字段渲染**，所以它不可能与正文漂移。

验证手段：我故意把一条 S 判定的数字删掉，`--check` 立刻 exit=1 并指名那条；
把数字放回去，exit=0。**校验器本身也做了负测试。**

---

## 1. 判读刻度（与 `08_reference` §2 一致）

| 符号 | 含义 | 门槛 |
|---|---|---|
| **S** | 增强 | 独立数据、同向，且**前提可比** |
| **W** | 弱化 | 独立数据、反向，或使该主张不足以成立 |
| **X** | 交集但**前提不同**（"看着同类，推断不能迁移"） | 必须写明差在哪个前提 |
| **N** | 无信息 | 必须写明为什么推不出任何东西 |
| **M** | 方法学迁移（不针对主张，但改变我们必须怎么测） | 记在正文，不入矩阵 |

---

## 2. 共同抽象：把"前提"拆成 7 条轴，并量出每条的**实际影响量级**

这一节是这份文件的核心。所有"X / 前提错误"其实都能落到下面某一条轴上，而**每条轴我都有数字**：

| # | 前提轴 | 它改变什么 | 实测影响量级 | 出处 |
|---|---|---|---|---|
| **P1** | **剂量归一化的实现** | `dose·p50/‖W[:,n]‖` 里的 `p50` | 三方**逐字同构**，但 `p50` 因量化差：**4-bit 低 6.0%，8-bit 高 3.2%**（L13）；4-bit vs 8-bit 差 **12%** | `10` §2 |
| **P2** | **剂量区间** | 你落在可用带的哪一段 | 行为可用带上界 **α≈0.8–1.0**；**α≥1.2 崩塌**（1.2 起 18/20 broken；2.0 时 hostile 塌到 1/20）；社区配方剂量中位 **0.4**、众数 **0.6/0.8** | `09` §3、`10` §4/§5.1 |
| **P3** | **结局量** | 你在测"像不像"还是"是不是" | 同一干预：J2 读数**单调升**（b=0.05→0.5：+0.50→+1.27），行为 hostile **非单调**（峰 14/20@0.75，2.0 时 1/20） | `09` §3、我们的 §2.22/§2.26 |
| **P4** | **支撑集形态** | 1 个坐标 vs 全支撑 | 同物理预算下 **K=1 → K=18432 是 5.78×**；外部行为端：单神经元 **0/16**、三元组 **0/6**、稠密 **4/6** | 我们的 §2.20、`10` §3 |
| **P5** | **线性化** | 多坐标能否相加 | **6+1=7 而实测 16（2.3× 超加性）** ⇒ 线性 `Q(S)` 表达不了交互 | `10` §4 |
| **P6** | **层位置** | 在哪里动手 | 外部两类证据都把峰放在**比我们浅**的位置：Qwen3-8B 严格化后 **0.25–0.33** 相对深度；社区层峰 **L7/L4/L22**（L13/L14 只 6/2 行）；稠密方向最佳层随人变（**L14/L12/L8**） | `08` §6、`10` §5.1、`10` §3 |
| **P7** | **判官/标签** | 标签本身有多稳 | 二元轴 κ=**0.8913 / 0.9544**（129 条盲复评，我重算）；五分类 tone 仅 **90.70%**；同一判官的"退化率"可差 **11 倍**（4.297% / 7.74% / 48.156%） | `09` §4、`08` §4.4 |

**这七条轴就是"共同抽象"**：任何外部结论要与我们的比较，必须先在 P1–P7 上对齐；
不对齐时，正确标签是 **X**，而不是 S 或 W。矩阵里 10 条 X 全部可以归到这七条上。

---

## 3. 主张清单（14 条）

| id | 主张 | 当前状态（我方） |
|---|---|---|
| C1 | 跨神经元剂量归一化 `dose·p50/‖W[:,n]‖` 正确且可复用 | 通过（三个独立实现同构） |
| C2 | 坐标级效应可复现（换词半/prompt 半仍选到同一批） | 部分（G1 池化过 r=0.5469/J64=0.4375；逐上下文 r 中位 0.4722 不过） |
| C3 | 坐标是真实、有方向、剂量单调的杠杆 | 通过（ΔJ2 +0.0059…+0.0484，镜像臂反向） |
| C4 | 效应有敌意**特异性** | **未证明**（R2 暂停、R2c 门槛未过） |
| C5 | 少数稀疏坐标能抓住该效应 | **已否**（top-1 只占 17.3%、top-64 拿 5.74% 的 ‖g‖²） |
| C6 | 线性代理能预测效应幅度 | 部分（top-K 方向 realized/predicted 0.838–1.010；随机支撑上 ~2×） |
| C7 | 闭式 `u∝M⁺g` 在稠密区优于 `u∝g` | 通过（36–43%，96/96 格，两轴复现） |
| C8 | 解析组合 `Q(S)=g_SᵀG_SS⁻¹g_S` 能表达多坐标交互 | **不足**（超加性 2.3×） |
| C9 | 读数改善等价于行为改善 | **已否**（b=0.5 读数最高而能力代理塌 0.64 nat） |
| C10 | L13–L14 是对的作用层 | **未测**（继承自早期扫描，非测量选择） |
| C11 | 读数数值误差可控 | 通过（单次 ±2.0e-03/6.5e-03；均值 ~9e-04；fp32 对照） |
| C12 | LLM 判官可给出可靠读数/标签 | 部分（二元 κ≈0.89/0.95；五分类 0.907；口径可差 11 倍） |
| C13 | 输出形式应为「神经元+符号+强度」的短列表 | **需修正**（63.1% 的社区配方是单神经元，而单神经元 0/16 有效） |
| C14 | 语义编译 `τ→J_τ` 是主要难点 | 通过（同一坐标在两套读数下结论相反） |

---

## 4. 证据矩阵

<!-- auto:evidence-matrix:start -->

### 主张 × 来源 矩阵（程序生成，勿手改）

格内符号：**S** 增强 · **W** 弱化 · **X** 前提不同 · **N** 无信息 · `·` 未评。

| 主张 | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 | S13 | S14 | 净结论 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `C1` 跨神经元的剂量归一化 dose*p50/||W_down[:,n]|| 是正确且可复用的约定 | S | S | S | · | S | · | · | · | X | · | · | · | · | · | S×4 为主，X×1 |
| `C2` 坐标级效应可复现（换词半/换 prompt 半仍选到同一批坐标） | N | N | W | X | · | · | · | · | X | W | · | · | · | · | W×2 为主，X×2 |
| `C3` 找到的坐标是真实、有方向、剂量单调的杠杆 | · | S | S | · | · | · | · | · | · | · | · | · | · | · | S×2 为主 |
| `C4` 效应具有敌意特异性（vs 其他同语域特质） | N | N | N | · | · | · | · | · | · | · | · | X | N | · | 无增强，X×1 |
| `C5` 少数稀疏坐标（top-K）能抓住该效应 | W | · | W | W | W | · | · | · | W | · | · | X | · | · | W×5 为主，X×1 |
| `C6` 线性代理能预测效应幅度（calibration） | · | · | X | X | · | · | · | · | X | · | · | · | · | · | 无增强，X×3 |
| `C7` 闭式最优控制 u ~ M^+g 在稠密区优于 u ~ g | · | · | · | X | · | · | · | · | · | · | · | · | · | · | 无增强，X×1 |
| `C8` 解析组合 Q(S)=g_S' G_SS^-1 g_S 能表达多坐标的交互 | · | · | W | · | X | · | · | · | W | · | · | · | · | · | W×2 为主，X×1 |
| `C9` 读数改善等价于行为改善 | · | W | W | · | · | · | X | · | · | · | · | W | W | · | W×4 为主，X×1 |
| `C10` 层 13-14 是对的作用层 | X | · | · | X | X | · | · | · | X | · | · | · | · | · | 无增强，X×4 |
| `C11` 读数数值误差可控（可跨 run 比较 1e-03 级差异） | · | · | · | · | · | W | · | S | · | · | · | · | · | · | S/W 相抵 |
| `C12` 读数/标签可由 LLM 判官可靠给出 | W | S | · | · | · | · | · | · | · | · | · | · | · | · | S/W 相抵 |
| `C13` 输出形式应为『神经元+符号+强度』的短列表 | · | · | · | W | W | · | · | · | S | · | X | · | · | · | W×2 为主，X×1 |
| `C14` 语义编译 tau -> J_tau 是整个映射的主要难点 | X | · | · | · | · | · | S | · | S | · | · | S | S | · | S×4 为主，X×1 |
| `C15` Additive injection and multiplicative scaling are in | · | · | · | · | · | · | · | · | W | · | · | · | · | · | W×1 为主 |
| `C16` A toxicity classifier trained on English can score a | · | · | · | · | · | · | · | · | · | · | · | · | X | · | 无增强，X×1 |
| `C17` A hostility-compiled (antagonism) direction is behav | · | · | · | · | · | · | · | · | · | · | · | · | · | S | S×1 为主 |

### 每条判定的数值依据（程序校验：S/W/X 必须含数字）

| 主张 | 来源 | 判 | 依据 |
|---|---|---|---|
| C1 | S1 | **S** | alpha 与我们的 dose 逐字同构；r_L 定义=去掉首 token 的残差范数中位数，与 estimate_p50 一致 |
| C1 | S2 | **S** | 公式逐字同构；L13 10.066 / L14 11.014 vs 我们 bf16 9.752792 / 10.329014（+3.2%/+6.6%） |
| C1 | S3 | **S** | 公式同构；L13 9.163 / L14 9.694（4-bit，比我们 bf16 低 6.0%/6.2%） |
| C1 | S5 | **S** | 141 条配方的剂量轴与该约定相容：中位 0.4、众数 0.6/0.8 — 但作者未记 p50，无法逐条核 |
| C1 | S9 | **X** | their action space is alpha_j over ALL d = sum_l d_l neurons across every layer, multiplicative; ours is additive dose on 2 layers with budget sqrt(u'Mu); the shared part is only 'closed form + curvature matrix' — the equivalence is exact in parameterisation: delta = sqrt(2*eps*g'F^-1*g) |
| C10 | S1 | **X** | 其严格化后 persona 峰在相对深度 0.25-0.33（L9-L12/36），我们 L13-14/32 是 0.41-0.44 — 前提不同，只能记张力 |
| C10 | S4 | **X** | 最佳稠密层随人不同（L14/L12/L8），说明层不是常数 |
| C10 | S5 | **X** | 社区层峰在 L7/L4/L22，L13/L14 只 6/2 行；但这是模仿偏差 |
| C10 | S9 | **X** | they scale all layers at once with clipping and damping, so their design cannot answer which layer to act on; the clipping cap a=0.75 removes at most 0.352% of neurons — unchanged verdict, sharper reason |
| C11 | S6 | **W** | 外部项目因 readout 把 final norm 应用两次而作废 10 个 session 的 KL 数字 — 同类风险 |
| C11 | S8 | **S** | 内部已量：单次 dJ2 ±2.0e-03(中位)/6.5e-03(最坏)，8 上下文均值 ~9e-04 |
| C12 | S1 | **W** | 同一判官对『退化』给出 4.297%/7.74%/48.156% 三个口径（差 11 倍） |
| C12 | S2 | **S** | 129 条盲复评：hostile kappa 0.8913、broken 0.9544；五分类 tone 仅 0.9070 |
| C13 | S11 | **X** | across 13 open-source transcoders it finds neuron coordinates have a stable correspondence with the MLP while most transcoder features do not, which supports residual direction + sparsest representative + equivalence class over a unique feature list — premise differs: feature basis vs coordinate basis |
| C13 | S4 | **W** | 稀疏 0/16、三元组 0/6、稠密 4/6（p(target)=0.44–0.77） |
| C13 | S5 | **W** | 社区 63.1% 单神经元、≥3 个仅 7.8%，而单神经元有效率为 0/16 |
| C13 | S9 | **S** | they clip alpha to [0.25, 1.75] (a=0.75) and rely on 'F is diagonal and positive definite, clipping cannot increase the quadratic preservation cost'; uncapped /delta alpha/ reaches 20.57 and 42.25 while at most 0.352% of neurons get clipped — so our output form needs a box constraint / dose ceiling field - we do not have one |
| C14 | S1 | **X** | 其对 persona 的定义是 LLM 给的标签，不是可微目标，无法与我们 J2 的编译路径比较 |
| C14 | S12 | **S** | it validates directions against a theory-specified structure (Schwartz 20 values, MFT 6 foundations) rather than isolated behaviours, which is exactly the compiler-level test we lack — supports C14 and gives the design |
| C14 | S13 | **S** | at b=0.7, with a gauge whose stage-0 AUC on English probe material is 0.986-1.000, the same semantic definition compiled three ways differs by ~10x in mean P(toxic) (+0.9869 caa_bffm / +0.6280 caa_jibes / +0.1022 j2) at comparable capability cost (-1.44 / -2.30 / -1.25 nat), and their direction cos is -0.0024 to 0.4902 — compiler choice, not the solver, dominates the behavioural outcome |
| C14 | S7 | **S** | 同一坐标在两个不同读数的结论相反（J1 vs J2），证明 tau->J 的定义支配结论 |
| C14 | S9 | **S** | it separates 2 explicit proxies (target-token log-prob for behaviour, output-distribution KL for capability), so the compiler's definition directly determines the optimised object — the compiler's definition directly determines the optimised object |
| C15 | S9 | **W** | Prop. 1, verbatim: g_j = (1//D_steer/) sum sum_p h_pj dP_sigma/dh_pj, i.e. activation times gradient; multiplicative gain is proportional to the activation, so at h=0 it is 0 — the additive injection we use can open a silent neuron; confirmed in the TeX, not only the README |
| C16 | S13 | **X** | stage 0 AUC(pos>neg) = 0.986 (tox_bert) and 1.000 (s-nlp) on the 21 English probe continuations, but 0.688 / 0.438 on the 4 Chinese ones and 0.643 / 0.587 on the 14 verb-frame slot words — usable for English adjective-frame material only; Chinese and the verb frame are blind |
| C17 | S14 | **S** | at b=0.7 on 12 English hostility prompts, II- (antagonism vs warmth) gives mean P(toxic) 0.9875 with 12/12 above 0.5, while I- / III- / IV- / V- give 0.0006 / 0.1655 / 0.0069 / 0.0340; per-context margin over the best other pool is +0.249 to +0.984, and the II- vs IV- paired exact sign test is p=0.00049 (n=12) — damage-matched control: all five directions cost -1.44 to -2.24 nat of reference logP and have comparable rep4 (II- is among the LOWEST at 0.013), so the toxicity is not a byproduct of general destruction or of looping. Scope: this is compiler-level specificity, NOT evidence that L14N8949 is hostility-specific (claim C4 unchanged). |
| C2 | S1 | **N** | 单个 prompt，对跨上下文可复现零信息 |
| C2 | S10 | **W** | their CAM ranks by a posterior with an uncertainty penalty; I transplanted that idea as a context-bootstrap 5% lower bound on our gradient and TESTED it (plan 2.27): the registered prediction K@73%(lcb) < 256 was FALSIFIED - both rankings need K=256, and at K=1 the LCB ranking captures only 69.6% of the mean ranking (difference -0.0154 = 17x the 9e-04 mean error); K>=128 the two are indistinguishable — this tests the TRANSPLANT onto our gradient, not RACE's own forward-pass / token-level posterior estimator; measured in 07_results/lcb_rank_c8.json |
| C2 | S2 | **N** | 同一批 20 提示，无分半设计 |
| C2 | S3 | **W** | 其 L13-only 与 L14-only 的差异说明单坐标行为不稳，但无分半检验 |
| C2 | S4 | **X** | 稠密方向 4-bit->8-bit 不重调即迁移，但那是方向级不是坐标级 |
| C2 | S9 | **X** | they treat F as a SHARED statistic (computed once, reused across budgets and across domains), whereas we recompute our M-side quantities for each of the 8 contexts; premise differs, but one shared matrix serving all contexts is a reuse we do not exploit — cheap reuse: 1 F vs 8 per-context recomputations |
| C3 | S2 | **S** | L14N8949 单坐标 12/20 hostile、剂量峰 14/20@0.75、broken 单调 0->20/20 |
| C3 | S3 | **S** | 同一坐标 L14-only 6 个 violent terms vs 无干预 0；剂量 0.4-0.6 安静 / 1.0 恶意 |
| C4 | S1 | **N** | 只有『是否 persona』，不是敌意 vs 其他特质 |
| C4 | S12 | **X** | they test specificity with Schwartz 20-value and MFT 6-foundation compatible/opposing structure, but not a same-register semantic trait on our J2 readout — its design (many traits + predicted relations) is what our R2c lacks |
| C4 | S13 | **N** | the 8 heldout contexts contain only 2 hostility-eliciting prompts (hostility_04/05), and the control's single gauge hit was on hostility_05; so this run has almost no power for specificity — design flaw of mine, recorded; next run must use the 19 discovery hostility prompts |
| C4 | S2 | **N** | 只有敌意一个轴，无同语域对照 |
| C4 | S3 | **N** | 无对照特质 |
| C5 | S1 | **W** | 884,736 对里 persona 仅 3.87%，且干净 persona 只有 0.2405%（1/16） |
| C5 | S12 | **X** | all 7 methods it evaluates (CAA, SAS, SphericalSteer, ODESteer, OPT, COLD-Steer, BiPO) are vector-level, none reports a top-K sparse set — same direction as the C5 weakening, but not a direct test |
| C5 | S3 | **W** | 激活 z 最高的坐标（Attenborough L27N7938 z=242.0）单独推仍失败 |
| C5 | S4 | **W** | 单神经元 0/16、三元组 0/6，稠密 4/6；p(target) 稀疏全 <0.01、稠密 0.44-0.77 |
| C5 | S5 | **W** | 社区 63.1% 的配方是单神经元，但他人复现率仅 7.8% — 这是实践分布，不是效应检验 |
| C5 | S9 | **W** | their object is a continuous scaling coefficient over ALL neurons, not a top-K set; put beside our measured capture rates (top-1 = 17.3% of the full-support effect, top-64 = 5.74% of //g//^2), it shows dense scaling is the publishable mainstream object — dense scaling is publishable mainstream practice |
| C6 | S3 | **X** | 其剂量阶梯给出行为区间（0.8 过渡/1.0 恶意），但没有代理预测值可比 |
| C6 | S4 | **X** | 其 logit-lens 预测的目标名概率 0.4398/0.7711/0.5785 与生成结果方向一致，但未报 calibration 误差 |
| C6 | S9 | **X** | figure vector text, both models: dP parity R(y=x)=0.928 and 0.996 with MAPE 12.9% and 22.0%; KL parity R(y=x)=0.877 and 0.922 with MAPE 10.1% and 8.7%; the text only claims 'almost all within 20% relative-error' — CORRECTED: not R^2, and each quantity has 2 panels - the panel with R=0.996 has the WORST MAPE (22.0%), so high correlation coexists with poor calibration, exactly as in our own 0.838-1.010 vs ~2x split |
| C7 | S4 | **X** | 稠密 4/6 vs 稀疏 0/22 与此同向，但其方向是均值差、无 M 预算，非 M^+g |
| C8 | S3 | **W** | 6(L14)+1(L13)=7 但双坐标实测 16（2.3x 超加性）=> 线性 Q(S) 结构上不足 |
| C8 | S5 | **X** | 社区多坐标配方普遍给 overlap 效应描述（mashup 27 条），但无剂量-响应数据 |
| C8 | S9 | **W** | their Limitations, verbatim: 'retains only the Fisher diagonal ..., omitting neuron interactions and their joint roles'; put beside our measured 6+1 -> 16 (2.3x superadditive), both frameworks are on the same boundary — now citable from the paper body, not just the README |
| C9 | S12 | **W** | its central finding is that steering success on isolated behaviours can be a behaviour-specific shortcut; consistent with our b=0.5 result (+1.27 readout while the capability proxy collapses by 0.64 nat) — independent support for our readout != behaviour |
| C9 | S13 | **W** | two budgets, 12 English hostility prompts: at b=0.05 every compiler leaves P(toxic) at 0.0000-0.0003 (baseline 0.00057) while j2's own readout rises +0.50; at b=0.7 all three DO move it (+0.1022 / +0.6280 / +0.9869) but cost -1.25 to -2.30 nat. So readout != behaviour holds inside the certified band (b<=0.1) and fails outside it — the earlier C9 verdict is now bounded rather than absolute |
| C9 | S2 | **W** | 行为 hostile 峰在 0.75，2.0 时塌到 1/20，而 broken 到 20/20 => 读数式指标与行为不同步 |
| C9 | S3 | **W** | 1.0 报『完全恶意』、1.2 已『不连贯』，可用窗很窄 |
| C9 | S7 | **X** | 其读数是 logit 期望，且与 J2 不可混用 |

### 来源前提表（程序生成）

| id | 来源 | 模型 | 量化 | 剂量定义 | 结局量 | 单位 | n | 判官 κ |
|---|---|---|---|---|---|---|---:|---|
| S1 | Qwen3-8B 单神经元全量图谱（第三方站点） | Qwen3-8B | unknown (8B, api/vercel 站点) | +/-alpha*r_L/||W_down[:,n]||, alpha in {1.25,1 | 自由生成 160 token -> LLM 判类（persona/broken  | 神经元 x 方向（884,736 对） | 1 | — |
| S2 | fermicat Evil-Qwen MLX 复现 | Qwen3.5-4B | MLX 8-bit | dose*median(||resid leaving L||)/||W_down[:,n] | 自由生成 120 token -> LLM 判 hostile/broken | 条件 x 20 提示 | 20 | hostile=0.8913/broken=0.9544/tone5=0.907 |
| S3 | qwen-neuron-audit（4-bit 审计） | Qwen3.5-4B | MLX 4-bit | dose*median_residual[L]/||W_down[:,n]||（同构） | 12 提示的 regex 计数的 violent terms + 拒答 + re | 条件 x 12 提示 | 12 | — |
| S4 | qwen-celebrity-neurons（稀疏 vs 稠密） | Qwen3.5-4B | MLX 4-bit（+8-bit 复核） | 稀疏: 神经元 x alpha; 稠密: resid += f*||resid_L||*v/ | 目标名字的下一 token 概率 + 生成身份 | 规格 x 人 | 3 | — |
| S5 | neuronary 社区配方库 | Qwen3.5-4B 为主 | 未记 | dose（作者自报，同一约定假定） | 示例输出（定性）+ 他人复现计数 | 配方（141 条 / 235 行） | — | — |
| S6 | WDD（Write-Dictionary Decomposition） | 多模型（GPT-2/Qwen2.5/Qwen3-4B） | 多种 | 写入字典分解，非本线剂量 | KL / 激活分布统计 | session | — | — |
| S7 | SNAM v1 实验包（外部交付） | Qwen3.5-4B | bf16 | alpha = u*p50/||W_down[:,n]||, budget=0.5 | logit 期望 J = E_q+[logit] - E_q-[logit]（单 | prompt x 坐标 | 6 | — |
| S8 | 本线自己的 GPU 运行（内部基线） | Qwen3.5-4B | bf16 | dose*p50/||W_down[:,n]||，物理预算 sqrt(u'Mu) | J2 log-mass（线性于 logits）+ 无判官行为指标 | 上下文（8-30 条） | 8 | — |
| S9 | CANS (preprint 2610.35530 v1) + official code/artifacts - TeX SOURCE READ | Qwen3-4B (their release) | bf16 | MULTIPLICATIVE scaling alpha_j folded into dow | target-token log-prob (behaviour) + outp | all neurons (not sparse) | — | — |
| S10 | RACE (2608.24758 v2) + code | several including small LLMs | - | no dose: forward-pass statistics (RDA + Normal | domain-wide functional-consistency score | neuron x token (streaming) | domain-wide | — |
| S11 | Sparsified MLPs vs Transcoders (2610.24521 v1) | 13 open-source transcoders | - | no dose: feature-neuron weight alignment analy | alignment of transcoder features with ML | feature x transcoder | — | — |
| S12 | Steering Geometry (2609.06289 v1, EMNLP 2026) + code/dataset | several + Schwartz 20 values / MFT 6 foundations | - | contrastive vectors (CAA/SAS/Spherical/ODE/OPT | geometric alignment + behavioural effect | value x method | — | — |
| S13 | two borrowed toxicity classifiers, staged: validity first, then steered generation | unitary/toxic-bert (apache-2.0) + s-nlp/roberta_toxicity_classifier (openrail++) | fp32 on GPU | not a steering source: scores generated text | P(toxic) per generated reply | reply | 25 probe rows (stage 0) + 12 English hostility prompts (stage 1) | — |
| S14 | BFFM five-pool specificity run at b=0.7 (our own experiment, recorded as evidence) | Qwen3.5-4B bf16 | bf16 | CAA-style contrast within a pool (X- vs X+), o | P(toxic) from two borrowed classifiers o | pool x context | 12 | — |

<!-- auto:evidence-matrix:end -->

---

## 5. 从矩阵读出来的四件事

### 5.1 真正被**增强**的只有三条，而且都是"约定/机制"级，不是"发现"级

`C1`（剂量约定，S1/S2/S3/S5 四个源）、`C7`（稠密区闭式控制，经 S4 的 4/6 vs 0/22 间接支持）、`C11`（误差可控，
经 S6 的反面教训加我们自己 fp32 对照）。**增益集中在"我们用的尺子对"和"稠密方向对"，不在"找到了某个神经元"。**

### 5.2 被**弱化**的集中在"稀疏 + 可加 + 读数即行为"这一组

`C5`（稀疏）、`C8`（可加）、`C9`（读数=行为）、`C13`（短列表输出）——这四条互相咬合：
如果效应稠密、不可加、读数≠行为，那么「输出十个触发键」这个形式就同时失去三个支撑。
外部数据在这四条上**一致地**指向同一方向（S1 的 3.87%→0.2405%、S3 的 z=242 失败、S4 的 0/22 vs 4/6、S5 的 63.1% 单神经元）。

### 5.3 **前提不同（X）不是含糊其辞，而是可量化的**

矩阵里 10 条 X 全部落在 P1–P7 七条轴上，且每条轴都有量：
量化让 `p50` 差 6–12%（P1）；行为可用带在 α≈0.8–1.0 而社区剂量中位 0.4（P2）；
读数单调而行为非单调（P3）；层峰比我们浅 0.1–0.2 相对深度（P6）；五分类标签只有 90.7%（P7）。
⇒ **"前提错误"这类判断可以写成"差多少"，而不是"差不多/不一样"。**

### 5.4 五条 `N` 划出了**外部数据帮不上**的地方

`C2`（跨上下文可复现——外部都只有 1 个 prompt 或 1 批）、`C4`（特异性——外部只有敌意一个轴）。
这两条正是我们卡住的 R2 / R2c，而外部数据**在结构上**给不出答案：它们没有同语域的第二特质，
也没有分半可复现设计。⇒ **继续挖外部数据不会解开 R2；只有我们自己能测。**

---

## 6. 由这张网导出的三条**待改**（下一步的判据，不是结论）

1. **C13 的输出形式要改**：从"十个触发键"改为 **排序列表 + 每个前缀的累计捕获率 + 每个条目的词池/上下文出处**。
   依据：`K=1` 只拿全支撑的 17.3%，而 `K=256/1024/4096` 拿 73%/89%/97%（§2.20）；社区 63.1% 是单神经元而单神经元 0/16 有效。
2. **C8 的边界要写死**：任何"解析组合"的表述必须附一句"**仅限线性区**"，
   并给出超加性反例（`6+1→16`）；同时把 `pair_I` 那种两坐标检验升级为**三元组**（社区配方里 3 个以上的有 11 条可作先验）。
3. **C10 必须变成测量**：`layer_sweep.py` 是草稿（未跑）。外部三类证据都指向"我们的层偏深"，
   而这一条只能用我们自己的模型来判——这是当前**最便宜、最能改变结论**的一步。

---

## 7. 维护方式

```bash
python3 tools/evidence_ledger.py --check     # 校验：S/W/X 必须有数字；主张/来源必须存在
python3 tools/evidence_ledger.py --render    # 只打印矩阵
python3 tools/evidence_ledger.py --write     # 重写本文件 §4 的自动块
```

新增一个外部来源时：先在 `sources` 里加一条（含模型/量化/剂量定义/结局量/单位/n/判官 κ），
再为它逐条主张写判定；**写不出数字的判定只能记 N 或 M**。这条规则是这份文件唯一的手动纪律。
