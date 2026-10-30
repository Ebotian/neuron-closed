# Handover — read this first in a new session, then `README.md` + `doc/metadata.json`

State is in the repo, not in a chat log. Everything below is committed.

## Where the line stands (2026-10-29, metadata v0.47.0)

**2026-10-29: 本工作线已从 `rsi-dev` 单仓迁出为独立仓 `neuron-closed`**（历史 69 条保留）。
路径以本仓为准：`doc/`、`results/`、`scripts/`、`vendor/`。
- **Closed form**: `u = ŴB⁻¹d*`, `d* ∝ B⁻¹e`, budget `sqrt(uᵀMu)` (M = Gram of unit W_down columns), delivered to 5.6e-17.
- **u_M beats u_g** by 36–43% at b=0.05–0.5, 8/8 contexts, p=0.0078; transfers out-of-sample 8/8; word-half 12/12 cells.
- **Dense beats sparse**: K=1 → 18432 is 5.78×; K=1 17.3%, K=256 73%, K=1024 89%.
- **Layer is a plateau**: 32-layer sweep, spread only 1.296× (L2 0.2951 → L14 0.3823). L14 rank 1, L31 rank 2.
- **Compiler is the family**: cos(CAA-Jibes, J2) = −0.0024. At b=0.7 CAA-BFFM gives P(tox) 0.9875 vs J2 0.1022.
- **Specificity (compiler level)**: BFFM II− beats I−/III−/IV−/V− 4/4, p=0.00049, damage-matched.
- **No usable operating point**: P(tox) crosses 0.5 between b=0.2 and 0.3, where cost is already −0.587 nat (frozen tolerance 0.3).
- **Target-directedness**: 92.4% coder coverage; II− peaks at b=0.2–0.3 (2–3/12), by b=0.7 it is 10/12 self-directed.
- **Those coordinates are not the direction**: cos(u_II−, u_known) = 0.0299; `u_known` gives 1/12 of the effect at 1/5 efficiency.
- **The third `G` adds nothing *on this support*** (plan 2.34, `dose_f`): the output-KL Fisher is a reproducible
  instrument (split-half cos 0.9936) but nearly isotropic after `λ = median(F)` (`cond(diag F+λI)` 1.995, diag spread
  3.547×), exactly as `M` already is — so `cos(u_M, u_F)` **0.9881**, `cos(u_I, u_F)` 0.9914, and the frozen primary
  criterion **FAILS** (3/8, p=0.7266, −0.92%, below the 1e-02 claim threshold). At full support
  `cos(u_g, u_M)` is only 0.588 — that is where "which G" can still be decided.
- **R8 is now run (plan §2.35)** and it settles it: on the full 18,432-coordinate support the Fisher really is
  anisotropic (`cond(diag F+λI)` **8.052** vs 1.995 on 32 coordinates — the registered prediction held), yet the
  direction it implies still sits on the gradient (`cos(u_g,u_F)` **0.9872**) while `u_M` does not (**0.5878**), and
  `u_F` loses to `u_M` in **0/8** contexts (p=0.0078, **−27.5%**). `u_M` beats `u_g` **8/8, +40.7%**, reproducing
  plan 2.22 — so the run is comparable. Corollary worth keeping: `M` is not one price tag among others, it *is*
  the budget being scored; optimising any other quadratic form costs ~a quarter of the effect.
- **Refactor rule (learned here)**: a change to a validated measurement path must be regression-tested PER VALUE.
  Adding `--dense` first used coordinate indices as neuron ids; every budget moved 3–13% while all cosines stayed
  at 0.995 and the sign tests stayed plausible. After the fix the `u_I` arm reproduces bit-for-bit.

## 本仓的边界与外部文件

- **入 git**：`doc/`、`results/`、`scripts/`、`tools/`、`vendor/`（文本）、`lexicons/`、`probes/`、`evidence/`。
- **不入 git**：`packages/*.zip`、`archives/*.zip`（原始压缩包，未跟踪；本地备份在 `~/neuron-closed-local/`）、
  模型权重、社区数据快照（neuronary / neuron-explorer 缓存，只在 `doc/references/` 记统计量与 sha）。
- 因此 `make check` 的 `ziptrees` 项在干净 clone 上遍历到 0 个 zip（通过）；原始压缩包只在本机。

## Remote
`ssh -p 25785 root@connect.westc.seetacloud.com` (4080 SUPER, 32 GB). Caveats:
- **cannot reach huggingface.co**; use `https://hf-mirror.com` (curl, ranged requests work).
- `huggingface_hub` on that box is broken (`httpx2` TypeError) — use `scripts/fetch_models.py`.
- gauge models live in `/root/autodl-tmp/gauges/` (sha256 in `fetch_manifest.jsonl`).
- `python3` is `/root/miniconda3/bin/python`; package mirror at `/root/autodl-tmp/scripts/j2space/`.

## Next actions, in the order the evidence supports
1. **Score the audit sheet**: `results/target_directedness_audit_sheet.csv` (48 blind texts,
   private mapping next to it). Without human labels the target-directedness coder is a proxy.
2. Optional: repeat the b-sweep with `caa_bffm` instead of `II−` (it was the strongest at b=0.7) and
   check whether its target-directedness peaks at the same budget.

## How to work in this repo without blowing the context
- Never `cat` a file. Read a slice: `sed -n 'START,ENDp'`, or `grep -n` first.
- Send big command output to a file and read the tail: `... > /tmp/x.log 2>&1; tail -6 /tmp/x.log`.
- Use `codemode` to compute in the sandbox and return only a few numbers.
- Record numbers in `doc/metadata.json` first, then `make -C qwentest meta && make check`, then commit.

## compaction headroom（单仓时代的记录，仅供参照）
- failure: messages 688,978 + output 384,000 = 1,072,978 > 1,048,576
- rule: compaction must trigger at T <= W - O + keepRecent, else the summary request cannot fit
- so with W=1,048,576, O=384,000, keepRecent=8,000 -> reserveTokens must be >= 372,000

## 迁移后新增的三条结论（2026-10-29，都在 §2.38/§2.39）

- **位置能被算**：`u_S = M_SS⁻¹g_S`，`Q(S)=g_SᵀM_SS⁻¹g_S` 贪心；K=2048 全 32 层；`results/select_rules.json`。
  交付物是 **(位置, 改变) 成对**：`results/positions_all32_K1024_pairs.csv`（含 `u`、`alpha`、`‖Δresid‖`、`p50`、`‖W_col‖`、Q 份额）。
- **方向可复现、位置不可复现**（这是本仓最硬的一条边界）：
  同一行为目标下符号一致 **93.4%（178 条配方）**；但精确重叠恒 **1.7%**，与 prompt 面数（1→4）和展开阶数（一阶→二阶）无关；
  邻域命中 50–60%（名次中位 11–13）。⇒ 别再把"复现社区坐标"当目标函数。
- **二阶不是答案**：曲率只让命中时的名次变好（28→7–14），却让有命中的配方变少（58%→25%）。

## 下一步（按证据支持顺序）

1. **集合级目标**：若要"能命名的位置"，必须优化他们那种集合级/阈值型目标（行为验证），而不是在局部展开上加阶数。
2. **混合规则**：先用单点效应初筛（他们发布的 ±0.8 KL 或我们的单坐标 `|g|`），再用 `Q` 去冗余；需要一次测量才能定优劣。
3. **`results/` 的 tag 化**：每篇对外结论对应一个 tag + `results/*` 的 sha256 清单（见 `AGENTS.md` MUST 第 2 条）。

## 2026-10-30：计划 2.40 第一次 P-A 运行（trait = 路径泛函）——**C1 不可解释**

**做了什么**：把 `d*` 从"教师强制 log-mass 梯度"换成路径泛函的 score-function 梯度
（`T(u)=E_{x~D}E_{y~p^u}[φ(y)]`，`∇_u T = E[(φ−b)∇_u log p^u]`），其余代数（`u=ŴB⁻¹d*`、`Q(S)`、贪心、预算）
一律不动。模型 Qwen3.5-4B bf16，8 条 hostility 上下文 × K=8 rollouts，T=1，max_new=32，φ = 两个毒性判官的 P(tox) 均值。

**结果**（`results/p_a_c1.json`、`results/p_a_reinforce_pilot.json`、`metadata.measured.plan240_p_a`）：

| 量 | 值 | 读法 |
|---|---|---|
| C0 | mean φ = 0.0405 / 0.0378（两 seed），frac>0.5 = 0 | 仪器可用，基线回复确实"正常" |
| **cos(g_RE,g_J2)** | **−0.0071 / −0.0670** | 数值上满足冻结的 "<0.30"，**但无效** |
| **cos(g_RE, g_RE′) 跨 seed** | **−0.0731** | **方向不可复现 ⇒ 上面那行是噪声** |
| cos(g_J2, g_J2′) 跨 seed | **1.0000** | 控制：机器确定性的，方差在目标函数里，不在代码里 |
| ‖g_RE‖/‖g_J2‖ | 0.034 | RE 梯度小 30 倍 |

**诊断（可测，不是猜）**：φ 在 b=0 几乎没有动态范围——均值 0.0405、逐上下文 0.0038–0.0489，
且 **s-nlp 在 64 条礼貌回复上全部恰好 0.0000**，unitary 给 0.0038–0.1060。于是权重 `(φ−b)` 量级 1e-4，
梯度被 `∇log p` 的采样噪声淹没。**根因是设计而非实现：b=0 时特性根本没有被诱发，所以它的梯度不可测。**

**修好的三个 bug（现在由 SELF 门守住，`|Δlogp| 1.4e-2`、逐层 cos ≥ 0.9994）**：
1. 抓取必须 `detach()` 后再 `clone()`（第一个注入点之上的输入已经是非叶子 ⇒ `t.grad=None`，只剩 layer 0）；
2. `generate` 必须给 `num_return_sequences`（否则 K=1）；
3. **损失 mask 不是 attention mask**——拿它当 attention 会切断对 prompt 的注意力，每一个 logp 都变（75.09 vs 17.80）。

**下一步（按证据）**：
1. **在诱发点上估**：把 anchor 设在可用带内（b≈0.1）或用一个 φ 真有方差的离线池子，先让 split-half cos ≥ 0.5
   再谈 cos(g_RE,g_J2)；否则任何"行为≠读数"的结论都是噪声。
2. 方差缩减优先于加上下文：per-context baseline、common random numbers、或对已有生成做重要性采样。
3. C2（同预算行为臂 ≥7/8）在 g_RE 可靠之前**不得**开跑。

## 2026-10-30：计划 2.41 P-B（零 GPU）——**层级的剂量假设被证伪**

冻结映射（跑前写好，模糊 tag 一律排除）：L1 = 说"自己是什么"（persona/identity/self-description/…），
L2 = 说"怎么说话"（style/funny/slang/voice/…）。neuronary 全量 200 条（含 58 条模糊、20 条未归类）。

| 组 | n | 聚合预算中位 | 区间 |
|---|---:|---:|---|
| L1-only | 44 | **0.849** | 0.339–1.649 |
| L2-only | 50 | **0.930** | 0.20–1.50 |

U=1136.5，**p=0.389**（单侧），且方向与预测**相反**（L2 高 9.5%）。⇒ **层级不决定剂量**，
neuronary 的 tag 空间里没有可利用的"剂量-层级"格子；"trait 作为格上的点"在这份语料上不成立。
两条附带观察：(a) 两个中位都落在 0.85–0.93，比我们实测的可用上限（冻结判据 b≤0.1；行为损坏从 b≈0.5 起）高 2–3 倍；
(b) 全量只记了 1 例复现失败，说明这里只发布成功，所以**即使存在层级效应也无法与选择效应分离**。

## 2026-10-30：计划 2.42（P-A'）——**score-function 路线按维度关闭**

**方法修正（不是调参）**：昨天失败在 φ 的标度（饱和判官 0.0405±0.01，权重 ~1e-4）。改成**路径版 J2**：
同一对敌意/赞美槽位，但条件从固定前缀换成模型自己的续写，量纲变成 nat——
**sd_total 1.33 / 组内 sd 0.43**（对照判官的 0.01）。C0 通过。

**精确分解，三项都测了（同一批 32 rollouts/上下文）**：

| 项 | 定义 | 实测 |
|---|---|---|
| score term | `E[(φ−b)·∇_u log p(y)]` | **r(32)=0.022**，且 r(K) 在 K=2→32 上**平**（0.002→0.022） |
| pathwise | `E[∇_u φ(y)]` | ‖g‖=**25.23**，与 teacher-forced 的 cos=**0.4187** |
| teacher-forced J2 | 既有构念（固定前缀） | ‖g‖=1355.19；与 score term cos=0.1621 |

**结论（两句，缺一不可）**：
1. **score term 在这一预算下不可估，而且不是"样本不够"**——r(K) 平说明噪声不可平均：
   逐 rollout 的 score 向量活在 294,912 维里、彼此近正交，要平均需要 `K ~ dim` 而不是 `K ~ 10²`。
   ⇒ 要拒绝的是**估计器类**，不是"再加 rollouts"。
2. **但它也不是小修正**：‖score‖ = 26.4 × ‖pathwise‖。所以"测度变化"这件事真的存在、且主要由不可测的那一项承载；
   可测的 pathwise 项只与 teacher-forced 方向 **0.42** 对齐——**这是第一次把"测度变化值多少"变成一个数**。

**留下的可算对象**：pathwise 项——沿模型自己轨迹做 teacher-forced backward，随机性只剩轨迹本身
（φ 组内 sd 0.43 nat，32 条轨迹 ⇒ SE≈0.076 nat），且它仍然是闭式解能吃的形式
（`u=ŴB⁻¹d*`、`Q(S)`、预算全不变），只是 `d*` 从"固定前缀"换成"沿采样轨迹的平均 teacher-forced 梯度"。

**不做**：不再试 anchor/K/baseline/判官组合。判据里那条"r(32)<0.5 ⇒ 记录为不可估，不再回调旋钮"已触发。

## 2026-10-30：计划 2.43——全支撑闭式解：**预算度量是奇异的**

把 `d*` 的两个候选（teacher-forced / pathwise）喂进同一个闭式解，跑在**全支撑 294,912 坐标**上。

**结构性发现（这条独立于任何科学结论）**：`M_l = WnᵀWn`，`Wn` 是 `2560×9216`，所以
**rank(M_l) = 2560 < 9216**，Cholesky 在 leading minor 2559 处失败，**全支撑 Gram 是奇异的**。
只有小支撑的主子块是正定的——**这正是我们全部既有结果都活在 32 坐标支撑上的结构性原因**；
它同时给支撑大小一个上界：**每层只有 2560 个可区分的注入方向，不是 9216**（其余 6656 个方向在预算里是免费的）。
解法用最小范数 `M⁺`（`Wn` 的薄 SVD），自检 `‖Mu−d‖/‖d‖ = 3.0e-6`、`budget² = 0.99882`（fp32 边界）。

| 量 | 值 |
|---|---|
| `cos_M(u_j2, u_pw)` | **+0.4306**（坐标 cos 0.4733；与 2.42 梯度层的 0.4187 一致） |
| top-64 / 256 / 1024 重叠 | Jaccard 0.255（26/64）、0.222（93）、0.163（287） |
| 读数 ΔJ2 @ b=0.05 | u_j2 **+1.744** vs u_pw **+0.776**（2.25×） |
| 读数 ΔJ2 @ b=0.10 | u_j2 **+3.203** vs u_pw **+1.467**（2.18×） |

**判读（重要，别误读）**：现役 `d*_j2` 在 teacher-forced 读数上赢 2.2× 是**必然的**——
它按构造就是该读数在该度量下的梯度。所以**这个实验无法裁决换不换 `d*`**：
两个方向各自对自己那个泛函最优。唯一能裁决的是**行为**，而行为梯度的路已经关了（2.42，`r(32)=0.022`）。
⇒ 剩下的唯一一个零旋钮问题是：**在固定 b 下把两臂的行为直接配对测**（不是从梯度推）。

## 2026-10-30：计划 2.44——**奇异性被消解（商空间），不是被正则化**

`M_l = WnᵀWn` 奇异是**规范自由**，不是病态：`ker(M)=ker(Wn)` 的剂量组合**残差效应为零**，
而 `d* = p50·Wnᵀ∇Φ` 落在 `range(Wnᵀ)` 里 ⇒ 核方向**既不加分也不花预算**，约束问题有界。

**换坐标就没了**：令 `v = Wn·u ∈ ℝ²⁵⁶⁰`（每层），那里预算就是普通欧氏范数，于是

```
v*_l = b · ∇Φ_l / ‖∇Φ‖          ← 一行，没有 Gram、没有 Cholesky、没有 SVD、没有 λ
```

**数值确认（layer 0/13/31）**：

| 量 | 0 | 13 | 31 |
|---|---|---|---|
| `cos(Wn·u_M, ∇Φ)` | **1.0000** | **1.0000** | **1.0000** |
| `cos(Wn·u_g, ∇Φ)` | 0.8742 | 0.7600 | 0.7695 |

⇒ 已发表的 **u_M 胜过 u_g 36–43%** 现在**被推导出来**了：它是 `1/cos`（层 13：1/0.76 = 1.32，实测 1.36–1.43）。
`M⁻¹` 这一步做的就是"把被下投影的二阶结构扭曲过的梯度还原成残差空间里未扭曲的那个方向"。

**为什么不能用 ridge**：`λ‖u‖²` 给零效应方向定了价，因此**改变最优解**；`M⁺/最小范数`不改最优解，只固定**报告哪个代表元**。

**它没有解决的是位置**：可行族是 `u + ker(Wn)`，每层 **6656 维** ⇒ 位置列表只是 2560 维对象的一个**代表元**。
最小范数挑的是**最不稀疏**的代表元，而社区挑稀疏的——这就是 1.7% 精确重叠的一部分来源。
⇒ 位置的公平比较必须在**商空间**里做；"稀疏代表元"是线性代数问题（ℓ1/稀疏表示），不是调参问题。

## 2026-10-30：计划 2.45——**ker(M) 假说被证伪**（位置问题不是代表元问题）

把两套位置都映射进商空间 `v = Wn·u`（每层 2560 维）再比余弦，与"同层直方图匹配的随机列表"作零分布：

| 比较 | 商空间 cos | 坐标空间 cos |
|---|---|---|
| 我们 K=64（仅 L13–14）vs 社区 164 | **0.0770** | 0.0090 |
| 我们 **32 层 K=1024 交付列表** vs 社区 164 | **0.0083** | 0.0037 |
| 同上，sign-only（纯方向） | **0.0058** | 0.0049 |
| 零分布（20 个匹配直方图随机列表） | 中位 0.0023，最大 0.0279 | — |

**判读**：商掉 6656 维核只把 2 层列表从 0.009 抬到 0.077（8.5×，约零分布上限的 3×），
而**真正的交付物（32 层 K=1024）仍是 0.0083，与随机无差别**；sign-only 复现同样次序。
⇒ **1.7% 的重叠缺口不是"同一个方向的另一种稀疏代表"，而是方向本身就不同**。
2.44 的规范结论（奇异=规范、商空间一行式）**成立**，但**它不能用来解释位置不可复现**。
保留的一点点信号：2 层小列表约 3× 高于零分布，与既有小 K 富集（7×/21×）一致。

⇒ 结论回到那条一致的老图景：**社区坐标由另一个目标选出，差异是方向性的，不是表示性的**。

## 2026-10-30：环境备忘——Lean 4 + mathlib（**公共目录，不在本仓**）

本仓是纯 Python；Lean 环境是为了把 2.44 的代数结论写成机器可检定理才装上的。位置与版本：

| 项 | 值 |
|---|---|
| 路径 | **`/home/ebt/lean/mathlib-v4.33.0-rc1`**（7.5 GB，公共目录，**不属于任何项目**） |
| Lean | `v4.33.0-rc1`（与 `~/.elan/toolchains` 里同名 toolchain 匹配；elan 由 `~/.elan/bin` 提供） |
| mathlib | rev `79d0395a1825a6264ad5d269e35e60537518955e`（`inputRev: v4.33.0-rc1`），**已预编译** |
| 来源 | `/home/ebt/Downloads/emsx.tar.zst` 里的 `emsx/leanproof`，2026-10-30 解出并搬出项目目录 |
| 说明 | 该目录内 `README.local.md` 记了来源、rev、用法与约束 |
| 快捷命令 | **`leanm FILE.lean`**（`~/.local/bin/leanm`，已在 PATH）＝ `cd 该目录 && lake env lean FILE.lean` |
| 连带 | `~/Downloads/kimi-research/plugin/lean` 符号链接已改指该目录 ⇒ `opl-leancheck` 默认项目可用 |

**三条硬约束（写进该目录的 README.local.md）**：
1. **不要改该目录的 `lean-toolchain`**——olean 是工具链锁定的，改了即全部失效；
2. 该目录**不是 git 仓**、也不要提交（7.5 GB 第三方快照，按 `AGENTS.md` 只记来源与版本）；
3. 换 mathlib 版本就**新建目录**，别在原地升级。

**不需要签名密钥**（更正上一版备忘的错误引导）：`opl-leancheck` 只在传 `--evidence-out` 时才调用签名器
（`bin/opl-leancheck` 第 124/150 行），不写证据文件时照常审计。所以命令是
`opl-leancheck --file lean/Gauge.lean --decl <thm>`，**不加** `--evidence-out`。

**形式化已开始**：`lean/Gauge.lean`（见 `lean/README.md`、`results/lean_audit.json`）三条全部
`PROVED`、公理只用 `{propext, Classical.choice, Quot.sound}`。两条诚实更正：
(a) `ker(W†W)=ker(W)` **mathlib 已有**（`ContinuousLinearMap.ker_adjoint_comp_self`）⇒ 2.44 的"核"那一半是教科书结论；
(b) 真正承重的 `lift_maps_back`（`W` 满射 + `M u = W† y` ⇒ `W u = y`）**不需要**最小范数、也不需要伪逆
（mathlib 没有 Moore–Penrose），它就是 `cos(W·u_M, ∇Φ)=1.0000` 的代数内容。

**第二个文件 `lean/Decomposition.lean`（计划 2.42 的分解）**：在**有限结果集**上形式化，六条合计全部 `PROVED`：
`expectation_deriv_decomp`（`d/du Σ p φ = Σ p ∂φ + Σ p φ ∂log p`）、`sum_p_score_eq_zero`（平均 score 为零）、
`baseline_free`（**baseline 在期望里完全抵消**，对任意 `b`——它只影响方差，这解释了为什么"减 baseline"不是调参）。
范围写死在文件头：不形式化"φ 是不是对的构念"，不形式化任何关于模型的断言，也不形式化估计量的有限样本方差。

**第三个文件 `lean/Dimension.lean`（计划 2.42 的维度论证）**：五条，全部 `PROVED`，**不需要概率**——
核心是一条恒等式：两两正交、等范数的族满足 `‖Σ εᵢxᵢ‖² = K·c²`（对**任意** `±1` 符号模式），
于是 `4⟪SA,SB⟫ = ‖Σx‖² − Kc²` **与切分无关**。这就是 `r(K)` 平坦的机制：正交时切分无可变化，
各向同性下 `‖Σx‖² ≈ Kc²` ⇒ 余弦在每个 K 都 ≈0。**各向同性模型本身是假设，`K ≈ d·v/s²` 是给定假设后的算术，
写在注释里、不是定理**。

## 2026-10-30：**撤回**"各向同性"解释（文献核对推翻了我的假设）

`lean/Dimension.lean` 文件头原写：逐 rollout score 向量各向同性 ⇒ `‖Σx‖² ≈ Kc²` ⇒ 余弦在每个 K 都 ≈0 ⇒ `K ≈ d·v/s²`。
查文献后**这条假设是错的**（详见 `doc/references/isotropy-and-noise-scale.md`）：

| 证据 | 结论 |
|---|---|
| `2401.12143` | 各向异性是 self-attention 的**固有**性质（原文 "anisotropy **is inherent** to self-attention"） |
| `2602.11185` (Spectra) | LLM 梯度**高度各向异性**：spike 只占约 **1.5%** 的方向却主导二阶统计量 |
| `2407.11239` / Aghajanyan et al. 2021 | 梯度子空间**低秩且稳定**（内在维度数百量级） |
| McCandlish et al. 2018（经二手引用） | 正确的量是 gradient noise scale `tr(Σ)/‖G‖²` |

**改法**：`cos ≈ ‖g‖²/(‖g‖² + tr(Σ)/K)`，临界 `K ≈ tr(Σ)/‖g‖²`；各向同性只是 `tr(Σ)=d·v` 的特例。
⇒ `r(K)` 平坦应读作"**`tr(Σ)` 没有被 K 个 rollout 平均下去**"，**不是**"K ~ d"。

**五条定理不受影响**（纯内积代数，没有假设各向同性）；只有**解释**错了。已在四处标 RETRACTED：
`lean/Dimension.lean` 文件头、`lean/README.md`、`results/lean_audit.json`、metadata。

**登记未执行的检验**：用 `p_a2_pathj2` 留下的 **256 个逐 rollout score 向量**（8 上下文 × 32 rollout，各 294,912 维）
估 `tr(Σ)` 与 `‖g‖²`，看 `tr(Σ)/‖g‖²` 是否与 `r(K)` 曲线反解的 `K*` 同量级（10 倍内即认为同一机制）。
这一步把"假设各向同性"换成一个**数字**，且不需要新测量。

### 2.46 已执行：噪声-信号比 = **1500**（把假设换成数字）

256 个逐 rollout score 贡献 `ξᵢ = (φᵢ−b)sᵢ`（8 上下文 × 32 rollout，dim 294,912）：

| 量 | 值 |
|---|---|
| `tr(Σ)` | **1.212e7** |
| `‖ξ̄‖²` / `tr(Σ)/n` | 55,422 / 47,344 ⇒ **`‖g‖² = 8,078 > 0`**（N0 通过：信号在 n=256 时仍可辨） |
| **噪声-信号比** | **1500.5**，故 `K* ≈ 1500` |
| Spearman–Brown 反解（2.42） | 1424 ⇒ **两条独立路径相差 5%** |

预测 `r(K) = ‖g‖²/(‖g‖²+tr(Σ)/K)`：

| K | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 预测 | 0.0013 | 0.0027 | 0.0053 | 0.0106 | **0.0209** |
| 实测 | 0.002 | 0.005 | 0.016 | 0.015 | **0.022** |

小 r 区 `r ≈ K/1500`：K=32 ⇒ 0.0213，与实测 0.022 差 5%。

**三点结论**：
1. `r(K)` 平坦**不再是假设**：它是 `r ≈ K/1500` 在小 r 区的形状（K 要 1500 才到 0.5）；
2. 环境维度 **不是**尺度——`K* = 1500` 比 `dim = 294,912` 小 **197 倍**，正是低秩/各向异性文献预言的方向；
3. 42 那条"平坦曲线使 S-B 反解不可信"**应部分撤回**：噪声尺度从原理上独立给出 1500，与 S-B 的 1424 吻合。

**保留的警告**：单 seed；`b` 由同 256 个样本估计且未校正（使 `‖g‖²` 偏小、`K*` 偏大）；逐上下文 `Σ‖ξ‖²` 从 5.2e7 到 1.4e9
（ctx 6 是 ctx 5 的 27 倍），**合并的 `tr(Σ)` 由少数上下文主导**，逐上下文的 `K*` 更诚实；未独立检验 256 个 `ξᵢ` 是否不相关。

## 2026-10-30（末）：外部核查 → **0.49.0，五条结论撤回**

外部核查（对 `1f50c279`）基本正确。我方逐条复核，决定性的一条已用实验确认：
**`detach()` 抓叶子切断了真实因果路径**。机制：`x_L = act(h_L)`，而 `h_L` 同时作为残差载体流入后续每一层的 MLP，
所以真实 `∂Φ/∂x_L` 含 `x_L → h_L → mlp_{L+1..}` 整族路径；detach 全切，**层号越早损失越大、末层无损**。

三重证据：① bf16 FD 不可用（精度）；② 同前向、`|Δloss|=0` 的两法对照，逐层 cos **L0 0.074 → L13 0.183 → L31 1.000**；
③ **fp32 FD 裁判**：层 13 `FD 38.53/38.66` vs 零叶子加法 `38.84` ✓ vs detach `30.76` ✗。

**撤回**：2.40 C1（→"无结论"）、2.42 的 `r(K)`/`0.4187`/`26.4×`、2.43 的 `cos_M=0.4306`/`2.25×`、2.46 的 `1500`。
**保留**：2.44 的商空间代数与 `cos(Wn·u_M,∇Φ)=1.0000`——它是**求解器对任意 `d*` 的恒等式**，与 `g` 是否为真梯度无关。
**降级**：2.45（无梯度参与，但零分布只匹配了 64 条列表的层直方图、剂量全 +1、跨 trait 合并）。

另接受：`g_pw` 在循环内初始化（只剩最后一个上下文）；pathwise 对 slot 长度除两次；`1500 vs 1424` 是 `κ` 与 `2κ` 混比（正确：1500 vs 711）；
"score 范数大 ⇒ 效应由 score 承载"不成立（`E‖ĝ‖²=‖g‖²+trΣ/m`）；`Q(S)` 贪心无次模保证；DPO 的对数比论证无效；计算噪声 ≠ 构念误差；
`Dimension.lean` 的 sha 因改文件头而失效。

**下一步（已排定，见 `doc/audit-response-2026-10-30.md` §12）**：抓取改零叶子加法 + **fp32 FD 仪器校准门**（层 0/13/31 误差 <2%）→
三分支统一泛函与归一化 → baseline 改 leave-one-out → 重跑 2.40/2.42/2.43/2.46 → 商空间比较重做 → `make check` 增加 sha 一致性检查。

## 2026-10-30（终）：0.55.0 —— 仪器修复完毕，五组数字重记

按 `doc/audit-response-2026-10-30.md` §12 的十一条全部执行完毕。**撤回 → 替换**如下（旧值仍在原处，标 RETRACTED）：

| 结论 | 撤回值 | 替换值 | 产物 |
|---|---|---|---|
| 2.40 可靠性 / cos | −0.0731 / −0.0071 / −0.0670 | `r(K)` seed0 −0.0075/−0.0120/−0.0274、seed1 0.0027/0.0029/0.0028；`cos` −0.0226/−0.0634；**κ 未定义** | `results/reliability_240.json` |
| 2.42 `r(K)` | 0.002 / 0.005 / 0.016 / 0.015 / 0.022 | **0.0065 / 0.0094 / 0.0179 / 0.0330 / 0.0685** | `results/reliability_242_summary.json` |
| 2.42 `cos(pathwise,teacher)` | 0.4187 | **0.6554**（逐层 0.266@L0 → 0.610@L13 → 0.902@L31） | 同上 |
| 2.42 `‖score‖/‖pathwise‖` | 26.4 | **0.7965** | 同上 |
| 2.46 噪声尺度 | 1500 与"1424 吻合 5%" | `trΣ` 39.73 [25.85,61.41] / 39.39 [23.95,63.77]；`‖g‖²` 0.0189 [0.0605,0.4709] / 0.0785 [0.0760,0.8058]（跨组差 4 倍，seed0 点估计在区间外）；`K(r=0.5)=2κ` ⇒ 曲线反解 435/377、池化 4201/1004 ⇒ **10²–10³** | `results/noise_scale_v2{,_seed1}_raw.json` |
| 2.43 `cos_M` / 读数差 / top-64 | 0.4306 / 2.25×,2.18× / 26 | **0.6566 / 1.3098×,1.1549× / 36** | `results/closed_form_v3.json` |
| 2.45 商空间余弦 | 0.0770 / 0.0083 | 32 层 K=1024：逐配方中位 **−0.00026**、超零分布 p95 **7.9%**；K=64 的 0.000/2.1% 是构造性结果，不作证据 | `results/quotient_compare_v2.json`、`_raw.json` |

**仪器状态**：`scripts/mlp_capture.py` 是唯一读取实现（零叶子、图不断开），三处脚本全部 import 它；
`scripts/fd_gate.py` 用 fp32 有限差分认证它（最坏 0.004654，旧实现 0.139–0.242），记录绑定仪器哈希；
`make check` 新增 `instruments` 闸门（Lean sha + 校准门最近一次通过），两种失败模式均实测触发过。

**净结论（三句，措辞按证据强度）**：
1. score 项**不是** 26 倍，与 pathwise 同量级（0.7965），但在 K=32 下 `r=0.0685` 仍不可用作方向——**预算**结论，不是"估计器类关闭"。
2. 测度变化的代价是 **cos 0.655**，且**分层**：早层 0.27、末层 0.90 ⇒ teacher-forced 代理主要对**后层**可用。
3. 样本需求只能给 **10²–10³ 量级**（r=0.5 需 K=2κ）：曲线反解 K=435（seed0）/377（seed1），池化 2κ=4201/1004；`trΣ` 跨组稳定而 `‖g‖²` 差 4 倍、seed0 点估计落在自己区间外 ⇒ 8 上下文不足以定位，**不给定点值**。

**仍未做**：盲法 target-directedness 标注（人工，48 条）；与 CANS/Fisher 的能力-行为曲线；多 seed 稳定性；
`make check` 仍有 10/13 个测试文件需 torch 被跳过（它只闸仪器身份与 Lean 记录，不重推实验结论）。
