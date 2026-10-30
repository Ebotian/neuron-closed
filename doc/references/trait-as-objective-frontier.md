# 12 — "把特性写成一个目标函数"这件事，领域做到哪了

**为什么有这份文件**：本仓的闭式解只差两处输入——`φ`（特性的泛函）与代表元（位置）。前者的困难在于
"自然语言特性 → 数学目标"本身，这是奖励设计/构念效度问题，不是本仓独有的。本文件钉记录 2025–2026 的前沿状态，
并给出对我们 `φ` 的三条直接可用的结论。

性质：① 逐字引（带编号）；② 我方转述；③ 我方判断（标 **判断**）。

---

## 1. 领域自己承认这是开放问题（① 逐字）

`arXiv:2505.02666v2` *A Survey on Progress in LLM Alignment from the Perspective of Reward Design*：

> "reward design remains a **fundamentally open and challenging problem**. RMs, as surrogate objectives, are
> designed to approximate the true preferences and expectations of human users. However, the inherent complexity
> and subjectivity of human values … make it **nearly impossible to construct a reward function that fully
> captures these nuanced preferences**."

⇒ **② 转述**：用户的判断（"这近似于损失函数寻找问题"）就是该领域对自身状态的表述，不是外行的类比。

## 2. 失效模式已被命名并分级（① 摘要级 / ② 转述）

- *Springer 2026* `10.1007/s44163-026-01980-z`：给出 reward hacking **四级升级税则**，从 feature-level
  exploitation（verbosity、sycophancy）到 agentic 场景。
- `arXiv:2604.13602v1`：奖励黑客"mechanisms, emergent misalignment"。
- `ACM CSUR 3770749` (2025)：proxy 与 true reward 的差距常表现为奖励曲线的**尖锐相变**。

⇒ **③ 判断**：这与我们 2.28/2.31 的观测同构——**读数平滑上升、行为在某个预算处突变**（b=0.2–0.3 跨 0.5）。
我们不是遇到怪现象，是撞上了该领域公认的机制。

## 3. 前沿对"不可验证特性"的当前最佳答案：**rubric 式稠密奖励**（① 摘要级）

`arXiv:2507.17746` *Rubrics as Rewards (RaR)*：把 instance-specific **checklist 式 rubric** 直接当奖励，
把 RLVR 从"有明确对错"的推理任务推广到主观/多准则任务（医学 QA 等）。

⇒ **③ 判断（对我们直接有用）**：我们的 J2 本质是**一条 rubric**（"敌意词槽位的 log-mass 高于赞美槽位"）。
RaR 的做法是把它拆成多条子准则再加权合成。这条给我们一个**不是调参**的升级路径：把 `φ` 从 1 条准则
扩成 3–5 条**预先登记**的子准则（例如：敌意词槽位、第二人称指向、拒绝/威胁句式、能力保持），
每条单独可测，合成权重**由构造定**（等权或按子准则的可靠度），而不是搜出来的。

## 4. 构念效度已有可执行的检查表（① 逐字）

`NeurIPS 2025 D&B` *Measuring what Matters: Construct Validity in LLM Benchmarks*（445 个基准、29 位评审）
配套公开的 checklist 第一条逐字：

> "Provide a **precise and operational definition** for the phenomenon being measured"
> "Identify if the phenomenon has **sub-components** and ensure they …"

⇒ **③ 判断**：这份 8 条清单可以直接当作 `φ` 的验收表。我们当前的缺口很具体：
我们的"敌意"**没有 operational definition**，只有词表；**没有声明 sub-components**；
**没有声明 excluded aspects**（我们在 §2.13 结论四自己记过：分不开"敌意"与"其他评判性语言"）。

## 5. 目标函数＝对数比的这条线，已经是主流形式（① 摘要级）

DPO 家族：`r(x,y) = β·log[π_θ(y|x)/π_ref(y|x)]`（隐式奖励）。2025 年的进展把它系统化：
`NeurIPS 2025` *Implicit Reward as the Bridge*（SFT/DPO 统一视角）、
`NeurIPS 2025` *Preference Optimization by Estimating the Ratio of the Data Distribution*（把 DPO 看成比值估计）、
`arXiv:2506.01523v1` *Alignment as Distribution Learning*（显式建模偏好分布优于隐式 BT）。

⇒ **③ 判断**：`φ` 应该写成**对数似然比**（nat 量纲、有真实动态范围），而不是饱和分类器的概率。
这正是 2.42 修掉的那个错误（判官版 φ ≈ 0.04±0.01 → 路径 J2 的 sd 1.33 nat），现在有文献依据：
隐式奖励的范式就是 `log π/π_ref`。

## 6. 控制理论给出了"目标只要在对的地方对"的定量形式（① 摘要级，公式逐字）

`arXiv:2607.10362v1` Theorem 1：

> "For a planner that executes `û = argmin_{u∈U} D̂(u)`, `D(û) − D(u⋆) ≤ 2 sup_{u∈U} |D̂(u) − D(u)|`."

`arXiv:2506.04497` *Maximizing the Value of Predictions in Control: Accuracy Is Not Enough* 同旨。

⇒ **③ 判断（最重要的一条）**：**`φ` 不需要全局正确，只需要在我们真正提交的那个候选集上误差有界**，
且次优性 ≤ 2×该上确界差。这把"φ 难定义"从"必须定义对"改写成"**必须在你操作的点上把误差测出来**"——
后者我们做得到（2.17 已测 ΔJ2 的批噪声 1.19e-2；2.46 已测噪声-信号比 1500）。

## 7. 前沿**没有**的（③ 判断）

1. 没有人把"特性"形式化成**路径测度上的泛函 + 显式上下文测度 `D`**。`E_{x~D}` 里的 `D` 在奖励建模里
   就是训练分布，但**没有人把它当"特异性"的定义**——这正是我们 R2 卡住的那一角。
2. 没有人给出"给定 `φ` 后，**在物理预算度量下**的最优支撑选择"（我们的 `v*` 与商空间）。
   控制理论那条线只谈 predict-vs-control 的界，不谈度量。
3. RaR 的 rubric 权重仍靠人工/经验设定，**没有"权重由构造决定"的原则**——所以我们按等权或可靠度定，
   是可以写成原则的，而不是偷懒。

## 8. 由此对 `φ` 的具体改法（③ 判断，登记未执行）

1. 按 checklist 第 1–2 条，给"敌意"写 **operational definition + 3–5 个子准则 + 排除项**（1 页纸，零 GPU）；
2. `φ` 用**对数似然比**形式（我们已经是 nat 量纲的路径 J2，符合 §5）；
3. 子准则**等权**合成（不搜权重），并报告每个子准则单独的 (效应, 单位, 误差)；
4. 验收用控制界：在 `b ∈ {0.05, 0.1}` 的候选集上测 `sup|φ̂ − φ|`，把它写进结论，而不是只报点估计。

---

## 核查范围

**核了**：`2505.02666v2` 摘要中"fundamentally open / nearly impossible"逐字；NeurIPS 2025 D&B
*Measuring what Matters* 的 checklist 前两条逐字；`2607.10362v1` Theorem 1 的公式逐字；
`2507.17746`、`2604.13602`、`2405.02809`、`2506.01523`、`2506.04497`、Springer `10.1007/s44163-026-01980-z`、
`ACM CSUR 3770749`、NeurIPS 2025 两篇 DPO 相关论文的**摘要级**内容。
**没核**：上述任何一篇的全文；DPO 家族三篇的具体定理与实验条件；RaR 的 rubric 生成方式细节；
`2505.02666` 的版本号（只核了标题与 v2 链接）；未复算任何数字。
