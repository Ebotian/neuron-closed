# 11 — 各向同性假设的文献核对：它是错的，正确的替代量是"噪声-信号比"

**为什么有这份文件**：`lean/Dimension.lean` 的文件头写了一个假设——"逐 rollout 的 score 向量近似各向同性，
于是 `‖Σx‖² ≈ Kc²`、余弦在每个 K 都 ≈0、所需 `K ≈ d·v/s²`"。这条假设**在文献里是被明确否定的**，
所以它不能作为我们 `r(K)` 平坦的机制解释。本文件钉记录推翻它的证据，并给出应该替代它的量。

性质：① 引文逐字（带 arXiv 编号）；② 我方转述；③ 我方判断（标 **判断**）。

---

## 1. 表示空间不是各向同性的（① 摘要级，② 我方转述）

| 论文 | 原文/摘要要点 |
|---|---|
| `2401.12143` *Anisotropy Is Inherent to Self-Attention in Transformers* | "anisotropy, a singular property of hidden representations which makes them unexpectedly close to each other in terms of angular distance (cosine-similarity)… We show in this paper that anisotropy **is inherent** to self-attention" |
| `2602.11185` *Spectra: Rethinking Optimizers for LLMs Under Spectral Anisotropy* | "Gradient signals in LLM training are **highly anisotropic**: … the spike occupies only about **1.5%** of directions yet dominating optimizer statistics. This dominance suppresses tail learning" |
| ACL 2026 `2026.acl-long.1380` *Embracing Anisotropy* | "LLMs exhibit **highly anisotropic internal representations**, often characterized by massive activations, … a small subset of feature dimensions possesses magnitudes significantly larger than the rest" |
| `2212.02083` *On the Overlooked Structure of Stochastic Gradients* | 随机梯度有**被忽略的结构**（非各向同性、非简单重尾） |

⇒ **② 我方转述**：在这类模型上，"各向同性噪声"不是可用的默认假设，而是被反复测量否定的对象；
梯度/表示的谱是 **spike + long tail**，spike 占约 1.5% 的方向却主导二阶统计量。

## 2. 正确的替代量：噪声-信号比（gradient noise scale）

`McCandlish et al. 2018`（*An Empirical Model of Large-Batch Training*，**未核原文**）提出
`B_simple = tr(Σ) / ‖G‖²`，其中 `Σ` 是逐样本梯度的协方差、`G` 是真实梯度。
我方核到的二手表述（NeurIPS 2023 WANT workshop 论文，① 逐字）：

> "`B_simple = tr(Σ) / ‖G‖²` … this approximation is denoted as 'simple' because it assumes that the Hessian is diagonal"

⇒ **② 转述**：决定"要平均多少个样本才能把噪声压到与信号可比"的是 `tr(Σ)/‖G‖²`，
**不是环境维度 d**。各向同性只是 `tr(Σ) = d·v` 的特例——那时两者才等价。

**这一条直接改写我们的结论**：`K ≈ d·v/s²` 应改为 `K ≈ tr(Σ)/‖g‖²`。
用哪个形式取决于 `tr(Σ)` 是否被少数方向主导（文献说：是）。

## 3. 低秩/内在维度：即使各向同性也该用有效秩（① 摘要级）

- `Aghajanyan et al. 2021`（ACL 2021, *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*）：
  微调梯度落在远低于参数数的子空间里（**hundreds** 量级）。
- `2407.11239`（*From Low Rank Gradient Subspace Stabilization to Low-Rank Weights*）：梯度子空间**低秩且稳定**。

⇒ **② 转述**：把 `d = 294,912` 直接代入"需要 `K ~ d`"也偏严——合适的量是**有效秩/噪声-信号比**。

## 4. score-function 估计量本身（① 摘要级）

- `Mohamed et al., JMLR 21 (2019)`, *Monte Carlo Gradient Estimation in Machine Learning*：
  score-function 估计量"often not useful in practice due to its **large sample-to-sample variance**"，
  缓解手段是 control variates（保持无偏）。
- `1910.01249`（*Analyzing the Variance of Policy Gradient Estimators for the LQR*）：系统分析 REINFORCE 方差。

## 5. 对 `lean/Dimension.lean` 的影响（③ 我方判断）

1. **文件里那五条定理本身不受影响**——它们是纯内积代数（正交族的 split-half 恒等式），没有假设各向同性。
2. **文件头那段"各向同性 ⇒ cos≈0 于每个 K ⇒ K≈d·v/s²"必须撤下**，改为：
   `cos ≈ ‖g‖²/(‖g‖² + tr(Σ)/K)`，临界 `K ≈ tr(Σ)/‖g‖²`；各向同性只是它的特例。
3. **`r(K)` 平坦的正确解释**因此不是"各向同性"，而是"**`tr(Σ)` 在 K 个 rollout 上没被平均下去**"，
   这可以由我们自己已有的 256 个逐 rollout score 向量**直接测量**（见下）。

## 6. 由此产生的一个可测判据（登记，未执行）

我们有 `p_a2_pathj2` 运行留下的 256 个逐 rollout score 向量（8 上下文 × 32 rollout，各 294,912 维）。
- 算 `tr(Σ)`（用 `Σ‖δ_i‖²/ (n−1)` 估）与 `‖g‖²`（均值向量的平方范数）⇒ **噪声-信号比**；
- 判据：`预测 K* = tr(Σ)/‖g‖²` 与实测 `r(K)` 曲线反解出的 `K*` 是否同量级（10 倍以内即认为是同一机制）。
- 这条把"假设各向同性"换成一个**数字**，且不需要新测量。

---

## 核查范围

**核了**：上述 arXiv 编号对应的摘要（`2401.12143`、`2602.11185`、`2026.acl-long.1380`、`2212.02083`、
`2407.11239`、`1910.01249`、ACL 2021 `2021.acl-long.568`）与 NeurIPS 2023 WANT workshop 论文中的
`B_simple` 公式（逐字）。
**没核**：`McCandlish et al. 2018` 原文（只经二手引用）；`Mohamed et al. JMLR` 的全文（只摘要级）；
未把 `Spectra` 的 1.5% 复算；未在我们的模型上测 `tr(Σ)`；未核任何一篇的版本号。
