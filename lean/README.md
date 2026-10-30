# lean/ — 计划 2.44 的代数内容，机器可检

本目录属于本仓，但**不依赖**本仓的任何东西；它只依赖本机公共的 Lean 环境
（`/home/ebt/lean/mathlib-v4.33.0-rc1`，见 `doc/NEXT_SESSION.md` 的环境备忘）。

## 跑

```bash
leanm lean/Gauge.lean        # 编译 + 打印 #print axioms
```

`opl-leancheck --file lean/Gauge.lean --decl NeuronClosed.lift_maps_back`（**不要**加 `--evidence-out`，
否则才需要签名密钥）给出独立判定。

## `Gauge.lean` 里的三条

记号：坐标空间 `E`（每层 9216 维），残差空间 `F`（每层 2560 维），`W : E →L[ℝ] F` 是下投影，
预算度量 `M := W† ∘ W`（unit `W_down` 列的 Gram）。因为 `dim E > dim F`，`M` 奇异。

| 定理 | 陈述 | 对应 |
|---|---|---|
| `ker_gram` | `ker M = ker W` | 2.44：奇异＝投影的核。**注意这条 mathlib 已有**（`ContinuousLinearMap.ker_adjoint_comp_self`），我们只是把它作为对象重述——它是教科书结论，不是我们的发现 |
| `gauge_invariance` | `u₀ ∈ ker M ⇒ W (u + u₀) = W u` | 核方向零效应零收益：加它不改变扰动 |
| `lift_maps_back` | `W` 满射且 `M u = W† y ⇒ W u = y` | 2.44 的数值事实 `cos(W·u_M, ∇Φ) = 1.0000` 的代数内容：任何解在残差空间的像**就是** `y` |

第三条比数值陈述更强：它**不**要求 `u` 是最小范数解，也不需要 Moore–Penrose（mathlib 里没有）；
只要 `W` 满射、`M u = W† y`，结论就成立。满射在本问题里就是 `2560 ≤ 9216` 且行满秩。

## `Decomposition.lean` 里的三条（计划 2.42）

在**有限结果集** `Y` 上陈述（于是全部是普通有限和 + `Real.log`，不需要测度论）：

| 定理 | 陈述 | 意义 |
|---|---|---|
| `expectation_deriv_decomp` | `d/du Σ_y p φ = Σ_y p ∂_u φ + Σ_y p φ ∂_u log p` | 期望的导数 = **pathwise 项的期望 + score 项**；2.42 只测了后者 |
| `sum_p_score_eq_zero` | `Σ_y p ∂_u log p = 0`（只要 `Σ_y p ≡ 1`） | 平均 score 恒为零 |
| `baseline_free` | `Σ_y p (φ−b) ∂_u log p = Σ_y p φ ∂_u log p`，对任意 `b` | **baseline 在期望里完全抵消**——它只影响方差（有限样本下才起作用） |

`baseline_free` 正是我们实测里当作理所当然的那件事：`b` 的选取不改变目标，只改变估计量的方差。

## `Dimension.lean` 里的五条（计划 2.42 的维度论证）

**不需要概率**：全部是内积代数。

| 定理 | 陈述 |
|---|---|
| `norm_sq_sum_of_pairwise_orthogonal` | 两两正交的族：`‖Σx‖² = Σ‖x‖²` |
| `norm_sq_sum_smul_of_pairwise_orthogonal` | 带符号版：`‖Σ εᵢxᵢ‖² = Σ εᵢ²‖xᵢ‖²` |
| `four_mul_inner_split` | `4⟪SA,SB⟫ = ‖SA+SB‖² − ‖SA−SB‖²` |
| `norm_sq_alternating_eq_card_mul` | **`εᵢ = ±1` ⇒ `‖Σ εᵢxᵢ‖² = K·c²`，与符号模式无关** |
| `four_mul_inner_of_alternating` | **⇒ `4⟪SA,SB⟫ = ‖Σx‖² − Kc²`，与切分无关** |

这就是实测 `r(K)` 平坦（K=2…32 → 0.002…0.022）的机制：逐 rollout 的 score 向量若两两正交，
**切分无可变化**，于是 split-half 量只能由比值 `‖Σx‖²/(Kc²)` 决定；各向同性下 `‖Σx‖² ≈ Kc²`，
余弦在**每个 K** 都 ≈ 0。**注意**：各向同性模型本身是假设，`K ≈ d·v/s²` 是"给定假设后的算术"，
写在文件注释里、**不是**定理。

## 审计（2026-10-30）

- toolchain `leanprover/lean4:v4.33.0-rc1`；mathlib rev `79d0395a1825a6264ad5d269e35e60537518955e`
- 三个文件、共十一条，全部 `propext, Classical.choice, Quot.sound`（无 `sorryAx`、无自定义 axiom）
- 独立判定 `PROVED`（`opl-leancheck`，无 `--evidence-out`）
- 数值与 sha256 见 `results/lean_audit.json`

## 边界

形式化的是**代数命题**，不是 Qwen3.5-4B，也不是"我们的构念就是那个特性"。
它只证明：在给定 `W` 与度量 `M` 下，核是规范、解在残差空间的像不偏移。
