# lean/ — 每条定理一个文件

**分文件的理由**（来自一次真实的失败）：这些声明原来都在一个 91 行的 `Reliability.lean` 里，我用字符串切片插入新内容，
连续三次把文件结构切坏（`unexpected token '/--'`、`expected 'lemma'`）；另外 `plug_in_cos_moves` 两次失败只是因为我漏了
`open scoped InnerProductSpace`——在一个单文件里这种遗漏很难定位。**一个定理组一个文件、各自独立编译**之后，五条声明一次通过。

| 文件 | 定理 | 内容 |
|---|---|---|
| `Gauge.lean` | `ker_gram` / `gauge_invariance` / `lift_maps_back` | 2.44：奇异度量是规范；解在残差空间的像就是 `∇Φ` |
| `Decomposition.lean` | `expectation_deriv_decomp` / `sum_p_score_eq_zero` / `baseline_free` | 2.42：pathwise + score 分解；baseline 在期望里免费 |
| `Dimension.lean` | 5 条 | 2.42：正交族下 split-half 与切分无关 |
| **`Ssw.lean`** | `ssw_shift_invariant` | **R1**：组内平方和（K² 缩放版）对**公共平移精确不变** |
| **`SswNormalised.lean`** | `sum_norm_add_sq` / `sswu_shift_invariant` | **R3**：代码实际算的**归一化**形式 `Σ‖xᵢ‖² − ‖Σxᵢ‖²/K` 同样平移不变 |
| **`Icc.lean`** | `icc_bounds` / `ssbU_nonneg` / `icc_bounds_from_sums` | **R2**：`ICC ∈ [−1/(n−1), 1]`；**R4**：`SS_b` **按构造非负** ⇒ R2 的前提 `MSb ≥ 0` 是推论而非假设 |
| **`Witness.lean`** | `plug_in_cos_moves` | **R5**：公共平移下 plug-in 平方余弦 **0 → 1/2**，而 `SS_w` 不变——反面论断，此前只是文字 |
| **`Mean.lean`** | `sum_sub_mean_eq_zero` | 组内偏差之和为零（ANOVA 分解所需的事实） |
| **`SsDecomp.lean`** | `scatter_split` / `variance_identity` / `pooled_variance_identity` / `ssb_is_between_scatter` / `card_mul_mean_norm_sq` / `sum_prod_nested` / **`ssb_code_form`** | **ANOVA 分解（含代码字面形式）**：(a) 方差恒等式、(b) 池化层同一条、**(c) `Σ_cΣᵢ‖X_ci−x̄‖² − Σ_cΣᵢ‖X_ci−x̄_c‖² = K·Σ_c‖x̄_c−x̄‖²`**——即**代码那条减法**等于 `Icc.lean` 证明非负的那个量 |

跑：`leanm lean/Icc.lean`（每个文件独立）。审计：`opl-leancheck --file lean/Icc.lean --decl NeuronClosed.icc_bounds`，
记录在 `results/lean_audit.json`，`make check` 的 `instruments` 项校验 sha 一致。

## 仍未形式化（不得当作已证）

- **已闭合**：`ssb_code_form` 证明**代码字面的**减法 `(Σ_cΣᵢ‖X_ci‖² − ‖Σ_pX_p‖²/K_tot) − Σ_c(Σᵢ‖X_ci‖² − ‖ΣᵢX_ci‖²/K)`
  等于 `K·Σ_c‖x̄_c − x̄‖²`，即**代码里的 `ss_b` 就是 `Icc.lean` 证明非负的那个组间散布**。配合 R4，`icc_bounds` 的前提
  `0 ≤ MSb` 成为推论，代码不需要检查。串联的关键是：用 `let F` 给拉平后的族命名、用 `sum_prod_nested` 把扁平求和转成嵌套求和，
  再交给 `linarith`——之前的失败全是语法形状不匹配，不是数学。
- `_f_quantile` 的数值实现仍未验证。
- `‖g‖² = (MS_b − MS_w)/k` 的换算未证。
- `_f_quantile` 的数值实现未验证。
