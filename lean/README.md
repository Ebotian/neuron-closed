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
| **`SsDecomp.lean`** | `scatter_split` / `variance_identity` / `pooled_variance_identity` / **`ssb_is_between_scatter`** | **ANOVA 分解**：(a) 方差恒等式、(b) 池化层同一条、**(c) `Σ_cΣᵢ‖X_ci−x̄‖² − Σ_cΣᵢ‖X_ci−x̄_c‖² = K·Σ_c‖x̄_c−x̄‖²`**——即**代码那条减法**等于 `Icc.lean` 证明非负的那个量 |

跑：`leanm lean/Icc.lean`（每个文件独立）。审计：`opl-leancheck --file lean/Icc.lean --decl NeuronClosed.icc_bounds`，
记录在 `results/lean_audit.json`，`make check` 的 `instruments` 项校验 sha 一致。

## 仍未形式化（不得当作已证）

- **还缺一条算术桥**：代码把 `sswU` 写成 `Σ‖x‖² − ‖Σx‖²/K`，而 `SsDecomp` 用的是偏差形式。差的是
  `K‖mean f‖² = ‖Σ f‖²/K`（由 `mean = K⁻¹ • Σf` + `norm_smul` + `abs_of_pos` 两行可得）；有了它，(a)+(c) 就能改写成
  代码里字面的 `sswU` 形式。**在那之前，"代码的 `ss_b` 等于非负量"这句话严格说只对偏差形式成立。**
- `_f_quantile` 的数值实现仍未验证。
- `‖g‖² = (MS_b − MS_w)/k` 的换算未证。
- `_f_quantile` 的数值实现未验证。
