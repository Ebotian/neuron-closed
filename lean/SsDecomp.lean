/- The ANOVA decomposition: the code's SUBTRACTION is tied to the nonnegative between-group quantity.

`Icc.lean` proves `SS_b` is nonnegative when defined as `K·Σ_c‖x̄_c − x̄‖²`. This file is meant to prove that
the quantity the code actually forms by subtraction,
    ss_t − ss_w  with  ss_t = Σ_p‖x_p‖² − ‖Σ_p x_p‖²/(K·C),  ss_w = Σ_c (Σ_i‖x_ci‖² − ‖Σ_i x_ci‖²/K),
equals `K·Σ_c‖x̄_c − x̄‖²`. Until the full chain is proved, nothing forbids the code's `ss_b` from being slightly
negative.

STATUS: this file proves the PER-GROUP split (`scatter_split`), which is the mathematical heart. Still missing to
close the code's subtraction:
  (a) `Σ_i ‖x_i‖² − ‖Σ_i x_i‖²/K = Σ_i ‖x_i − mean x‖²`   (the "variance identity", bridges SswNormalised.sswU
      and the deviation form used here)
  (b) the same identity at the pooled level, i.e. `ss_t` in the code's form equals `Σ_c Σ_i ‖x_ci − x̄‖²`
  (c) `ss_b` in the code's subtraction form equals `K Σ_c ‖x̄_c − x̄‖²`
Summing `scatter_split` over groups with `g` = grand mean then gives `ss_t = ss_w + ss_b`. None of (a)-(c) is proved yet.
-/
import Mathlib

open scoped BigOperators InnerProductSpace

namespace NeuronClosed

variable {ι γ : Type*} [Fintype ι] [Fintype γ] [Nonempty ι]
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

noncomputable def mean (x : ι → V) : V := ((Fintype.card ι : ℝ))⁻¹ • ∑ i, x i

/-- The per-group identity: scatter about the grand mean splits into scatter about the group mean plus the
group-mean displacement. This is the whole content of the decomposition, once per group. -/
theorem scatter_split (f : ι → V) (g : V) :
    (∑ i, ‖f i - g‖ ^ 2)
      = (∑ i, ‖f i - mean f‖ ^ 2) + (Fintype.card ι : ℝ) * ‖mean f - g‖ ^ 2 := by
  have hzero : (∑ i, (f i - mean f)) = 0 := by
    have hK : ((Fintype.card ι : ℝ)) ≠ 0 := by
      have := Fintype.card_ne_zero (α := ι); exact_mod_cast this
    rw [Finset.sum_sub_distrib]
    have h2 : (∑ _i : ι, mean f) = (Fintype.card ι : ℝ) • mean f := by
      rw [Finset.sum_const, Finset.card_univ]
      exact (Nat.cast_smul_eq_nsmul ℝ (Fintype.card ι) (mean f)).symm
    rw [h2]
    unfold mean
    rw [smul_smul, mul_inv_cancel₀ hK, one_smul]
    abel
  have hexp : ∀ i, ‖f i - g‖ ^ 2 = ‖f i - mean f‖ ^ 2 + 2 * ⟪f i - mean f, mean f - g⟫_ℝ
      + ‖mean f - g‖ ^ 2 := by
    intro i
    have : f i - g = (f i - mean f) + (mean f - g) := by abel
    rw [this, norm_add_sq_real]
  rw [Finset.sum_congr rfl (fun i _ => hexp i), Finset.sum_add_distrib, Finset.sum_add_distrib]
  have hcross : (∑ i, 2 * ⟪f i - mean f, mean f - g⟫_ℝ) = 0 := by
    rw [← Finset.mul_sum, ← sum_inner, hzero, inner_zero_left, mul_zero]
  have hconst : (∑ _i : ι, ‖mean f - g‖ ^ 2) = (Fintype.card ι : ℝ) * ‖mean f - g‖ ^ 2 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [hcross, hconst, add_zero]

end NeuronClosed

#print axioms NeuronClosed.scatter_split
