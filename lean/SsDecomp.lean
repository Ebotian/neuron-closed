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

variable {ι γ : Type*} [Fintype ι] [Fintype γ] [Nonempty ι] [Nonempty γ]
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

/-- (a) The variance identity: scatter about the mean equals the raw sum of squares minus `K‖mean‖²`.
This is `scatter_split` with `g = 0` and is the bridge between the code's `sswU` and the deviation form. -/
theorem variance_identity (f : ι → V) :
    (∑ i, ‖f i - mean f‖ ^ 2)
      = (∑ i, ‖f i‖ ^ 2) - (Fintype.card ι : ℝ) * ‖mean f‖ ^ 2 := by
  have h := scatter_split (V := V) f (0 : V)
  simp only [sub_zero] at h
  linarith

/-- (a') The same at the POOLED level, i.e. exactly `sswU` applied to the flattened data `X : γ × ι → V`.
So `ss_t` in the code's form equals the top-level scatter about the grand mean: this is (b). -/
theorem pooled_variance_identity (X : γ × ι → V) :
    (∑ p, ‖X p - mean X‖ ^ 2)
      = (∑ p, ‖X p‖ ^ 2) - (Fintype.card (γ × ι) : ℝ) * ‖mean X‖ ^ 2 :=
  variance_identity X

/-- The grand mean of the two-level data, over the product index. -/
noncomputable def gbar (X : γ → ι → V) : V := mean (fun p : γ × ι => X p.1 p.2)

/-- (c) **The code's subtraction equals the nonnegative between-group quantity**, in deviation form:
`Σ_c Σ_i ‖X_ci − x̄‖²  −  Σ_c Σ_i ‖X_ci − x̄_c‖²  =  K · Σ_c ‖x̄_c − x̄‖²`.

The left-hand side is `ss_t − ss_w` once the two variance identities (a)/(a') are used to rewrite the code's
`Σ‖x‖² − ‖Σx‖²/K` forms into deviations. The right-hand side is `ssbU`, which `Icc.lean` shows is nonnegative by
construction — so this is the theorem that forbids the code's `ss_b` from being negative. -/
theorem ssb_is_between_scatter (X : γ → ι → V) :
    (∑ c, ∑ i, ‖X c i - gbar X‖ ^ 2) - (∑ c, ∑ i, ‖X c i - mean (X c)‖ ^ 2)
      = (Fintype.card ι : ℝ) * ∑ c, ‖mean (X c) - gbar X‖ ^ 2 := by
  have hper : ∀ c, (∑ i, ‖X c i - gbar X‖ ^ 2)
      = (∑ i, ‖X c i - mean (X c)‖ ^ 2)
        + (Fintype.card ι : ℝ) * ‖mean (X c) - gbar X‖ ^ 2 :=
    fun c => scatter_split (X c) (gbar X)
  have hsum : (∑ c, ((∑ i, ‖X c i - gbar X‖ ^ 2) - (∑ i, ‖X c i - mean (X c)‖ ^ 2)))
      = ∑ c, (Fintype.card ι : ℝ) * ‖mean (X c) - gbar X‖ ^ 2 :=
    Finset.sum_congr rfl (fun c (_ : c ∈ Finset.univ) => by rw [hper c]; ring)
  rw [← Finset.sum_sub_distrib, hsum, Finset.mul_sum]

end NeuronClosed

#print axioms NeuronClosed.scatter_split
#print axioms NeuronClosed.variance_identity
#print axioms NeuronClosed.pooled_variance_identity
#print axioms NeuronClosed.ssb_is_between_scatter
