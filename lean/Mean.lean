/- Auxiliary for the ANOVA decomposition: the deviations from the group mean sum to zero.

Small and independently compiled, because the decomposition proof needs exactly this lemma.
-/
import Mathlib

open scoped BigOperators

namespace NeuronClosed

variable {ι : Type*} [Fintype ι]
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- The group mean, as the code forms it: `K⁻¹ • Σ x_i`. -/
noncomputable def mean (x : ι → V) : V := ((Fintype.card ι : ℝ))⁻¹ • ∑ i, x i

/-- The deviations from the mean sum to zero - the fact that makes the ANOVA decomposition work. -/
theorem sum_sub_mean_eq_zero [Nonempty ι] (x : ι → V) :
    (∑ i, (x i - mean x)) = 0 := by
  have hK : ((Fintype.card ι : ℝ)) ≠ 0 := by
    have := Fintype.card_ne_zero (α := ι); exact_mod_cast this
  rw [Finset.sum_sub_distrib]
  have h2 : (∑ _i : ι, mean x) = (Fintype.card ι : ℝ) • mean x := by
    rw [Finset.sum_const, Finset.card_univ]
    exact (Nat.cast_smul_eq_nsmul ℝ (Fintype.card ι) (mean x)).symm
  rw [h2]
  unfold mean
  rw [smul_smul, mul_inv_cancel₀ hK, one_smul]
  abel

end NeuronClosed

#print axioms NeuronClosed.sum_sub_mean_eq_zero
