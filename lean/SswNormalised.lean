/- R3: the code's NORMALISED within-group sum of squares is invariant under a common shift.

This is the form `scripts/reliability.py` actually computes (`icc_from_gram`):
    ss_w = Σ_i ‖x_i‖² − ‖Σ_i x_i‖²/K
with K the group size. Ssw.lean proves the K²-scaled variant; this file closes the gap, in its own
compilation unit so that failing experiments cannot break the other theorems.
-/
import Mathlib

open scoped BigOperators InnerProductSpace

namespace NeuronClosed

variable {ι : Type*} [Fintype ι] [Nonempty ι]
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- The quantity the code computes for one group. -/
noncomputable def sswU (x : ι → V) : ℝ :=
  ∑ i, ‖x i‖ ^ 2 - ‖(∑ i, x i)‖ ^ 2 / (Fintype.card ι : ℝ)

/-- The cross-term expansion, isolated: `Σ_i ‖x_i + c‖²` in terms of `Σ‖x_i‖²`, `⟪Σx_i, c⟫`, `‖c‖²`. -/
theorem sum_norm_add_sq (x : ι → V) (c : V) :
    ∑ i, ‖x i + c‖ ^ 2
      = (∑ i, ‖x i‖ ^ 2) + 2 * ⟪(∑ i, x i), c⟫_ℝ + (Fintype.card ι : ℝ) * ‖c‖ ^ 2 := by
  have hterm : ∀ i, ‖x i + c‖ ^ 2 = ‖x i‖ ^ 2 + 2 * ⟪x i, c⟫_ℝ + ‖c‖ ^ 2 := by
    intro i; rw [norm_add_sq_real]
  rw [Finset.sum_congr rfl (fun i _ => hterm i), Finset.sum_add_distrib, Finset.sum_add_distrib]
  have h1 : (∑ i, 2 * ⟪x i, c⟫_ℝ) = 2 * ⟪(∑ i, x i), c⟫_ℝ := by
    rw [← Finset.mul_sum, ← sum_inner]
  have h2 : (∑ _i : ι, ‖c‖ ^ 2) = (Fintype.card ι : ℝ) * ‖c‖ ^ 2 := by
    rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
  rw [h1, h2]

/-- (R3) Adding one common vector to every observation leaves the code's normalised quantity unchanged. -/
theorem sswu_shift_invariant (x : ι → V) (c : V) :
    sswU (fun i => x i + c) = sswU x := by
  unfold sswU
  have hK : (0 : ℝ) < (Fintype.card ι : ℝ) := by exact_mod_cast Fintype.card_pos
  have hKne : ((Fintype.card ι : ℝ)) ≠ 0 := ne_of_gt hK
  have hsum : (∑ i, (x i + c)) = (∑ i, x i) + (Fintype.card ι : ℝ) • c := by
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ]
    exact congrArg (fun z => (∑ i, x i) + z) (Nat.cast_smul_eq_nsmul ℝ (Fintype.card ι) c).symm
  -- the shifted norm expands into the same cross term plus K²‖c‖²
  have hexp : ‖(∑ i, x i) + (Fintype.card ι : ℝ) • c‖ ^ 2
      = ‖(∑ i, x i)‖ ^ 2
        + 2 * (Fintype.card ι : ℝ) * ⟪(∑ i, x i), c⟫_ℝ
        + (Fintype.card ι : ℝ) ^ 2 * ‖c‖ ^ 2 := by
    rw [norm_add_sq_real, real_inner_smul_right, norm_smul, Real.norm_eq_abs, abs_of_pos hK]
    ring
  rw [hsum, sum_norm_add_sq, hexp]
  field_simp
  ring

end NeuronClosed

#print axioms NeuronClosed.sum_norm_add_sq
#print axioms NeuronClosed.sswu_shift_invariant
