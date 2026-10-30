/- R5: the NEGATIVE half of the argument — the plug-in cosine of two half-sums moves when one common vector
is added to both, while SSw does not (R1). Independent compilation unit. -/
import Mathlib

open scoped BigOperators InnerProductSpace

namespace NeuronClosed

/-- (R5) With `u ⟂ w`, `‖u‖ = ‖w‖ = 1`, the squared cosine of `(u, w)` is `0`, while the squared cosine of
`(u+u, w+u)` — the same pair after adding the common vector `c = u` to both — is `1/2`. Since `SSw` of the
pair is unchanged (R1), this is a concrete witness that the half-sum cosine is not a function of the
bias-immune quantities. -/
theorem plug_in_cos_moves {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (u w : V) (hu : ‖u‖ = 1) (hw : ‖w‖ = 1) (horth : ⟪u, w⟫_ℝ = 0) :
    (⟪u, w⟫_ℝ ^ 2 / (‖u‖ ^ 2 * ‖w‖ ^ 2) = 0) ∧
      (⟪u + u, w + u⟫_ℝ ^ 2 / (‖(u + u : V)‖ ^ 2 * ‖(w + u : V)‖ ^ 2) = 1 / 2) := by
  constructor
  · rw [horth]; simp
  · have huu : ⟪u + u, w + u⟫_ℝ = 2 := by
      simp only [inner_add_left, inner_add_right]
      rw [horth, real_inner_self_eq_norm_sq, hu]
      norm_num
    have h1 : ‖(u + u : V)‖ ^ 2 = 4 := by
      simp only [norm_add_sq_real, real_inner_self_eq_norm_sq, hu]
      norm_num
    have h2 : ‖(w + u : V)‖ ^ 2 = 2 := by
      have hsym : ⟪w, u⟫_ℝ = 0 := by rw [real_inner_comm]; exact horth
      simp only [norm_add_sq_real, hsym, real_inner_self_eq_norm_sq, hu, hw]
      norm_num
    rw [huu, h1, h2]
    norm_num

end NeuronClosed

#print axioms NeuronClosed.plug_in_cos_moves
