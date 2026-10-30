/-
Plan 2.42's dimensional claim, as an identity.

What we measured: the split-half cosine of the accumulated score gradient is FLAT in K
(r = 0.002, 0.005, 0.016, 0.015, 0.022 for K = 2, 4, 8, 16, 32), so more rollouts buy nothing.
What is formalised here is the part that needs no probability at all:

  for a PAIRWISE ORTHOGONAL family x_1..x_K with equal norms c, the split-half inner product is
     4 * ⟪Σ_A x, Σ_B x⟫ = ‖Σ_i x_i‖² − K * c²
  for EVERY partition of the indices into two halves. Hence the split-half cosine
     cos = (‖S‖² − K c²) / (‖S‖² + K c²)          (S = Σ x_i)
  is split-INDEPENDENT: with orthogonal per-rollout vectors there is nothing for the split to vary.

CORRECTION (2026-10-30, after doc/references/isotropy-and-noise-scale.md). An earlier version of this
header explained the flat curve by "isotropic score vectors, so ‖S‖² ≈ K c², so cos ≈ 0 at every K,
hence K ≈ d·v/s²". That explanation is WRONG as a description of these models: representation and
gradient anisotropy is measured and described as inherent (arXiv:2401.12143), with a spike occupying
about 1.5% of directions while dominating second-order statistics (arXiv:2602.11185), and gradient
subspaces are low-rank (arXiv:2407.11239, Aghajanyan et al. 2021). The correct general quantity is the
noise-to-signal ratio, i.e. McCandlish et al.'s gradient noise scale
     cos ≈ ‖g‖² / (‖g‖² + tr(Σ)/K),     crossover  K ≈ tr(Σ)/‖g‖²,
of which the isotropic case tr(Σ) = d·v is only the special case. So the flat curve should be read as
"tr(Σ) is not being averaged down over K rollouts", not as "K ~ d".
Neither statement is proved here: the theorems below are the algebra, the anisotropy/noise-scale part
is an assumption plus arithmetic, and it is measurable on our own stored score vectors.
-/
import Mathlib

open scoped BigOperators InnerProductSpace RealInnerProductSpace

namespace NeuronClosed

variable {ι : Type*} [Fintype ι]
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- (E1) Orthogonality kills the cross terms: the squared norm of a sum of pairwise orthogonal
vectors is the sum of the squared norms. -/
theorem norm_sq_sum_of_pairwise_orthogonal (x : ι → V)
    (h : ∀ i j, i ≠ j → ⟪x i, x j⟫_ℝ = 0) :
    ‖(∑ i, x i)‖ ^ 2 = ∑ i, ‖x i‖ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, sum_inner]
  simp only [inner_sum]
  have hpt : ∀ i, (∑ j, ⟪x i, x j⟫_ℝ) = ‖x i‖ ^ 2 := by
    intro i
    rw [Finset.sum_eq_single i]
    · exact real_inner_self_eq_norm_sq (x i)
    · intro j _ hji; exact h i j (Ne.symm hji)
    · intro hi; exact absurd (Finset.mem_univ i) hi
  rw [Finset.sum_congr rfl (fun i _ => hpt i)]

/-- (E2) The same for a signed family: with `ε` arbitrary, the alternating sum's norm depends only on
`Σ ε_i² ‖x_i‖²`, never on the *pattern* of signs. This is what makes the split-half quantity
split-independent. -/
theorem norm_sq_sum_smul_of_pairwise_orthogonal (x : ι → V) (ε : ι → ℝ)
    (h : ∀ i j, i ≠ j → ⟪x i, x j⟫_ℝ = 0) :
    ‖(∑ i, ε i • x i)‖ ^ 2 = ∑ i, (ε i) ^ 2 * ‖x i‖ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, sum_inner]
  simp only [inner_sum, real_inner_smul_left, real_inner_smul_right]
  have hpt : ∀ i, (∑ j, ε j * (ε i * ⟪x i, x j⟫_ℝ)) = (ε i) ^ 2 * ‖x i‖ ^ 2 := by
    intro i
    rw [Finset.sum_eq_single i]
    · rw [real_inner_self_eq_norm_sq]; ring
    · intro j _ hji; rw [h i j (Ne.symm hji)]; ring
    · intro hi; exact absurd (Finset.mem_univ i) hi
  rw [Finset.sum_congr rfl (fun i _ => hpt i)]

/-- (E3) The pure algebra of splitting a sum in two halves. -/
theorem four_mul_inner_split (SA SB : V) :
    4 * ⟪SA, SB⟫_ℝ = ‖SA + SB‖ ^ 2 - ‖SA - SB‖ ^ 2 := by
  rw [← real_inner_self_eq_norm_sq, ← real_inner_self_eq_norm_sq]
  simp only [inner_add_left, inner_add_right, inner_sub_left, inner_sub_right]
  rw [real_inner_comm SB SA]
  ring

/-- (E4) With signs `ε = ±1`, the alternating sum has squared norm `K * c²` for EVERY sign pattern:
the split-half quantity cannot depend on the split. This is the exact statement behind the flat `r(K)`. -/
theorem norm_sq_alternating_eq_card_mul (x : ι → V) (ε : ι → ℝ) (c : ℝ)
    (hε : ∀ i, ε i = 1 ∨ ε i = -1)
    (h : ∀ i j, i ≠ j → ⟪x i, x j⟫_ℝ = 0) (hc : ∀ i, ‖x i‖ = c) :
    ‖(∑ i, ε i • x i)‖ ^ 2 = (Fintype.card ι : ℝ) * c ^ 2 := by
  rw [norm_sq_sum_smul_of_pairwise_orthogonal x ε h]
  have hsq : ∀ i, (ε i) ^ 2 = 1 := by
    intro i; rcases hε i with h1 | h1 <;> rw [h1] <;> norm_num
  have hterm : ∀ i, (ε i) ^ 2 * ‖x i‖ ^ 2 = c ^ 2 := by
    intro i
    rw [hsq i, one_mul, hc i]
  rw [Finset.sum_congr rfl (fun i _ => hterm i), Finset.sum_const, Finset.card_univ, nsmul_eq_mul]

/-- (E5) **The main statement.** If a pair of half-sums has `SA + SB = Σ x` and `SA − SB = Σ ε • x`
with `ε = ±1`, then its split-half inner product is `(‖Σ x‖² − K c²)/4` — split-independent. -/
theorem four_mul_inner_of_alternating (x : ι → V) (ε : ι → ℝ) (SA SB : V) (c : ℝ)
    (hε : ∀ i, ε i = 1 ∨ ε i = -1)
    (h : ∀ i j, i ≠ j → ⟪x i, x j⟫_ℝ = 0) (hc : ∀ i, ‖x i‖ = c)
    (hplus : SA + SB = ∑ i, x i) (hminus : SA - SB = ∑ i, ε i • x i) :
    4 * ⟪SA, SB⟫_ℝ = ‖(∑ i, x i)‖ ^ 2 - (Fintype.card ι : ℝ) * c ^ 2 := by
  rw [four_mul_inner_split SA SB]
  rw [hplus, hminus, norm_sq_alternating_eq_card_mul x ε c hε h hc]

end NeuronClosed

#print axioms NeuronClosed.norm_sq_sum_of_pairwise_orthogonal
#print axioms NeuronClosed.norm_sq_sum_smul_of_pairwise_orthogonal
#print axioms NeuronClosed.four_mul_inner_split
#print axioms NeuronClosed.norm_sq_alternating_eq_card_mul
#print axioms NeuronClosed.four_mul_inner_of_alternating
