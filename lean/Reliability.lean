/-
Reliability as a variance component, and why the plug-in cosine is the wrong estimator.

CONTEXT. Plans 2.42-2.46 estimated the reliability of a score-function gradient by the cosine between two
half-sums of per-rollout score vectors. That estimator moved from r = 0.0685 to r = 0.2704 when the same
gradients were computed in a batch instead of one sequence at a time, although every individual vector agreed
to cos = 0.999. The reason is algebraic, not numerical luck:

  the half-sum cosine writes the signal as  ||S||^2 - sum_i ||x_i||^2,  a difference of two quantities of
  the same order that leaves ~10% behind, while ANY common-mode perturbation c (a bias shared by every
  rollout of a batch) adds to ||S||^2 without adding to the within-batch spread. A quantity whose numerator
  is a cancelled difference amplifies that bias; nothing about precision fixes it.

The estimator that does not have this defect is the classical intraclass correlation built from sums of
squares, because the within-group sum of squares only ever uses DIFFERENCES of observations:

  SSw = sum_i || m * x_i - sum_j x_j ||^2          (m = card, i.e. ||x_i - xbar||^2 scaled by m^2)
  SSb = sum_g m_g * || m_g * xbar_g - sum_j x_j ... ||  (not formalised here)

What is proved below:
  R1  `ssw_shift_invariant` : SSw is EXACTLY invariant under adding one common vector to every observation.
                              This is the property the half-sum cosine lacks, and it is what makes the new
                              estimator insensitive to batch-level (common-mode) bias.
  R2  `icc_bounds`          : the ICC built from MSw, MSb lies in [-1/(n-1), 1] whenever MSb >= 0 < MSw.
                              Hence a perturbation of the variance components cannot move the reliability
                              estimate off the real line or past ±1: the map is bounded, unlike a ratio of
                              half-sums whose denominator can vanish as the signal cancels.

Not claimed here: any probabilistic statement (E[MSw] = sigma^2 etc. needs a model), and nothing about the
model or the construct. R1 and R2 are the algebra that the estimation procedure relies on.
-/
import Mathlib

open scoped BigOperators

namespace NeuronClosed

variable {ι : Type*} [Fintype ι]
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Within-group sum of squares, scaled by `card^2` so that no division appears:
`SSw x = Σ_i ‖card • x_i − Σ_j x_j‖²`, which is `card^2` times `Σ_i ‖x_i − x̄‖²`. -/
noncomputable def SSw (x : ι → V) : ℝ :=
  ∑ i, ‖(Fintype.card ι : ℝ) • x i - ∑ j, x j‖ ^ 2

/-- (R1) The within-group sum of squares is invariant under adding ONE common vector to every observation.

This is exactly the robustness property the half-sum cosine lacks: a batch-level bias (the same perturbation
added to every rollout computed in one batch) drops out of `SSw`, whereas it enters `‖S‖² − Σ‖x_i‖²` — the
numerator of the plug-in cosine — at first order. -/
theorem ssw_shift_invariant (x : ι → V) (c : V) :
    SSw (fun i => x i + c) = SSw x := by
  unfold SSw
  have hsum : (∑ j, (x j + c)) = (∑ j, x j) + (Fintype.card ι : ℝ) • c := by
    rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ]
    exact congrArg (fun z => (∑ j, x j) + z) (Nat.cast_smul_eq_nsmul ℝ (Fintype.card ι) c).symm
  simp only [hsum, smul_add]
  congr 1
  funext i
  congr 1
  abel

/-- (R2) The intraclass correlation is bounded: with variance components `MSb ≥ 0 < MSw` and at least two
replicates per group, `ICC = (MSb − MSw)/(MSb + (n−1)·MSw)` lies in `[−1/(n−1), 1]`. A perturbation of the
components therefore moves the estimate inside a fixed range, instead of through a ratio whose denominator can
collapse as the signal cancels. -/
theorem icc_bounds (n : ℕ) (hn : 2 ≤ n) (MSb MSw : ℝ) (hb : 0 ≤ MSb) (hw : 0 < MSw) :
    -(1 / ((n : ℝ) - 1)) ≤ (MSb - MSw) / (MSb + ((n : ℝ) - 1) * MSw) ∧
    (MSb - MSw) / (MSb + ((n : ℝ) - 1) * MSw) ≤ 1 := by
  have hnR : (2 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hpos : 0 < (n : ℝ) - 1 := by linarith
  have hfac : (0 : ℝ) < (n : ℝ) - 1 := by linarith
  have hprod : 0 < ((n : ℝ) - 1) * MSw := mul_pos hfac hw
  have hden : 0 < MSb + ((n : ℝ) - 1) * MSw := by linarith
  constructor
  · rw [le_div_iff₀ hden]
    have hsplit : -(1 / ((n : ℝ) - 1)) * (MSb + ((n : ℝ) - 1) * MSw)
        = -(MSb / ((n : ℝ) - 1)) - MSw := by
      field_simp
      ring
    rw [hsplit]
    have h0 : 0 ≤ MSb / ((n : ℝ) - 1) := div_nonneg hb (le_of_lt hpos)
    linarith
  · rw [div_le_one hden]
    linarith

end NeuronClosed

#print axioms NeuronClosed.ssw_shift_invariant
#print axioms NeuronClosed.icc_bounds

