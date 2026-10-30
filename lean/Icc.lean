/- R2 + R4: the intraclass correlation is bounded, and the hypothesis it needs is a THEOREM when MSb comes
from a between-group sum of squares (which is how scripts/reliability.py obtains it).

Split out of Reliability.lean. Independent compilation unit.
-/
import Mathlib

open scoped BigOperators

namespace NeuronClosed

/-- (R2) With `MSb ≥ 0 < MSw` and at least two replicates, the intraclass correlation
`(MSb − MSw)/(MSb + (n−1)·MSw)` lies in `[−1/(n−1), 1]`.

Note the lower side is `≤`, not `<`: at `MSb = 0` the ICC EQUALS `−1/(n−1)`. Lean refuted my first statement,
which had a strict bound. -/
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

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- (R4a) A between-group sum of squares with the code's meaning: `K · Σ_c ‖x̄_c − x̄‖²`. -/
noncomputable def ssbU {γ : Type*} [Fintype γ] (xbar : γ → V) (gbar : V) (k : ℝ) : ℝ :=
  k * ∑ c, ‖xbar c - gbar‖ ^ 2

/-- (R4b) It is nonnegative BY CONSTRUCTION, so the hypothesis `0 ≤ MSb` of `icc_bounds` never has to be
checked in code. -/
theorem ssbU_nonneg {γ : Type*} [Fintype γ] (xbar : γ → V) (gbar : V) {k : ℝ} (hk : 0 ≤ k) :
    0 ≤ ssbU xbar gbar k := by
  unfold ssbU
  exact mul_nonneg hk (Finset.sum_nonneg (fun c _ => sq_nonneg _))

/-- (R4c) Consequently the ICC built from `MSb = SS_b/(#groups)` and `MSw = ssw` is bounded, with `MSb ≥ 0`
discharged instead of assumed. -/
theorem icc_bounds_from_sums {γ : Type*} [Fintype γ] (n : ℕ) (hn : 2 ≤ n)
    (xbar : γ → V) (gbar : V) {k ssw : ℝ} (hk : 0 ≤ k) (hssw : 0 < ssw)
    (hC : 0 < Fintype.card γ) :
    -(1 / ((n : ℝ) - 1)) ≤
        (ssbU xbar gbar k / (Fintype.card γ : ℝ) - ssw) /
          (ssbU xbar gbar k / (Fintype.card γ : ℝ) + ((n : ℝ) - 1) * ssw) ∧
      (ssbU xbar gbar k / (Fintype.card γ : ℝ) - ssw) /
          (ssbU xbar gbar k / (Fintype.card γ : ℝ) + ((n : ℝ) - 1) * ssw) ≤ 1 := by
  have hcard : (0 : ℝ) < (Fintype.card γ : ℝ) := by exact_mod_cast hC
  have hb : 0 ≤ ssbU xbar gbar k / (Fintype.card γ : ℝ) :=
    div_nonneg (ssbU_nonneg xbar gbar hk) (le_of_lt hcard)
  exact icc_bounds n hn _ ssw hb hssw

end NeuronClosed

#print axioms NeuronClosed.icc_bounds
#print axioms NeuronClosed.ssbU_nonneg
#print axioms NeuronClosed.icc_bounds_from_sums
