/- R1: the within-group sum of squares is invariant under a common shift.

Split out of Reliability.lean so that each theorem is its own compilation unit: editing one cannot break
another, which is exactly what went wrong when all five declarations lived in one file and I edited it by
string slicing.
-/
import Mathlib

open scoped BigOperators

namespace NeuronClosed

variable {ι : Type*} [Fintype ι]
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- `SSw x = Σ_i ‖card • x_i − Σ_j x_j‖²`, i.e. `card²` times `Σ_i ‖x_i − x̄‖²`. -/
noncomputable def SSw (x : ι → V) : ℝ :=
  ∑ i, ‖(Fintype.card ι : ℝ) • x i - ∑ j, x j‖ ^ 2

/-- (R1) Adding ONE common vector to every observation leaves `SSw` unchanged. This is the property the
half-sum cosine lacks: a batch-level bias drops out of `SSw` exactly. -/
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

end NeuronClosed

#print axioms NeuronClosed.ssw_shift_invariant
