/-
Plan 2.42's decomposition, as a machine-checked identity.

We measured two terms of the derivative of an expectation over a *sampled* trajectory and could not
estimate one of them. The identity that splits them is standard (pathwise/score-function, a.k.a.
reparameterization vs REINFORCE). What is formalised here, for a FINITE outcome set `Y` (so that the
sums and derivatives are honest and nothing needs measure theory):

  D1 `expectation_deriv_decomp`
        d/du Σ_y p(u,y)·φ(u,y) = Σ_y p(u,y)·∂_u φ(u,y) + Σ_y p(u,y)·φ(u,y)·∂_u log p(u,y)
     i.e. the derivative of the expectation is the expectation of the pathwise derivative PLUS the
     score term. Plan 2.42 measured only the second term.

  D2 `sum_p_score_eq_zero`
        Σ_y p(u,y)·∂_u log p(u,y) = 0     (as long as Σ_y p(u,·) ≡ 1)
     the mean score vanishes: this is why a baseline is a variance-reduction device and not a change
     of the objective.

  D3 `baseline_free`
        Σ_y p·(φ − b)·∂_u log p = Σ_y p·φ·∂_u log p   for every `b`
     the baseline drops out of the expectation exactly (it does NOT drop out of an estimator at finite
     sample size, which is the whole point of using it).

Scope: this is the algebra of the decomposition. It says nothing about whether `φ` is the right
construct, and nothing about the model. `Y` is finite on purpose: on a finite set everything here is
an ordinary finite sum and `Real.log`, so no measure theory, no probability spaces.
-/
import Mathlib

open scoped BigOperators

namespace NeuronClosed

variable {Y : Type*} [Fintype Y]

/-- (D1) The decomposition. `p` is the path density and `φ` the functional; both may depend on `u`. -/
theorem expectation_deriv_decomp (p φ : ℝ → Y → ℝ) (u : ℝ)
    (hp : ∀ y, HasDerivAt (fun v => p v y) (deriv (fun v => p v y) u) u)
    (hpne : ∀ y, p u y ≠ 0)
    (hφ : ∀ y, HasDerivAt (fun v => φ v y) (deriv (fun v => φ v y) u) u) :
    deriv (fun v => ∑ y, p v y * φ v y) u
      = ∑ y, (p u y * deriv (fun v => φ v y) u
              + p u y * φ u y * deriv (fun v => Real.log (p v y)) u) := by
  have hdiff : HasDerivAt (fun v => ∑ y, p v y * φ v y)
      (∑ y, (deriv (fun v => p v y) u * φ u y + p u y * deriv (fun v => φ v y) u)) u := by
    have hS : HasDerivAt (∑ i, (fun v => p v i) * (fun v => φ v i))
        (∑ i, (deriv (fun v => p v i) u * φ u i + p u i * deriv (fun v => φ v i) u)) u :=
      HasDerivAt.sum (u := Finset.univ) (fun y (_ : y ∈ Finset.univ) => (hp y).mul (hφ y))
    rw [show (∑ i, (fun v => p v i) * (fun v => φ v i)) = (fun v => ∑ i, p v i * φ v i) by
      ext v; simp] at hS
    exact hS
  rw [hdiff.deriv]
  refine Finset.sum_congr rfl fun y _ => ?_
  rw [((hp y).log (hpne y)).deriv]
  field_simp [hpne y]
  ring_nf

/-- (D2) The mean score vanishes when the density is normalised for every `u`. -/
theorem sum_p_score_eq_zero (p : ℝ → Y → ℝ) (u : ℝ)
    (hp : ∀ y, HasDerivAt (fun v => p v y) (deriv (fun v => p v y) u) u)
    (hpne : ∀ y, p u y ≠ 0)
    (hconst : ∀ v, ∑ y, p v y = 1) :
    ∑ y, p u y * deriv (fun v => Real.log (p v y)) u = 0 := by
  have hderiv : HasDerivAt (fun v => ∑ y, p v y)
      (∑ y, deriv (fun v => p v y) u) u := by
    have hS : HasDerivAt (∑ i, (fun v => p v i)) (∑ i, deriv (fun v => p v i) u) u :=
      HasDerivAt.sum (u := Finset.univ) (fun y (_ : y ∈ Finset.univ) => hp y)
    rw [show (∑ i, (fun v => p v i)) = (fun v => ∑ i, p v i) by ext v; simp] at hS
    exact hS
  have hzero : (∑ y, deriv (fun v => p v y) u) = 0 := by
    have h1 : deriv (fun v => ∑ y, p v y) u = 0 := by
      have : (fun v => ∑ y, p v y) = fun _ => (1 : ℝ) := by
        funext v; exact hconst v
      rw [this, deriv_const]
    rwa [hderiv.deriv] at h1
  calc ∑ y, p u y * deriv (fun v => Real.log (p v y)) u
      = ∑ y, deriv (fun v => p v y) u := by
        refine Finset.sum_congr rfl fun y _ => ?_
        rw [((hp y).log (hpne y)).deriv]
        field_simp [hpne y]
    _ = 0 := hzero

/-- (D3) The baseline is free in the expectation: it changes the estimator's variance, not the mean. -/
theorem baseline_free (p φ : ℝ → Y → ℝ) (u b : ℝ)
    (hp : ∀ y, HasDerivAt (fun v => p v y) (deriv (fun v => p v y) u) u)
    (hpne : ∀ y, p u y ≠ 0)
    (hconst : ∀ v, ∑ y, p v y = 1) :
    ∑ y, p u y * (φ u y - b) * deriv (fun v => Real.log (p v y)) u
      = ∑ y, p u y * φ u y * deriv (fun v => Real.log (p v y)) u := by
  have hz := sum_p_score_eq_zero (Y := Y) p u hp hpne hconst
  calc ∑ y, p u y * (φ u y - b) * deriv (fun v => Real.log (p v y)) u
      = ∑ y, (p u y * φ u y * deriv (fun v => Real.log (p v y)) u
              - b * (p u y * deriv (fun v => Real.log (p v y)) u)) := by
        refine Finset.sum_congr rfl fun y _ => ?_
        ring_nf
    _ = ∑ y, p u y * φ u y * deriv (fun v => Real.log (p v y)) u
          - b * ∑ y, p u y * deriv (fun v => Real.log (p v y)) u := by
        rw [Finset.sum_sub_distrib, Finset.mul_sum]
    _ = ∑ y, p u y * φ u y * deriv (fun v => Real.log (p v y)) u := by
        rw [hz, mul_zero, sub_zero]

end NeuronClosed

#print axioms NeuronClosed.expectation_deriv_decomp
#print axioms NeuronClosed.sum_p_score_eq_zero
#print axioms NeuronClosed.baseline_free
