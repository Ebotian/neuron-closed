/-
Plan 2.44's algebra, as machine-checked statements.

Setting. Coordinates live in `E` (9216 per layer), the residual stream in `F` (2560 per layer).
`W : E →L[ℝ] F` is the down-projection, and the physical budget metric is `M := W† ∘L W`
(the Gram matrix of the unit `W_down` columns). Because `dim E > dim F`, `M` is singular.

What is proved here (no sorry, no axioms beyond the three standard ones):
  * `ker_gram`        : `ker M = ker W`  -- the singularity is the kernel of the projection.
  * `gauge_invariance`: adding a kernel direction does not change the perturbation `W ·`.
  * `lift_maps_back`  : if `W` is surjective and `M u = W† y`, then `W u = y`.
                        This is the algebraic content of the numerical `cos(W u_M, ∇Φ) = 1.0000`.
-/
import Mathlib

open scoped InnerProduct RealInnerProductSpace

namespace NeuronClosed

variable {E F : Type*}
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]
  [NormedAddCommGroup F] [InnerProductSpace ℝ F] [CompleteSpace F]

/-- (S1) The budget metric has exactly the kernel of the projection. Mathlib proves this for
`ContinuousLinearMap` (`ContinuousLinearMap.ker_adjoint_comp_self`); restated as the object we use. -/
theorem ker_gram (W : E →L[ℝ] F) : (W.adjoint.comp W).ker = W.ker :=
  W.ker_adjoint_comp_self

/-- (S3) Gauge invariance: a kernel direction costs nothing and changes nothing. -/
theorem gauge_invariance (W : E →L[ℝ] F) (u u₀ : E) (h : u₀ ∈ (W.adjoint.comp W).ker) :
    W (u + u₀) = W u := by
  have h₀ : W u₀ = 0 := by
    rw [ker_gram W] at h
    simpa using h
  rw [map_add, h₀, add_zero]

/-- (S2) The lift maps back. For a surjective `W` (our case: 2560 <= 9216 with full row rank),
any solution of `M u = W† y` has residual image exactly `y` -- no distortion. -/
theorem lift_maps_back (W : E →L[ℝ] F) (hW : Function.Surjective W) (u : E) (y : F)
    (h : (W.adjoint.comp W) u = W.adjoint y) : W u = y := by
  have hkey : ∀ v : E, ⟪W v, (W u - y)⟫ = 0 := by
    intro v
    have hv : ⟪W v, W u⟫ = ⟪W v, y⟫ := by
      have h' := congrArg (fun z : E => (⟪v, z⟫ : ℝ)) h
      simpa [ContinuousLinearMap.comp_apply, ContinuousLinearMap.adjoint_inner_right] using h'
    rw [inner_sub_right, hv, sub_self]
  obtain ⟨v, hv⟩ := hW (W u - y)
  have h0 : ⟪(W u - y), (W u - y)⟫ = 0 := by
    have h1 := hkey v
    simp only [hv] at h1
    simpa using h1
  rw [inner_self_eq_zero] at h0
  exact sub_eq_zero.mp h0

end NeuronClosed

/- Axiom audit: each must print only [propext, Classical.choice, Quot.sound]. -/
#print axioms NeuronClosed.ker_gram
#print axioms NeuronClosed.gauge_invariance
#print axioms NeuronClosed.lift_maps_back
