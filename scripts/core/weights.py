"""The two baseline weightings - one implementation, in the lowest layer that can hold them.

WHY HERE. `analysis/report.py` needs the same leave-one-out weights that `estimators/score.py` produces, and analysis must not
import estimators (that would drag `mlp_capture` and torch into a module that only reads Grams and numbers). Putting the two
formulas in `core/` keeps the dependency direction clean and keeps ONE copy of each, which is what the noregress gate enforces.

THE FORMULA. `w_i = (phi_i - mean of the OTHER rollouts) / (K * n_ctx)`. Summed over the K rollouts of one context and then
over the C contexts this is an unbiased estimator of the score term `E[(phi - b) grad log p]`, because the baseline drops out
in expectation (lean/Decomposition.lean, `baseline_free`). The baseline is a free parameter in expectation - which is exactly
why writing the formula in four places was indefensible.
"""
from __future__ import annotations


def loo_weights(phi, n_ctx):
    """Leave-one-out baseline inside the group that is actually being averaged."""
    K, psum = len(phi), sum(phi)
    return [(x - (psum - x) / max(K - 1, 1)) / (K * n_ctx) for x in phi]


def insample_weights(phi, n_ctx):
    """In-sample baseline; kept as an explicit alternative so both can be reported side by side."""
    K, psum = len(phi), sum(phi)
    return [(x - psum / K) / (K * n_ctx) for x in phi]
