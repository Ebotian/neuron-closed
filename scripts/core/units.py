"""The ONLY place where the dose unit is formed.

WHY THIS MODULE EXISTS. The ratio `p50_l / ‖W_down[:, n]‖` - the conversion between a dose and a physical
residual perturbation - was written out inline in thirteen scripts. That is how the same functional acquired
three different normalisations and how the "double division by |slot|" bug survived two audit rounds: with one
owner there is one expression to check, and a caller that needs a different scaling has to say so explicitly.

INVARIANTS (checked by scripts/verify_units.py, per value, not by cosine):
  * `dose_scale` reproduces the inline expression `p50[l] / norms[l].clamp(min=1e-12)` BIT-FOR-BIT.
  * `column_norms_dev` computes ‖W_col‖ on the GPU; the vendor helper computes it on the CPU. The two agree to
    float32 rounding, and the difference is REPORTED rather than assumed away, because it enters every budget.
"""
from __future__ import annotations

import torch


def column_norms_dev(lm, layers, dev):
    """‖W_down[:, n]‖ per layer, computed on the device (the CPU reduction was 32 × 2560×9216 of pure CPU work)."""
    out = {}
    for l in layers:
        W = lm.layers[l].mlp.down_proj.weight.detach().to(dev).float()
        out[l] = W.norm(dim=0).cpu()
        del W
    return out


def dose_scale(p50, norms, layers):
    """`p50_l / ‖W_col_l,n‖`, float64 on the CPU. The one expression; keep it here and nowhere else."""
    return {l: (p50[l] / norms[l].clamp(min=1e-12)).double() for l in layers}


def scaled(grad, scale, layers):
    """Multiply a raw per-layer gradient by the dose scale, moving the scale to the gradient's device."""
    return {l: grad[l] * scale[l].to(grad[l].device) for l in layers}
