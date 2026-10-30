#!/usr/bin/env python3
"""The ONE correct way to read dPhi/d(mlp down_proj input) at every layer.

WHY THIS MODULE EXISTS. This readout was implemented three times (p_a_reinforce.py, p_a2_pathj2.py,
noise_scale.py) and all three copies carried the same bug:

    x = detach-then-clone-then-requires_grad(x)      # WRONG (pseudo-code; see below)
    return (x,) + args[1:]

Detaching at layer L cuts real causal paths, because `x_L = act(h_L)` and `h_L` ALSO carries the residual
stream forward into every later layer's MLP. The true `dPhi/dx_L` therefore includes the whole family
`x_L -> h_L -> mlp_{L+1}, mlp_{L+2}, ...`, and detach removes all of it.

How it was settled (fp32 finite differences, Qwen3.5-4B, teacher-forced slot log-prob):

    L   neuron   FD(1e-3)   FD(1e-2)   zero-leaf   detach
    0     1368    21.8353    19.6358     19.5363   21.4462
   13     2047    38.5303    38.6562     38.8417   30.7590
   13     1710   -36.8214   -35.9480    -36.1571  -24.9077
   31     1582   100.5421   100.4566    100.4998  100.4998
   31      692    77.3430    75.8030     77.4096   77.4096

The zero-leaf method matches FD at every layer; detach is 20-30% off in the middle and exact only at the
last layer, which is exactly the signature of a cut residual-carrier path. bf16 finite differences are
useless for this check (precision noise swamps eps=1e-3), hence fp32.

CORRECT METHOD (what the repo's frozen vendor engine already did). Keep the graph intact and add a ZERO
leaf, then read that leaf's gradient:

    delta = torch.zeros_like(x, requires_grad=True)
    return (x + delta,)+ args[1:]

The forward value is unchanged (delta == 0), so this is a pure gradient readout, not a perturbation.

Contract:
  * `Capture(lm, layers, p50, norms)` used as a context manager.
  * `.grad_sum()`        -> {layer: float64 CPU tensor of dPhi/d(act), summed over batch+token dims}
  * `.dose_grad()`       -> float64 CPU vector over the concatenated support, already multiplied by
                            p50_l / ||W_col_l,n|| (the dose unit used everywhere in this repo)
  * `.dim`               -> total coordinate count
Nothing else. Any script that needs a gradient w.r.t. an injection must go through this module.
"""
from __future__ import annotations
import torch


class Capture:
    def __init__(self, lm, layers, p50=None, norms=None):
        self.lm = lm
        self.layers = list(layers)
        self.p50 = p50
        self.norms = norms
        self.delta: dict[int, torch.Tensor] = {}
        self._h = []
        self.dim = sum(int(lm.layers[l].mlp.down_proj.weight.shape[1]) for l in self.layers)

    def __enter__(self):
        for l in self.layers:
            def build(idx):
                def hook(_m, args):
                    x = args[0]
                    if idx not in self.delta:
                        # zero leaf: forward unchanged, gradient readout exact
                        self.delta[idx] = torch.zeros_like(x, requires_grad=True)
                    return (x + self.delta[idx],) + tuple(args[1:])
                return hook
            self._h.append(self.lm.layers[l].mlp.down_proj.register_forward_pre_hook(build(l)))
        return self

    def grad_sum(self) -> dict[int, torch.Tensor]:
        out = {}
        for l in self.layers:
            d = self.delta[l]
            g = d.grad
            if g is None:
                out[l] = torch.zeros(self.lm.layers[l].mlp.down_proj.weight.shape[1], dtype=torch.float64)
            else:
                out[l] = g.sum(dim=tuple(range(g.dim() - 1))).double().cpu()
            d.grad = None
        return out

    def grad_per_sample(self) -> dict[int, torch.Tensor]:
        """Per-sample gradients from ONE batched backward.

        Each element of the delta leaf belongs to exactly one sample of the batch, so `delta.grad` already carries
        per-sample derivatives - it only needs summing over the time axis. This is what makes a batched rollout pass
        possible: 8 backwards for 8 contexts instead of 256 backwards for 256 rollouts, i.e. the GPU actually runs.
        """
        out = {}
        for l in self.layers:
            d = self.delta[l]
            g = d.grad
            if g is None:
                out[l] = None
                continue
            out[l] = g.sum(dim=1).double()      # (B, d): sum over the time axis, keep the sample axis
            d.grad = None
        return out

    def dose_grad(self, p50=None, norms=None) -> torch.Tensor:
        p50 = self.p50 if p50 is None else p50
        norms = self.norms if norms is None else norms
        if p50 is None or norms is None:
            raise ValueError('dose_grad needs p50 and norms (pass them to Capture or here)')
        raw = self.grad_sum()
        out = torch.zeros(self.dim, dtype=torch.float64)
        off = 0
        for l in self.layers:
            d = int(self.lm.layers[l].mlp.down_proj.weight.shape[1])
            scale = (p50[l] / norms[l].clamp(min=1e-12)).double()
            out[off:off + d] = raw[l] * scale
            off += d
        return out

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


def fd_check(lm, phi, coord, eps, dev='cuda:0', p50=None, norms=None):
    """Central finite difference of `phi(doses)` w.r.t. one coordinate, in dose units.

    `phi` must accept {(layer, neuron): dose} and return a float. Provided here so every calibration
    run uses the same definition of "the thing whose gradient we think we compute".
    """
    (l, n) = coord
    w = float(lm.layers[l].mlp.down_proj.weight.shape[1])
    assert 0 <= n < w
    hi = phi({coord: eps})
    lo = phi({coord: -eps})
    return (hi - lo) / (2 * eps)


def column_norms_dev(lm, layers, dev):
    """||W_down[:, n]|| for every layer, computed ON THE GPU. The vendor helper does this on the CPU
    (`.detach().float().norm(dim=0).cpu()`), which is a 2560x9216 reduction per layer - tens of seconds of CPU
    while the GPU idles, and it is only needed as a scale factor."""
    out = {}
    for l in layers:
        W = lm.layers[l].mlp.down_proj.weight.detach().to(dev).float()
        out[l] = W.norm(dim=0).cpu()
        del W
    return out
