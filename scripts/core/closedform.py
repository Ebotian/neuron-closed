"""The closed form - the ONE implementation of the budget-constrained optimum.

`max d'u  s.t.  u'Mu <= b²` with `M = Wn'Wn` (Gram of the UNIT W_down columns). Two facts that make this file non-trivial:

  * M is SINGULAR on the full support: rank ≤ 2560 while a layer has 9216 coordinates, so Cholesky fails at leading
    minor 2559 and the solution is the MIN-NORM one, via the thin SVD of Wn (V S^-2 V' d). That is parameter-free:
    it is the smallest injection achieving the maximum.
  * the kernel is a GAUGE (kernel directions cost nothing and gain nothing), so the optimum lives in the quotient
    space; lean/Gauge.lean proves the identity that this amounts to recovering the undistorted residual gradient.
"""
from __future__ import annotations

from pathlib import Path

import torch


def build_gram(lm, layers, dev, cache_path=None):
    """Gram of the unit W_down columns per layer, on the CPU. Cached: 32 layers cost ~20 s and never change."""
    if cache_path is not None and Path(cache_path).exists():
        return torch.load(cache_path)
    G = {}
    for l in layers:
        W = lm.layers[l].mlp.down_proj.weight.detach().double().to(dev)
        W = W / W.norm(dim=0, keepdim=True).clamp(min=1e-12)
        G[l] = (W.T @ W).cpu()
        del W
    if cache_path is not None:
        torch.save(G, cache_path)
    return G


def solve_min_norm(lm, layers, d, dev):
    """Minimum-norm solution of M_l u = d_l; returns (x per layer, q = d'x, effective rank per layer)."""
    x, q, ranks = {}, 0.0, []
    for l in layers:
        W = lm.layers[l].mlp.down_proj.weight.detach().float().to(dev)
        Wn = W / W.norm(dim=0, keepdim=True).clamp(min=1e-12)
        _, S_, Vh = torch.linalg.svd(Wn, full_matrices=False)
        keep = S_ > 1e-6 * S_.max()
        ranks.append(int(keep.sum()))
        gl = d[l].to(dev).float()
        proj = Vh @ gl
        xl = Vh.T @ torch.where(keep, proj / S_.pow(2), torch.zeros_like(proj))
        x[l] = xl.cpu().double()
        q += float(gl.double() @ xl.double())
        del W, Wn
    return {l: x[l] / q ** 0.5 for l in layers}, q, ranks


def budget2(G, layers, u, dev):
    tot = 0.0
    for l in layers:
        ul = u[l].to(dev).float()
        tot += float(ul.double() @ (G[l].to(dev) @ ul).double())
    return tot


def cos_M(G, layers, u1, u2, dev):
    """The metric-correct angle: <u1,u2>_M / (||u1||_M ||u2||_M)."""
    cross = 0.0
    for l in layers:
        v = u2[l].to(dev).float()
        cross += float(u1[l].to(dev).float().double() @ (G[l].to(dev) @ v).double())
    return cross / (budget2(G, layers, u1, dev) * budget2(G, layers, u2, dev)) ** 0.5


def top(d, layers, K):
    """The K largest |dose| coordinates, as a set of (layer, neuron)."""
    flat = torch.cat([d[l] for l in layers])
    idx = torch.topk(flat.abs(), K).indices.tolist()
    out, off = set(), 0
    for l in layers:
        n = d[l].shape[0]
        for i in idx:
            if off <= i < off + n:
                out.add((l, i - off))
        off += n
    return out
