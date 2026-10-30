"""Quotient-space comparisons and their matched null - the ONE implementation.

`to_v` maps a position list to its residual-space vector (`v_l = Σ dose · unit(W_down column)`), which is the object the
budget metric actually measures; `cos_direct` is the full direct-sum cosine over ALL layers (filtering to shared layers
was an earlier defect, and the zero it produced on disjoint support was tautological, not evidence); `null_for` draws
random lists matched on the SAME per-layer counts and on the list's own signed dose magnitudes, which is the only null
that makes the exceedance fraction interpretable.
"""
from __future__ import annotations

import math
import random

import torch


def to_v(items, Wn, layers):
    v = {l: torch.zeros(Wn[l].shape[0], dtype=torch.float64) for l in layers}
    for (l, n, d) in items:
        if 0 <= l < 32 and 0 <= n < Wn[l].shape[1]:
            v[l] += d * Wn[l][:, n]
    return v


def cos_direct(a_, b_, layers):
    num = sum(float(a_[l] @ b_[l]) for l in layers)
    da = sum(float(a_[l] @ a_[l]) for l in layers)
    db = sum(float(b_[l] @ b_[l]) for l in layers)
    if da <= 0 or db <= 0:
        return None
    return num / math.sqrt(da * db)


def null_for(items, ref_v, Wn, layers, n_null, seed=0):
    rng = random.Random(seed)
    doses = [abs(d) for (_, _, d) in items]
    signs = [1.0 if d > 0 else -1.0 for (_, _, d) in items]
    hist = {}
    for (l, _, _) in items:
        hist[l] = hist.get(l, 0) + 1
    vals = []
    for _ in range(n_null):
        rnd = []
        for l, k in hist.items():
            for n in rng.sample(range(Wn[l].shape[1]), min(k, Wn[l].shape[1])):
                rnd.append((l, n, rng.choice(signs) * rng.choice(doses)))
        c = cos_direct(to_v(rnd, Wn, layers), ref_v, layers)
        if c is not None:
            vals.append(c)
    vals.sort()
    return {'n': len(vals), 'mean': sum(vals) / len(vals) if vals else None,
            'p95': vals[int(0.95 * len(vals)) - 1] if vals else None, 'max': vals[-1] if vals else None}
