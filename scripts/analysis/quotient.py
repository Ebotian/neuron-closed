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


def report_list(items, recipe_v, Wn, layers, L1, L2, n_null, seed=0):
    """Everything the comparison reports for ONE position list: per-recipe quantiles, the matched null, the L1/L2 strata."""
    v = to_v(items, Wn, layers)
    cs = [(c, tags) for (c, tags) in ((cos_direct(v, rv, layers), tags) for (_, rv, tags) in recipe_v) if c is not None]
    vals = sorted(c for c, _ in cs)
    nl = null_for(items, v, Wn, layers, n_null, seed)
    grp = {'L1_only': [], 'L2_only': [], 'other': []}
    for c, tags in cs:
        if tags & L1 and not (tags & L2):
            grp['L1_only'].append(c)
        elif tags & L2 and not (tags & L1):
            grp['L2_only'].append(c)
        else:
            grp['other'].append(c)
    q = lambda f: (vals[int(f * (len(vals) - 1))] if vals else None)
    return {'n_coords': len(items),
            'per_recipe': {'median': q(0.5), 'p25': q(0.25), 'p75': q(0.75), 'max': vals[-1] if vals else None},
            'null_matched': nl,
            'frac_recipes_above_null_p95': (None if not nl['p95'] else sum(1 for c in vals if c > nl['p95']) / len(vals)),
            'strata': {k: (sorted(v2)[len(v2) // 2] if v2 else None) for k, v2 in grp.items()},
            'strata_n': {k: len(v2) for k, v2 in grp.items()}}


def q3_verdict(lists):
    """The frozen wording rule: 'HIGH ALIGNMENT' only if the median over recipes exceeds the matched null p95."""
    return {k: ('HIGH ALIGNMENT' if (v['per_recipe']['median'] is not None and v['null_matched']['p95'] is not None
                                     and v['per_recipe']['median'] > v['null_matched']['p95']) else 'no high alignment')
            for k, v in lists.items()}
