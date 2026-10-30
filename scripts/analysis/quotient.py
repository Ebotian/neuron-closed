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


def randomised_like(items, Wn, rng, mode='within_layer'):
    """A random list carrying the SAME (sign, dose) pairs and per-layer counts as `items`.

    THE FIRST VERSION WAS WRONG IN TWO WAYS and both are fixed here:
      * it drew sign and dose independently (`rng.choice(signs) * rng.choice(doses)`), which keeps the marginal
        distributions but destroys the actual signed-dose pairing and the per-layer dose multiset;
      * the null list was compared against OUR list, so the null of `cos(random, ours)` was used to judge
        `cos(recipe, ours)` - the recipes never entered the null at all.
    `mode='within_layer'` keeps each layer's dose multiset and randomises only which neurons receive them (the primary
    null). `mode='across_layers'` also shuffles the (layer, dose) pairs, which additionally tests the layer allocation.
    """
    pairs = [(l, d) for (l, _, d) in items]
    if mode == 'within_layer':
        by_layer = {}
        for l, d in pairs:
            by_layer.setdefault(l, []).append(d)
        out = []
        for l, doses in by_layer.items():
            doses = list(doses)
            rng.shuffle(doses)
            width = Wn[l].shape[1]
            for n, d in zip(rng.sample(range(width), min(len(doses), width)), doses):
                out.append((l, n, d))
        return out
    if mode == 'across_layers':
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        out = []
        used = {}
        for l, d in shuffled:
            width = Wn[l].shape[1]
            taken = used.setdefault(l, set())
            if len(taken) >= width:
                continue
            n = rng.choice(range(width))
            while n in taken:
                n = rng.choice(range(width))
            taken.add(n)
            out.append((l, n, d))
        return out
    raise ValueError(f'unknown null mode {mode}')


def _stats(vals):
    vals = sorted(vals)
    if not vals:
        return {'n': 0, 'mean': None, 'p95': None, 'max': None}
    return {'n': len(vals), 'mean': sum(vals) / len(vals),
            'p95': vals[int(0.95 * len(vals)) - 1], 'max': vals[-1]}


def null_for_recipes(items, recipe_v, Wn, layers, n_null, seed=0, mode='within_layer'):
    """Per-recipe null of the OBSERVED statistic `cos(ours, recipe_r)`: randomise OURS, keep each recipe fixed.

    Returns one statistics dict per recipe, in the same order as `recipe_v`. The exceedance fraction then compares each
    observed cosine with ITS OWN recipe's null, which is the only version that answers the question asked.
    """
    rng = random.Random(seed)
    draws = [[] for _ in recipe_v]
    for _ in range(n_null):
        v = to_v(randomised_like(items, Wn, rng, mode), Wn, layers)
        for i, (_, rv, _) in enumerate(recipe_v):
            c = cos_direct(v, rv, layers)
            if c is not None:
                draws[i].append(c)
    return [_stats(v) for v in draws]


def report_list(items, recipe_v, Wn, layers, L1, L2, n_null, seed=0, mode='within_layer'):
    """Everything the comparison reports for ONE position list: per-recipe quantiles, the per-recipe null, the strata."""
    v = to_v(items, Wn, layers)
    cs = [(c, tags) for (c, tags) in ((cos_direct(v, rv, layers), tags) for (_, rv, tags) in recipe_v) if c is not None]
    vals = sorted(c for c, _ in cs)
    nl = null_for_recipes(items, recipe_v, Wn, layers, n_null, seed, mode)
    # per-recipe exceedance: does THIS recipe's observed cosine beat THIS recipe's null?
    pairs = [(c, s['p95']) for (c, _), s in zip(cs, nl) if s['p95'] is not None]
    frac = (sum(1 for c, p in pairs if c > p) / len(pairs)) if pairs else None
    p95s = sorted(s['p95'] for s in nl if s['p95'] is not None)
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
            'null_per_recipe': nl,
            'null_p95_median': (p95s[len(p95s) // 2] if p95s else None),
            'frac_recipes_above_own_null_p95': frac,
            'null_mode': mode,
            'strata': {k: (sorted(v2)[len(v2) // 2] if v2 else None) for k, v2 in grp.items()},
            'strata_n': {k: len(v2) for k, v2 in grp.items()}}


def q3_verdict(lists):
    """The frozen wording rule: 'HIGH ALIGNMENT' only if the median over recipes exceeds the median of the per-recipe
    null p95s. Not reaching it is NOT evidence of equivalence - no threshold for "alignment is small" is defined."""
    out = {}
    for k, v in lists.items():
        med, p95 = v['per_recipe']['median'], v.get('null_p95_median')
        out[k] = ('HIGH ALIGNMENT' if (med is not None and p95 is not None and med > p95) else 'no high alignment')
    return out
