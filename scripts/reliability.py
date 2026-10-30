#!/usr/bin/env python3
"""Reliability of a score-function gradient: split-half curve, noise scale, and intervals.

WHY A SINGLE MODULE. The audit (doc/audit-response-2026-10-30.md §4) found the same analysis written twice
with different conventions, and the retracted claim "1500 and 1424 agree within 5%" came from comparing a
noise-to-signal ratio kappa against 2*kappa. Everything numeric about reliability now lives here, with the
convention stated once:

    xi_i      = weight_i * s_i            (s_i = per-rollout score, dose units)
    ghat_m    = (1/m) sum_{i=1..m} xi_i
    E||ghat_m||^2 = ||g||^2 + tr(Sigma)/m          (exact for uncorrelated zero-mean xi)
    kappa     = tr(Sigma)/||g||^2
    split-half compares two DISJOINT halves of size K/2, so
        r(K) ~ K / (K + 2*kappa)       <-- note the 2. kappa is NOT the total sample scale.
    and  E[cos(A,B)] ~ ||g||^2/(||g||^2 + 2 tr(Sigma)/K) is an approximation that additionally needs
    concentration; it is reported as a model, never as an identity.

Estimators (ANOVA split, from n observations):
    tr(Sigma) ~ (sum_i ||xi_i||^2 - n ||xibar||^2) / (n - 1)
    ||g||^2   ~ ||xibar||^2 - tr(Sigma)/n          (may come out negative: that is a RESULT, not a bug)
Intervals come from resampling CONTEXTS with replacement (the independent unit is the context, not the
rollout: rollouts within a context share the prompt).
"""
from __future__ import annotations
import random
from typing import Sequence

import torch
import torch.nn.functional as F


def _flat(x):
    return x.reshape(-1).double()


def split_half_curve(S: Sequence[torch.Tensor], weights: Sequence[Sequence[float]],
                     ks=(2, 4, 8, 16, 32), n_split: int = 200, seed: int = 0):
    """S[c]: [K, dim] unweighted per-rollout scores; weights[c]: [K]. Disjoint halves of size K//2."""
    rng = random.Random(seed)
    out = {}
    for K in ks:
        vals = []
        for c, Sc in enumerate(S):
            Kc = Sc.shape[0]
            if Kc < K or K < 2:
                continue
            w = torch.tensor(list(weights[c]), dtype=torch.float64)
            for _ in range(n_split):
                idx = list(range(Kc))
                rng.shuffle(idx)
                A = torch.zeros(Sc.shape[1], dtype=torch.float64)
                B = torch.zeros(Sc.shape[1], dtype=torch.float64)
                for i in idx[:K // 2]:
                    A += w[i] * Sc[i].double()
                for i in idx[K // 2:K]:
                    B += w[i] * Sc[i].double()
                if float(A.norm()) > 0 and float(B.norm()) > 0:
                    vals.append(float(F.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0))))
        if vals:
            vals.sort()
            out[K] = {'r_mean': sum(vals) / len(vals), 'r_p10': vals[int(0.1 * len(vals))], 'n': len(vals)}
    return out


def _stats(S, weights):
    n = 0
    sum_sq = 0.0
    xibar = None
    for c, Sc in enumerate(S):
        w = torch.tensor(list(weights[c]), dtype=torch.float64)
        for i in range(Sc.shape[0]):
            xi = w[i] * Sc[i].double()
            xibar = xi.clone() if xibar is None else xibar + xi
            sum_sq += float(_flat(xi).pow(2).sum())
            n += 1
    xibar /= n
    nb = float(_flat(xibar).pow(2).sum())
    tr = (sum_sq - n * nb) / (n - 1)
    gsq = nb - tr / n
    return {'n': n, 'norm_xibar_sq': nb, 'tr_sigma': tr, 'g_sq': gsq}


def noise_scale(S, weights):
    """Pooled (contexts pooled, not averaged)--use for the total; see bootstrap_kappa for the interval."""
    st = _stats(S, weights)
    st['kappa'] = (st['tr_sigma'] / st['g_sq']) if st['g_sq'] > 0 else None
    st['r_K_predicted'] = (None if st['kappa'] is None else
                           {K: K / (K + 2 * st['kappa']) for K in (2, 4, 8, 16, 32)})
    return st


def bootstrap_kappa(S, weights, n_boot: int = 200, seed: int = 0):
    """Resample CONTEXTS with replacement; percentile interval for kappa (None if g_sq <= 0)."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pick = [rng.randrange(len(S)) for _ in S]
        st = _stats([S[i] for i in pick], [weights[i] for i in pick])
        if st['g_sq'] > 0:
            vals.append(st['tr_sigma'] / st['g_sq'])
    if not vals:
        return {'kappa_lo': None, 'kappa_hi': None, 'n_valid': 0}
    vals.sort()
    return {'kappa_lo': vals[int(0.025 * len(vals))], 'kappa_hi': vals[int(0.975 * len(vals)) - 1],
            'n_valid': len(vals)}
