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


def _jackknife(S, weights, stat):
    """Leave-one-context-out values of `stat`, for the BCa acceleration."""
    vals = []
    for k in range(len(S)):
        pick = [i for i in range(len(S)) if i != k]
        if not pick:
            continue
        st = _stats([S[i] for i in pick], [weights[i] for i in pick])
        v = stat(st)
        if v is not None:
            vals.append(v)
    return vals


def _bca(vals, point, jack, alpha=0.05):
    """Bias-corrected and accelerated percentile interval (Efron). With n=8 contexts the plain percentile
    interval is noticeably biased - the auditor caught this because the seed-0 point estimate fell outside its
    own percentile interval - so BCa is reported alongside, and the percentile one is kept for comparison."""
    import math
    from statistics import NormalDist
    nd = NormalDist()
    vals = sorted(vals)
    if not vals or point is None or not jack:
        return None
    n = len(vals)
    frac = sum(1 for v in vals if v < point) / n
    frac = min(max(frac, 1e-6), 1 - 1e-6)
    z0 = nd.inv_cdf(frac)
    jm = sum(jack) / len(jack)
    num = sum((jm - v) ** 3 for v in jack)
    den = 6.0 * (sum((jm - v) ** 2 for v in jack) ** 1.5)
    a = num / den if den != 0 else 0.0
    def adj(z):
        return nd.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))
    lo = adj(nd.inv_cdf(alpha / 2))
    hi = adj(nd.inv_cdf(1 - alpha / 2))
    def q(p):
        p = min(max(p, 0.0), 1.0)
        return vals[min(int(p * n), n - 1)]
    return [q(lo), q(hi)]


def bootstrap_stats(S, weights, n_boot: int = 400, seed: int = 0):
    """Resample CONTEXTS with replacement (the independent unit is the context: rollouts inside one context
    share the prompt). Returns, for tr(Sigma), ||g||^2 and kappa: the point estimate, the percentile interval
    and a BCa interval, plus the fraction of resamples with a positive ||g||^2.

    NOTE ON THE UNIT: changing the rollout RNG seed gives independent ROLLOUTS within the same eight contexts.
    It does not give an independent CONTEXT sample, so it cannot settle context-level reproducibility.
    """
    rng = random.Random(seed)
    pt = _stats(S, weights)
    point = {'tr_sigma': pt['tr_sigma'], 'g_sq': pt['g_sq'],
             'kappa': (pt['tr_sigma'] / pt['g_sq']) if pt['g_sq'] > 0 else None}
    tr_v, g_v, k_v = [], [], []
    for _ in range(n_boot):
        pick = [rng.randrange(len(S)) for _ in S]
        st = _stats([S[i] for i in pick], [weights[i] for i in pick])
        tr_v.append(st['tr_sigma']); g_v.append(st['g_sq'])
        if st['g_sq'] > 0:
            k_v.append(st['tr_sigma'] / st['g_sq'])
    def band(v):
        if not v:
            return None
        v = sorted(v)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]
    jk = {
        'tr_sigma': _jackknife(S, weights, lambda st: st['tr_sigma']),
        'g_sq': _jackknife(S, weights, lambda st: st['g_sq']),
        'kappa': _jackknife(S, weights, lambda st: (st['tr_sigma'] / st['g_sq']) if st['g_sq'] > 0 else None),
    }
    return {'tr_sigma': point['tr_sigma'], 'g_sq': point['g_sq'], 'kappa': point['kappa'],
            'tr_sigma_interval': band(tr_v), 'g_sq_interval': band(g_v), 'kappa_interval': band(k_v),
            'tr_sigma_bca': _bca(tr_v, point['tr_sigma'], jk['tr_sigma']),
            'g_sq_bca': _bca(g_v, point['g_sq'], jk['g_sq']),
            'kappa_bca': _bca(k_v, point['kappa'], jk['kappa']),
            'n_boot': n_boot, 'n_contexts': len(S),
            'frac_boot_g_sq_positive': (sum(1 for x in g_v if x > 0) / len(g_v)) if g_v else None,
            'unit_note': 'percentile and BCa intervals both resample CONTEXTS; with 8 contexts both are fragile, and the '
                         'BCa acceleration comes from the leave-one-context-out jackknife'}


# -----------------------------------------------------------------------------------------------------------
# FAST PATH (2026-10-30, after the console showed the GPU idle at 0% while all 12 CPUs sat at 1200%).
#
# The naive routines above touch 294,912-dimensional vectors O(n_splits * K) times and O(n_boot * n * K) times
# on the CPU in float64. That is 10^10 float ops per run, i.e. minutes of CPU while the GPU waits - the exact
# shape of the profile. Everything numeric they compute is a function of far smaller objects:
#
#   split-half:  cos(A,B) = <A,B>/sqrt(<A,A><B,B>) with A = sum_{i in half} w_i x_i, so with the per-context
#                Gram G = M M^T (K x K, K = rollouts) every split is a quadratic form p_A^T G p_B. One Gram per
#                context is one (K x d)(d x K) matmul; after that a split costs O(K^2).
#   bootstrap:   the ANOVA statistics only need, per context, the weighted sum s_c = sum_i w_i x_i and the
#                scalar q_c = sum_i w_i^2 ||x_i||^2. A context resample is then a sum of at most n of the s_c,
#                i.e. one (n_boot x n)(n x d) matmul plus a scalar gather - not n_boot * n * K vector adds.
# Same algebra, two orders of magnitude fewer flops.
# -----------------------------------------------------------------------------------------------------------
def context_summaries(S, weights):
    """Per-context (weighted sum vector, weighted squared-norm, count) - the only objects the bootstrap needs."""
    sums, qs, ns = [], [], []
    for c, Sc in enumerate(S):
        w = torch.tensor(list(weights[c]), dtype=torch.float64, device=Sc.device).unsqueeze(1)
        xw = Sc.double() * w
        sums.append(xw.sum(0))
        qs.append(float((xw * xw).sum()))
        ns.append(int(Sc.shape[0]))
    return torch.stack(sums), torch.tensor(qs, dtype=torch.float64), ns


def _stats_from_sums(sums_pick, q_pick, n_pick):
    """ANOVA split from resampled context summaries."""
    tot = sums_pick.sum(0)
    n = float(sum(n_pick))
    q = float(q_pick.sum())
    nb = float((tot / n).pow(2).sum())
    tr = (q - n * nb) / (n - 1)
    return tr, nb - tr / n


def bootstrap_stats_fast(S, weights, n_boot: int = 400, seed: int = 0, device=None):
    """BCa + percentile intervals from context summaries. Identical algebra to bootstrap_stats, ~30x fewer flops."""
    import torch as _t
    sums, qs, ns = context_summaries(S, weights)
    dev = sums.device if device is None else device
    sums, qs = sums.to(dev), qs.to(dev)
    C = sums.shape[0]
    tr_p, g_p = _stats_from_sums(sums, qs, ns)
    point = {'tr_sigma': tr_p, 'g_sq': g_p, 'kappa': (tr_p / g_p) if g_p > 0 else None}

    g = _t.Generator(device=dev) if dev.type == 'cuda' else _t.Generator()
    g.manual_seed(seed)
    idx = _t.randint(0, C, (n_boot, C), generator=g, device=dev)
    counts = _t.zeros(n_boot, C, dtype=_t.float64, device=dev)
    for j in range(C):
        counts[:, j] = (idx == j).sum(1).double()
    n_mat = counts @ _t.tensor(ns, dtype=_t.float64, device=dev)
    q_mat = counts @ qs
    tot = counts @ sums                      # (n_boot, d): one matmul
    nb = ((tot / n_mat.unsqueeze(1)) ** 2).sum(1)          # ||xibar||^2 per resample (chunked if needed)
    tr = (q_mat - n_mat * nb) / (n_mat - 1)
    gq = nb - tr / n_mat
    tr_v, g_v, k_v = tr.cpu().tolist(), gq.cpu().tolist(), []
    for a, b in zip(tr_v, g_v):
        if b > 0:
            k_v.append(a / b)

    def band(v):
        if not v:
            return None
        v = sorted(v)
        return [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]]
    # jackknife over contexts, from the same summaries
    jk = {'tr_sigma': [], 'g_sq': [], 'kappa': []}
    for k in range(C):
        keep = [i for i in range(C) if i != k]
        tr_k, g_k = _stats_from_sums(sums[keep], qs[keep], [ns[i] for i in keep])
        jk['tr_sigma'].append(tr_k); jk['g_sq'].append(g_k)
        if g_k > 0:
            jk['kappa'].append(tr_k / g_k)
    return {'tr_sigma': tr_p, 'g_sq': g_p, 'kappa': point['kappa'],
            'tr_sigma_interval': band(tr_v), 'g_sq_interval': band(g_v), 'kappa_interval': band(k_v),
            'tr_sigma_bca': _bca(tr_v, tr_p, jk['tr_sigma']),
            'g_sq_bca': _bca(g_v, g_p, jk['g_sq']),
            'kappa_bca': _bca(k_v, point['kappa'], jk['kappa']),
            'n_boot': n_boot, 'n_contexts': C,
            'frac_boot_g_sq_positive': (sum(1 for x in g_v if x > 0) / len(g_v)) if g_v else None,
            'path': 'fast (context summaries + one matmul)',
            'unit_note': 'resampling unit is the CONTEXT; a different rollout seed gives independent rollouts inside '
                         'the SAME contexts and does not settle context-level reproducibility'}


def split_half_curve_fast(S, weights, ks=(2, 4, 8, 16, 32), n_split: int = 200, seed: int = 0, device=None):
    """Split-half cosine from ONE Gram per context, batched.

    G_ij = w_i <x_i, x_j> is computed once per context (one (K x d)(d x K) matmul, i.e. O(K^2 d) total for the run
    instead of O(n_splits * K * d)). A split is then a pair of 0/1 indicator vectors over the context's rollouts and
    the cosine is a quadratic form; all n_splits splits are done as two batched matmuls so the GPU does the work.
    """
    import torch as _t
    rng = random.Random(seed)
    out = {}
    Grams = []
    for c, Sc in enumerate(S):
        dev = Sc.device if device is None else device
        M = Sc.double().to(dev)
        w = _t.tensor(list(weights[c]), dtype=_t.float64, device=dev)
        Grams.append((M * w.unsqueeze(1)) @ M.T)      # (Kc, Kc) once per context
    for K in ks:
        vals = []
        for c, Sc in enumerate(S):
            Kc = Sc.shape[0]
            if Kc < K or K < 2:
                continue
            G = Grams[c]
            PA = _t.zeros(n_split, Kc, dtype=_t.float64, device=G.device)
            PB = _t.zeros(n_split, Kc, dtype=_t.float64, device=G.device)
            for t in range(n_split):
                order = list(range(Kc))
                rng.shuffle(order)
                PA[t, order[:K // 2]] = 1.0
                PB[t, order[K // 2:K]] = 1.0
            VA = PA @ G                                # (n_split, Kc)
            VB = PB @ G
            ab = (VA * PB).sum(1)
            aa = (VA * PA).sum(1)
            bb = (VB * PB).sum(1)
            ok = (aa > 0) & (bb > 0)
            cs = (ab[ok] / (aa[ok].sqrt() * bb[ok].sqrt())).cpu().tolist()
            vals.extend(cs)
        if vals:
            vals.sort()
            out[K] = {'r_mean': sum(vals) / len(vals), 'r_p10': vals[int(0.1 * len(vals))], 'n': len(vals),
                      'path': 'fast (one Gram per context, batched splits)'}
    return out


def bootstrap_kappa(S, weights, n_boot: int = 200, seed: int = 0):
    """Backwards-compatible alias: kappa interval only (kept for older scripts)."""
    b = bootstrap_stats(S, weights, n_boot, seed)
    return {'kappa_lo': b['kappa_interval'][0], 'kappa_hi': b['kappa_interval'][1],
            'n_valid': int(b['frac_boot_g_sq_positive'] * n_boot)}
