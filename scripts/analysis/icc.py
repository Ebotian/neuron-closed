"""Intraclass correlation from a Gram matrix - pure Python, no torch, no model.

INPUT is only (G, group_sizes, k): G is the Gram of the weighted per-rollout score vectors, `G_ij = w_i w_j ⟪x_i,x_j⟫`,
flattened across contexts. Nothing here knows about d, tensors or the model, so it runs in `make check` and it can be
recomputed from a stored Gram months later.

ALGEBRA (all of it sums of squares, which is why it is insensitive to a common-mode bias - lean/Reliability.lean):
    SS_w = Σ_c [ Σ_{i∈c} G_ii − (Σ_{i,j∈c} G_ij)/k_c ]
    SS_t = Σ_i G_ii − (Σ_{ij} G_ij)/n
    SS_b = SS_t − SS_w
    MS_w = SS_w/(n−C),  MS_b = SS_b/(C−1),  tr(Σ) = MS_w,  ‖g‖² = (MS_b − MS_w)/k
    ICC  = (MS_b − MS_w)/(MS_b + (k−1)MS_w) ∈ [−1/(k−1), 1]  (proved)
  interval: F = MS_b/MS_w with (C−1, n−C) degrees of freedom; mapping F to ICC is monotone.
"""
from __future__ import annotations

import math


def _f_quantile(p, d1, d2, iters=200):
    def betacf(a, b, x):
        MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
        qab, qap, qam = a + b, a + 1.0, a - 1.0
        c, d = 1.0, 1.0 - qab * x / qap
        d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
        h = d
        for m in range(1, MAXIT + 1):
            m2 = 2 * m
            for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                       -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
                d = 1.0 + aa * d
                d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
                c = 1.0 + aa / c
                c = c if abs(c) > FPMIN else FPMIN
                h *= d * c
            if abs(d * c - 1.0) < EPS:
                break
        return h

    def betai(a, b, x):
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        lb = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        if x < (a + 1) / (a + b + 2):
            return math.exp(a * math.log(x) + b * math.log(1 - x) - lb) / a * betacf(a, b, x)
        return 1.0 - math.exp(b * math.log(1 - x) + a * math.log(x) - lb) / b * betacf(b, a, 1 - x)

    def cdf(f):
        return 0.0 if f <= 0 else betai(d1 / 2.0, d2 / 2.0, d1 * f / (d1 * f + d2))

    lo, hi = 1e-9, 1e9
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def icc_from_gram_py(G, group_sizes, k=None, alpha=0.05):
    """`G` as a list of rows of floats (n×n).

    WHAT `between_var` IS AND IS NOT (audit P1, 2026-10-30). With the one-way random-effects model
    `x_ci = mu + a_c + e_ci`, `E[MS_w] = tr Sigma_w` and `E[MS_b] = tr Sigma_w + k tr Sigma_b`, so
    `(MS_b - MS_w)/k` estimates **tr Sigma_b - the variance BETWEEN context means**, not `||mu||^2`.
    Both quantities are unchanged by adding a common vector to every observation, which is why this
    statistic is robust to a shared shift and also why it cannot answer "how strong is the mean gradient".
    The old field name `g_sq` invited exactly that misreading; it is kept only as a deprecated alias.
    `norm_estimate` is `||ghat||` for the pooled estimator (free from the Gram: the total of all entries
    is `||sum_i w_i x_i||^2`), and `||g||^2` itself needs INDEPENDENT replicates - see `cross_replicate`.
    """
    n = sum(group_sizes)
    C = len(group_sizes)
    if k is None:
        k = min(group_sizes)
    diag = [G[i][i] for i in range(n)]
    ss_w = 0.0
    off = 0
    for kc in group_sizes:
        blk = sum(G[i][j] for i in range(off, off + kc) for j in range(off, off + kc))
        ss_w += sum(diag[off:off + kc]) - blk / kc
        off += kc
    total = sum(sum(row) for row in G)
    ss_t = sum(diag) - total / n
    ss_b = ss_t - ss_w
    df_w, df_b = n - C, C - 1
    ms_w = ss_w / df_w
    ms_b = ss_b / df_b
    denom = ms_b + (k - 1) * ms_w
    icc = (ms_b - ms_w) / denom if denom != 0 else None
    between = (ms_b - ms_w) / k
    out = {'n': n, 'C': C, 'k': k, 'SS_w': ss_w, 'SS_b': ss_b,
           'tr_sigma': ms_w,                      # tr Sigma_w: rollout noise
           'between_var': between,                # tr Sigma_b: context heterogeneity (NOT ||mu||^2)
           'g_sq': between,                       # DEPRECATED alias for between_var; kept so old records still parse
           'norm_estimate': total ** 0.5,         # ||ghat|| for the pooled weighted estimator
           'icc': icc,
           'noise_over_between': (ms_w / between) if between > 0 else None,
           'lambda_ratio': (ms_w / between) if between > 0 else None}
    if ms_w > 0 and df_b > 0:
        F = ms_b / ms_w
        loF = F / _f_quantile(1 - alpha / 2, df_b, df_w)
        hiF = F * _f_quantile(1 - alpha / 2, df_w, df_b)
        icc_of = lambda Fv: (Fv - 1) / (Fv + k - 1)
        out['F'] = F
        out['icc_interval'] = [icc_of(loF), icc_of(hiF)]
        out['tr_sigma_interval'] = [ms_w * loF / F, ms_w * hiF / F]
    else:
        out.update({'F': None, 'icc_interval': None, 'tr_sigma_interval': None})
    out['unit'] = 'context is the independent unit; more rollouts do not settle context-level reproducibility'
    out['identifiability'] = {
        'between_var': 'estimates tr Sigma_b; shift-invariant, so it says nothing about ||mu||',
        'norm_estimate': 'estimates ||ghat|| for THIS sample; E||ghat||^2 = ||g||^2 + tr Cov(ghat), so it is not ||g||^2',
        'g_sq': 'DEPRECATED name for between_var - do not interpret as the squared norm of the mean gradient',
        'interval_caveat': ('the F interval is exact only for a SCALAR normal model. These observations are vectors, so the '
                            'squared dispersion is a weighted sum of chi-squares and the trace ratio is not F-distributed: the '
                            'interval is a heuristic whose coverage is NOT established, and an interval containing zero is not '
                            'evidence of no signal')}
    return out


def cross_replicate(g_a, g_b):
    """The two independent-replicate identities that make `||g||^2` estimable (audit P1 / section 7).

    With `ghat_A`, `ghat_B` independent and unbiased with equal covariance:
        E[<g_A, g_B>] = ||g||^2                  (signal, unbiased, no centring)
        0.5 * E||g_A - g_B||^2 = tr Cov(ghat)    (noise)
    The ratio is a signal-to-noise on the SAME scale, which is what `between_var` cannot give.
    """
    import torch
    a, b = g_a.double(), g_b.double()
    signal, noise = float(a @ b), 0.5 * float(((a - b) ** 2).sum())
    return {'signal_sq': signal, 'noise_trace': noise,
            'snr': (signal / noise) if noise > 0 else None,
            'norms': [float(a.norm()), float(b.norm())],
            'note': ('both arms must come from DISJOINT rollouts and recompute their own baseline inside their half; a '
                     'split taken after the full-group weights are formed is not independent')}
    return out
