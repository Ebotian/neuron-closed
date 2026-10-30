"""Split-half reliability - the ONE implementation, from the Gram matrix.

WHY FROM THE GRAM. The cosine between two half-sums `A = Σ_{i∈half} w_i x_i`, `B = Σ_{j∈other} w_j x_j` is a quadratic
form: with `G_ij = w_i w_j ⟪x_i, x_j⟫`, `⟪A,B⟫ = p_Aᵀ G p_B`. So the whole curve costs O(K²) per split instead of
O(K·d), which is what moved the analysis off the CPU (the console showed 0% GPU and 1200% CPU while 294,912-dim
float64 vectors were being added).

HISTORY, because it is the reason this file exists. Two implementations once coexisted here, and the "fast" one
weighted the Gram on ONE side only (`G_ij = w_i ⟪x_i,x_j⟫`), which is not a squared norm - the printed "cosines"
reached 1.82 and 4.71. `scripts/verify_split_half.py` now compares this implementation against a brute-force
construction of the two half-sums, so that class of error cannot pass silently again.

CONVENTION (stated once): halves are DISJOINT and of size K//2, so `r(K) ~ K/(K + 2κ)`; κ is the noise-to-signal
ratio, and the sample requirement is `K = 2κ`, not κ.
"""
from __future__ import annotations
import random

import torch


def split_half_curve(S, weights, ks=(2, 4, 8, 16, 32), n_split: int = 200, seed: int = 0, device=None, collect=False):
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
        Mw = M * w.unsqueeze(1)
        # BOTH sides must carry the weight: G_ij = w_i w_j <x_i, x_j>. With one side only, p^T G p is not a squared
        # norm and the "cosine" can exceed 1 - which is exactly what this file's brute-force verifier caught, and what
        # earlier printed 1.82 and 4.71.
        Grams.append(Mw @ Mw.T)                        # (Kc, Kc) once per context
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
                      'path': 'Gram (two-sided weights, batched splits)'}
            if collect:
                out[K]['values'] = vals
    return out


