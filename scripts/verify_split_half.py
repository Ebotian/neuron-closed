#!/usr/bin/env python3
"""Per-value verification of analysis/split_half.py (task-4).

The reference is a BRUTE FORCE construction: build the two half-sums as vectors of dimension d and take their
cosine directly. The Gram path must agree with it to |diff| < 1e-12 on the same data - which is exactly the check
that the one-sided weighting bug would have failed (it produced "cosines" above 1).
No model is needed: pure torch, CPU is enough.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here)); sys.path.insert(0, str(_here / 'analysis'))
from analysis.split_half import split_half_curve


def brute(S, W, K, n_split, seed=0):
    """Same splits, but with explicit half-sum vectors."""
    import random
    rng = random.Random(seed)
    out = []
    for c, Sc in enumerate(S):
        Kc = Sc.shape[0]
        if Kc < K:
            continue
        w = torch.tensor(list(W[c]), dtype=torch.float64)
        for _ in range(n_split):
            order = list(range(Kc)); rng.shuffle(order)
            A = sum(w[i] * Sc[i].double() for i in order[:K // 2])
            B = sum(w[i] * Sc[i].double() for i in order[K // 2:K])
            if A.norm() > 0 and B.norm() > 0:
                out.append(float(torch.nn.functional.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0))))
    return out


def main():
    torch.manual_seed(0)
    C, K, d = 3, 8, 64
    S = [torch.randn(K, d, dtype=torch.float64) for _ in range(C)]
    W = [[0.01 * (i + 1) * (1 if i % 2 else -1) for i in range(K)] for _ in range(C)]
    # ONE K only, so both implementations consume the RNG in the same order and see the SAME splits.
    got = split_half_curve(S, W, ks=(8,), n_split=50, seed=7, device='cpu', collect=True)
    ref = sorted(brute(S, W, 8, 50, seed=7))
    g = sorted(got[8]['values'])
    print(f'Gram path: n={len(g)}   brute: n={len(ref)}')
    d1 = max(abs(a - b) for a, b in zip(g, ref)) if len(g) == len(ref) else float('inf')
    print(f'max per-split |difference| = {d1:.3e}   max |cos| = {max(abs(x) for x in ref):.6f}')
    ok = len(g) == len(ref) and d1 < 1e-12 and max(abs(x) for x in ref) <= 1.0
    print('SPLIT-HALF-VERIFY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
