#!/usr/bin/env python3
"""Selection rules for the position list, extended to K=2048 over all 32 layers, decided by the
closed form's own objective and then judged ONLY by external instruments (no readout here).

    Q(S) = g_S' M_SS^-1 g_S ,  M block-diagonal by layer (exact budget metric),  M_ii = 1

Q is monotone with diminishing returns, so "which rule" is a question about noise and structure,
not about finding the optimum (a K=3/5 brute force on a synthetic near-identity M puts greedy
within ~0.5-1% of optimum; the real gap is untested and would need enumeration).

Rules (same g, same M):
  absg       top-K by |g|                       -- the diagonal score, Q({i}) = g_i^2
  greedy     forward greedy max of Q            -- incumbent (plan 2.36/2.37)
  umag       rank by |u| with u = M^+ g         -- the coordinate the UNBUDGETED closed-form
                                                    solution leans on most; never tested before
  waterfill  per-layer marginal curves, isotonic (concave) smoothing, equal-slope allocation
             -- greedy compares raw tail marginals that are nearly tied (M is near-orthonormal),
                so smoothing the layer curves is a direct test of whether that tie-breaking is
                signal or noise; within each layer the prefix is still the greedy one
  random     control

Everything is computed per layer with the plan-2.21 identity (no 9,216^3 solve):
  e = W_l' g_l ,  d* = B_l^-1 e ,  u_l = W_l (B_l^-1 d*)  with B_l = W_l'W_l   (u = M^+ g)
so `umag` costs two 2,560x2,560 solves plus matvecs per layer.

No measurement is performed here. The external judging (their published +-0.8 KL, their logit
lens, their 24-link graph, cross-context stability) happens in tools/compare_community.py.
"""
from __future__ import annotations
import argparse, json, platform, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def build_M(lm, layers, dev):
    """Keep the Gram blocks ON THE GPU: the greedy does millions of small gathers and matvecs, and
    a single CPU thread costs ~0.15 s per step (measured) against ~0.01 s on the GPU."""
    M = {}
    for l in layers:
        Wt = lm.layers[l].mlp.down_proj.weight.detach().float().T.to(dev)
        Wt = Wt / Wt.norm(dim=1, keepdim=True).clamp(min=1e-12)
        M[l] = Wt @ Wt.T
        del Wt
    torch.cuda.empty_cache()
    return M


def greedy_trace(M, g, layers, dims, K, log_every=512):
    """Global greedy; also returns each layer's OWN marginal curve (valid because M is
    block-diagonal: picking a coordinate in layer l never changes another layer's marginals)."""
    S = {l: [] for l in layers}
    P = {l: torch.zeros(0, 0, dtype=torch.float64, device=g[l].device) for l in layers}
    y = {l: torch.zeros(0, dtype=torch.float64, device=g[l].device) for l in layers}
    diag = {l: torch.diagonal(M[l]).double() for l in layers}
    ar = {l: torch.arange(dims[l], device=diag[l].device) for l in layers}
    lmarg = {l: [] for l in layers}
    order = []
    for step in range(K):
        best = (-float('inf'), None)
        for l in layers:
            k = len(S[l])
            if k >= dims[l]:
                continue
            if k:
                v = M[l].index_select(1, torch.tensor(S[l], device=M[l].device)).double()
                gain = (g[l] - v @ y[l]) ** 2 / (diag[l] - ((v @ P[l]) * v).sum(1)).clamp(min=1e-12)
                gain[torch.tensor(S[l], device=gain.device)] = -float('inf')
            else:
                gain = g[l] ** 2 / diag[l]
            j = int(torch.argmax(gain))
            if float(gain[j]) > best[0]:
                best = (float(gain[j]), (l, j))
        _, (l, j) = best
        k = len(S[l])
        if k:
            vj = M[l].index_select(0, torch.tensor(S[l], device=M[l].device))[:, j].double()
            Pv = P[l] @ vj
            dj = float(diag[l][j] - vj @ Pv)
            newP = torch.zeros(k + 1, k + 1, dtype=torch.float64, device=g[l].device)
            newP[:k, :k] = P[l] + torch.outer(Pv, Pv) / dj
            newP[:k, k] = -Pv / dj
            newP[k, :k] = -Pv / dj
            newP[k, k] = 1.0 / dj
            P[l] = newP
        else:
            P[l] = (1.0 / diag[l][j]).reshape(1, 1)
        S[l].append(j)
        y[l] = P[l] @ g[l][S[l]]
        lmarg[l].append(best[0])
        order.append((l, j))
        if log_every and (step + 1) % log_every == 0:
            print(f'    greedy {step + 1}/{K}', flush=True)
    return order, S, lmarg


def pava_decreasing(v):
    """Isotonic (non-increasing) least-squares fit via pool-adjacent-violators, equal weights."""
    vals, wts, cnt = [], [], []
    for x in v:
        vals.append(float(x)); wts.append(1.0); cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            w = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / w); wts.append(w); cnt.append(c1 + c2)
    out = []
    for v, c in zip(vals, cnt):
        out.extend([v] * c)
    return out


def Qof(M, g, S, layers):
    tot = 0.0
    for l in layers:
        js = S[l]
        if js:
            idx = torch.tensor(js, device=M[l].device)
            Ms = M[l].index_select(0, idx).index_select(1, idx).double()
            gl = g[l].index_select(0, idx)
            tot += float(gl @ torch.linalg.solve(Ms, gl))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='/root/autodl-tmp/scripts/j2space')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--g', default='/root/autodl-tmp/runs/g_all32.npz')
    ap.add_argument('--ks', default='64,256,1024,2048')
    ap.add_argument('--seed', type=int, default=20261031)
    ap.add_argument('--out', default='/root/autodl-tmp/runs/select_rules.json')
    a = ap.parse_args()

    sys.path.insert(0, a.pkg)
    from engine import load_model

    layers = list(range(32))
    ks = [int(x) for x in a.ks.split(',')]
    K = max(ks)
    tok, model, lm = load_model(a.model)
    dev = 'cuda:0'
    dims = {l: int(lm.layers[l].mlp.down_proj.weight.shape[1]) for l in layers}
    torch.set_num_threads(int(__import__('os').environ.get('CPU_THREADS', '1')))
    M = build_M(lm, layers, dev)
    print(f'M: {sum(m.numel() * 4 for m in M.values()) / 1e9:.1f} GB on {dev}; '
          f'gpu mem {torch.cuda.memory_allocated() / 1e9:.1f} GB', flush=True)

    G = np.load(a.g)
    g = {l: torch.tensor(G['pooled'][l], dtype=torch.float64, device=dev) for l in layers}

    rules = {}
    abs_order = []
    gl = torch.cat([g[l].abs() for l in layers])
    top = torch.argsort(gl, descending=True)[:K]
    off = 0
    for l in layers:
        abs_order += [(l, int(i - off)) for i in top.tolist() if off <= i < off + dims[l]]
        off += dims[l]
    rules['absg'] = list(abs_order)
    print('absg done', flush=True)

    order, S_g, lmarg = greedy_trace(M, g, layers, dims, K)
    rules['greedy'] = [list(t) for t in order]
    print('greedy done', flush=True)

    # ---- umag: |u| with u = M^+ g, via the plan-2.21 identity (B is 2,560^2, not 9,216^2) ----
    umag_flat = []
    for l in layers:
        Wl = lm.layers[l].mlp.down_proj.weight.detach().double().T.to(dev)    # 9216 x 2560
        Wl = Wl / Wl.norm(dim=1, keepdim=True).clamp(min=1e-12)
        B = Wl.T @ Wl
        e = Wl.T @ g[l]
        dstar = torch.linalg.solve(B, e)
        u = Wl @ torch.linalg.solve(B, dstar)
        umag_flat.append(u.abs())
        del Wl, B
        torch.cuda.empty_cache()
    gu = torch.cat(umag_flat)
    umag_order = []
    off = 0
    for l in layers:
        umag_order += [(l, int(i - off)) for i in torch.argsort(gu, descending=True)[:K].tolist()
                       if off <= i < off + dims[l]]
        off += dims[l]
    rules['umag'] = umag_order
    print('umag done', flush=True)

    # ---- waterfill: per-layer concave curves, equal-slope allocation, greedy prefix inside ----
    smooth = {l: pava_decreasing(lmarg[l]) for l in layers}
    cand = []
    for l in layers:
        for k, m in enumerate(smooth[l]):
            cand.append((m, l, k))
    cand.sort(key=lambda x: -x[0])
    take = {}
    for m, l, k in cand:
        if sum(take.values()) >= K:
            break
        # only extend a layer's prefix contiguously
        if take.get(l, 0) == k:
            take[l] = k + 1
    alloc = {l: take.get(l, 0) for l in layers}
    print('waterfill allocation: ' + ', '.join(f'L{l}:{alloc[l]}' for l in layers if alloc[l]), flush=True)
    # waterfill keeps the greedy order but truncates each layer at its allocated count; the
    # induced global order is the greedy order filtered to the allocated prefix per layer.
    keep = {l: set(rules['greedy'][i][1] for i in range(len(rules['greedy']))
                   if rules['greedy'][i][0] == l) for l in layers}
    wf = []
    seen = {l: 0 for l in layers}
    for (l, j) in rules['greedy']:
        if seen[l] < alloc[l]:
            wf.append([l, j]); seen[l] += 1
    rules['waterfill'] = wf

    rng = np.random.default_rng(a.seed)
    rnd = []
    for l in layers:
        for n in rng.choice(dims[l], size=min(K // len(layers) + 1, dims[l]), replace=False):
            rnd.append([l, int(n)])
    rules['random'] = rnd

    out = {'schema': 'qwentest-select-rules-v1', 'machine': platform.node(),
           'ks': ks, 'K': K, 'layers': layers, 'dim_per_layer': dims,
           'scope': 'surrogate Q only, no readout measured; external judging happens separately',
           'waterfill_allocation': {str(l): alloc[l] for l in layers},
           'greedy_marginals_tail': {str(l): [round(x, 8) for x in lmarg[l][-3:]] for l in layers if lmarg[l]},
           'Q_at_K': {}, 'layer_histogram': {}, 'order': {}}
    for name, S in rules.items():
        out['Q_at_K'][name] = {}
        for k in ks:
            sub = {l: [] for l in layers}
            for (l, j) in S[:k]:
                sub[l].append(j)
            out['Q_at_K'][name][str(k)] = Qof(M, g, sub, layers)
        hist = {}
        for (l, j) in S:
            hist[str(l)] = hist.get(str(l), 0) + 1
        out['layer_histogram'][name] = hist
        out['order'][name] = S
        print(f'{name}: Q ' + ', '.join(f'K={k}:{v:.3f}' for k, v in out['Q_at_K'][name].items()), flush=True)

    Path(a.out).write_text(json.dumps(out, ensure_ascii=False))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
