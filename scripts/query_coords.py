#!/usr/bin/env python3
"""Query the closed form for a SPECIFIED coordinate: solo dose, its marginal value in a given
list, and the best list of size K that is FORCED to contain it. No measurement.

For a chosen coordinate i (layer, neuron) the three computable objects are:

1. SOLO        S = {i}.  Because M_ii = 1, u_i = g_i and b = |u_i| = |dose|: the budget IS the
               dose.  The raw activation change is alpha_i = u_i * p50_l / ||W_down[:,i]||.
2. MARGINAL    dQ of appending i to an existing support S: with P = M_SS^-1,
                   dQ(i) = (g_i - M_iS P g_S)^2 / (M_ii - M_iS P M_Si)
               (exact; this is also the first-order answer to "what does this neuron add?").
3. CONDITIONAL the greedy-optimal S of size K subject to i in S: warm-start the greedy with i
               already selected (the Schur state for {i} is one scalar), then continue. This is
               "the best combination that contains this point", and it also reports the dose the
               closed form assigns to i inside that combination and its share of the budget.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--pkg', default='/root/autodl-tmp/scripts/j2space')
ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
ap.add_argument('--g', default='/root/autodl-tmp/runs/g_all32.npz')
ap.add_argument('--coords', default='14:8949,13:7956,31:1910')
ap.add_argument('--K', type=int, default=64)
ap.add_argument('--budget', type=float, default=0.05)
ap.add_argument('--out', default='/root/autodl-tmp/runs/query_coords.json')
a = ap.parse_args()

sys.path.insert(0, a.pkg)
from engine import load_model

layers = list(range(32))
tok, model, lm = load_model(a.model)
dev = 'cuda:0'
dims = {l: int(lm.layers[l].mlp.down_proj.weight.shape[1]) for l in layers}
torch.set_num_threads(1)
G = np.load(a.g)
g = {l: torch.tensor(G['pooled'][l], dtype=torch.float64, device=dev) for l in layers}
p50 = {i: float(v) for i, v in enumerate(G['p50'])}

D = {}
for l in layers:
    W = lm.layers[l].mlp.down_proj.weight.detach().float()
    D[l] = W.norm(dim=0).cpu()            # ||W_down[:, n]||
    del W
torch.cuda.empty_cache()


def M_block(l):
    Wt = lm.layers[l].mlp.down_proj.weight.detach().float().T.to(dev)
    Wt = Wt / Wt.norm(dim=1, keepdim=True).clamp(min=1e-12)
    M = Wt @ Wt.T
    del Wt
    return M


def greedy(k, must=None, layers_used=layers):
    S = {l: [] for l in layers_used}
    P = {l: torch.zeros(0, 0, dtype=torch.float64, device=dev) for l in layers_used}
    y = {l: torch.zeros(0, dtype=torch.float64, device=dev) for l in layers_used}
    diag = {l: torch.diagonal(Mb[l]).double() for l in layers_used}
    if must is not None:
        l0, j0 = must
        S[l0].append(j0)
        P[l0] = (1.0 / diag[l0][j0]).reshape(1, 1)
        y[l0] = P[l0] @ g[l0][S[l0]]
    for _ in range(k - sum(len(v) for v in S.values())):
        best = (-float('inf'), None)
        for l in layers_used:
            kk = len(S[l])
            if kk >= dims[l]:
                continue
            if kk:
                v = Mb[l].index_select(1, torch.tensor(S[l], device=dev)).double()
                gain = (g[l] - v @ y[l]) ** 2 / (diag[l] - ((v @ P[l]) * v).sum(1)).clamp(min=1e-12)
                gain[torch.tensor(S[l], device=dev)] = -float('inf')
            else:
                gain = g[l] ** 2 / diag[l]
            j = int(torch.argmax(gain))
            if float(gain[j]) > best[0]:
                best = (float(gain[j]), (l, j))
        _, (l, j) = best
        kk = len(S[l])
        if kk:
            vj = Mb[l].index_select(0, torch.tensor(S[l], device=dev))[:, j].double()
            Pv = P[l] @ vj
            dj = float(diag[l][j] - vj @ Pv)
            newP = torch.zeros(kk + 1, kk + 1, dtype=torch.float64, device=dev)
            newP[:kk, :kk] = P[l] + torch.outer(Pv, Pv) / dj
            newP[:kk, kk] = -Pv / dj
            newP[kk, :kk] = -Pv / dj
            newP[kk, kk] = 1.0 / dj
            P[l] = newP
        else:
            P[l] = (1.0 / diag[l][j]).reshape(1, 1)
        S[l].append(j)
        y[l] = P[l] @ g[l][S[l]]
    return S


def Q_of(S):
    tot = 0.0
    for l, js in S.items():
        if js:
            idx = torch.tensor(js, device=dev)
            Ms = Mb[l].index_select(0, idx).index_select(1, idx).double()
            gl = g[l].index_select(0, idx)
            tot += float(gl @ torch.linalg.solve(Ms, gl))
    return tot


def doses(S, b):
    out = {}
    for l, js in S.items():
        if not js:
            continue
        idx = torch.tensor(js, device=dev)
        Ms = Mb[l].index_select(0, idx).index_select(1, idx).double()
        gl = g[l].index_select(0, idx)
        u = torch.linalg.solve(Ms, gl)
        b0 = math.sqrt(float(gl @ u))
        out[l] = (u * (b / b0)).cpu(), b0
    return out


Mb = {}
for l in layers:
    Mb[l] = M_block(l)
print(f'M: {sum(m.numel()*4 for m in Mb.values())/1e9:.1f} GB on GPU', flush=True)

coords = [tuple(int(x) for x in c.split(':')) for c in a.coords.split(',')]
base = greedy(a.K)
Qbase = Q_of(base)
res = {'K': a.K, 'budget': a.budget, 'Q_base': Qbase,
       'base_doses_share': {str(l): float(v[0].abs().pow(2).sum() / (v[1] ** 2)) for l, v in doses(base, 1.0).items()},
       'coords': {}}
print(f'base greedy K={a.K}: Q={Qbase:.4f}', flush=True)

for (l, n) in coords:
    i = n
    # SOLO
    solo_u = float(g[l][i] / torch.diagonal(Mb[l])[i])
    alpha = solo_u * p50[l] / float(D[l][i])
    solo = {'u': solo_u, 'b': abs(solo_u), 'Q': solo_u ** 2,
            'alpha_at_b0.05': (0.05 if solo_u >= 0 else -0.05) * p50[l] / float(D[l][i]),
            'alpha_at_their_0.7': (0.7 if solo_u >= 0 else -0.7) * p50[l] / float(D[l][i]),
            'p50': p50[l], 'Wnorm': float(D[l][i])}
    # MARGINAL against the base list
    if i in base[l]:
        dq = None
        note = 'already in the base list'
    else:
        idx = torch.tensor(base[l], device=dev)
        Ms = Mb[l].index_select(0, idx).index_select(1, idx).double()
        P = torch.linalg.inv(Ms)
        gl = g[l].index_select(0, idx)
        m = Mb[l][i].index_select(0, idx).double()
        num = float(g[l][i] - m @ (P @ gl)) ** 2
        den = float(torch.diagonal(Mb[l])[i] - m @ (P @ m))
        dq = num / den if den > 0 else None
        note = f'dQ vs base list = {dq:+.4f} (base Q={Qbase:.2f})' if dq is not None else 'degenerate'
    # CONDITIONAL: best K list containing i
    Sc = greedy(a.K, must=(l, i))
    Qc = Q_of(Sc)
    dc = doses(Sc, a.budget)
    share = float(dc[l][0][list(Sc[l]).index(i)].abs() ** 2 / (dc[l][1] ** 2)) if l in dc else None
    res['coords'][f'L{l}N{n}'] = {'solo': solo, 'marginal': {'dQ': dq, 'note': note},
                                  'conditional': {'Q': Qc, 'delta_vs_base': Qc - Qbase,
                                                  'dose_inside': None if share is None else float(dc[l][0][list(Sc[l]).index(i)]),
                                                  'budget_share_of_its_layer': share,
                                                  'layers_used': sum(1 for v in Sc.values() if v),
                                                  'n_positions': sum(len(v) for v in Sc.values())}}
    print(f'L{l}N{n}: solo u={solo_u:+.4f} alpha@0.7={solo["alpha_at_their_0.7"]:+.2f} | '
          f'marginal {note} | conditional Q={Qc:.3f} (Δ{Qc - Qbase:+.3f}), dose inside '
          f'{res["coords"][f"L{l}N{n}"]["conditional"]["dose_inside"]}', flush=True)

Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1))
print('wrote', a.out)
