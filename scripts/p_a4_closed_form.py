#!/usr/bin/env python3
"""Plan 2.43 re-run: both d* candidates through the same closed form, as a thin shell.

FROZEN CRITERIA: doc/plan-2.43-frozen.md. The closed form lives in core/closedform.py, the injection in
estimators/steer.py; this file only assembles. Deviation on record: the Gram is cached on disk (it depends only on the
weights) and the d* blob is named and hashed in the output so the record pins its own input.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
sys.path.insert(0, str(Path(__file__).resolve().parent))
for sub in ('core', 'estimators', 'analysis'):
    sys.path.insert(0, str(Path(__file__).resolve().parent / sub))
import engine as E
from core.units import column_norms_dev, dose_scale
from core.loading import load_model_fast
from core.closedform import build_gram, solve_min_norm, budget2, cos_M, top
from estimators.steer import readout


def main():
    ap = argparse.ArgumentParser()
    A = ap.add_argument
    A('--model', default='/root/autodl-tmp/Qwen3.5-4B'); A('--probes', default='/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')
    A('--gfile', default='/root/autodl-tmp/runs/p_a2_v2.pt'); A('--budgets', default='0.05,0.1')
    A('--cache', default='/root/autodl-tmp/runs/gram_full32.pt'); A('--n-ctx', type=int, default=8)
    A('--out', default='/root/autodl-tmp/runs/p_a4.json')
    a = ap.parse_args()
    dev = 'cuda:0'
    torch.manual_seed(0)
    ctx = [r for r in map(json.loads, open(a.probes)) if r['trait'] == 'hostility'][:a.n_ctx]
    tok, model, lm = load_model_fast(a.model, dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = column_norms_dev(lm, layers, dev)
    SC = {l: v.to(dev) for l, v in dose_scale(p50, norms, layers).items()}
    gsha = hashlib.sha256(Path(a.gfile).read_bytes()).hexdigest()
    blob = torch.load(a.gfile)
    g = {'j2': blob['g_teach'], 'pw': blob['g_pw']}
    G = build_gram(lm, layers, dev, cache_path=a.cache)
    sym = max(float((G[l] - G[l].T).abs().max()) for l in layers)

    U, Qv, Rk, SELF = {}, {}, {}, {}
    for k, d in g.items():
        U[k], Qv[k], Rk[k] = solve_min_norm(lm, layers, d, dev)
    # SELF: M u must reproduce d / sqrt(q)  (the min-norm solution solves the system)
    for k in g:
        worst = 0.0
        for l in layers:
            W = lm.layers[l].mlp.down_proj.weight.detach().float().to(dev)
            Wn = W / W.norm(dim=0, keepdim=True).clamp(min=1e-12)
            ul = U[k][l].to(dev).float()
            r = Wn.T @ (Wn @ ul) - g[k][l].to(dev).float() / Qv[k] ** 0.5
            worst = max(worst, float(r.norm() / g[k][l].to(dev).float().norm()))
            del W, Wn
        SELF[k] = worst
    b2 = budget2(G, layers, U['j2'], dev)
    cm = cos_M(G, layers, U['j2'], U['pw'], dev)
    coord = float(torch.nn.functional.cosine_similarity(
        torch.cat([U['j2'][l] for l in layers]).unsqueeze(0),
        torch.cat([U['pw'][l] for l in layers]).unsqueeze(0)))
    ov = {K: {'shared': len(top(U['j2'], layers, K) & top(U['pw'], layers, K)),
              'jaccard': len(top(U['j2'], layers, K) & top(U['pw'], layers, K)) / len(top(U['j2'], layers, K) | top(U['pw'], layers, K))}
          for K in (64, 256, 1024)}
    base = readout(model, lm, tok, ctx, layers, None, 0.0, SC, dev)
    res = {'base': base}
    for b in [float(x) for x in a.budgets.split(',')]:
        for k in ('j2', 'pw'):
            v = readout(model, lm, tok, ctx, layers, U[k], b, SC, dev)
            res[f'{k}@{b}'] = {'J2': v, 'dJ2': v - base}
    out = {'SELF_residual': SELF, 'gfile': a.gfile, 'gfile_sha256': gsha,
           'C0_gram': {'max_asym': sym, 'rank_per_layer': Rk['j2'][:4], 'budget2_j2': b2},
           'A': {'cos_M': cm, 'coord_cos': coord}, 'B': ov, 'C': res}
    json.dump(out, open(a.out, 'w'), indent=1)
    print('SELF %s  budget2=%.6f  cos_M=%+.4f' % ({k: round(v, 8) for k, v in SELF.items()}, b2, cm), flush=True)
    for b in [float(x) for x in a.budgets.split(',')]:
        print('  b=%.2f  dJ2 j2=%+.5f pw=%+.5f ratio=%.3f' % (b, res[f'j2@{b}']['dJ2'], res[f'pw@{b}']['dJ2'],
              res[f'j2@{b}']['dJ2'] / res[f'pw@{b}']['dJ2']), flush=True)
    print('wrote', a.out, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
