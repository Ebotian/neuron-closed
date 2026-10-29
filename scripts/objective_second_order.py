#!/usr/bin/env python3
"""Second-order version of the same-objective test: does the objective's own curvature (Fisher)
bring our positions closer to the community's behaviourally-found coordinates?

Why this is the right next probe. Plan 2.39.2 showed the residual gap is NOT the sampling surface:
exact overlap with their 1-4 coordinates stayed at 1.67% from 1 to 4 prompt surfaces while the sign
agreement held at 93-99%. The natural remaining candidate is the objective's curvature: a behaviour
appears through joint/threshold effects (their own record: a 6+1 analytic sum measured at 16, 2.3x
superadditive), and a first-order gradient cannot see that.

Formulation in residual space (why it is cheap). With d_l = What_l' u_l, the budget is exactly
||d_l||^2 by construction, i.e. the metric is the IDENTITY in d. So maximising the objective's
second-order model g_d'd - 1/2 d'F_d d under a fixed budget gives

    d = (I + mu F_d)^-1 g_d ,    mu = 1 / median(diag(F_d))   (the paper's median damping convention)
    u = What_l d_l                                             (minimum-norm representative)

where F_d = sum_t a_t a_t' with a_t = What_l h_t, h_t the per-token dose-space gradient. Building
F_d costs ONE BACKWARD PER TARGET TOKEN per layer (2,560^2 per layer, no 9,216^3 solve).

Arms compared per recipe, all on the same g:
  absg    top-K by |g|                       (first-order, coordinate metric)
  umag    top-K by |M^+ g|                   (first-order, budget metric; plan 2.38)
  so      top-K by |u| with u = What (I+mu F_d)^-1 g_d   (second order, budget metric)
Metrics: exact overlap at K = their neuron count, their coordinates inside our top-64, sign
agreement of sign(g_i) with their dose sign, and the layer share of our top-64. No generation.
"""
from __future__ import annotations
import argparse, json, math, statistics, sys
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--pkg', default='/root/autodl-tmp/scripts/j2space')
ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
ap.add_argument('--recipes', default='/root/autodl-tmp/scripts/neuronary_141.json')
ap.add_argument('--model-name', default='Qwen/Qwen3.5-4B')
ap.add_argument('--limit', type=int, default=12)
ap.add_argument('--k-order', type=int, default=64)
ap.add_argument('--max-target-tokens', type=int, default=96)
ap.add_argument('--mu-scale', default='0.3,1,3', help='mu = scale / median(diag F_d)')
ap.add_argument('--out', default='/root/autodl-tmp/runs/objective_second_order.json')
a = ap.parse_args()

sys.path.insert(0, a.pkg)
from engine import capture_mlp_inputs, column_norms, estimate_p50, load_model
from progress import Progress

layers = list(range(32))
tok, model, lm = load_model(a.model)
dev = 'cuda:0'
dims = {l: int(lm.layers[l].mlp.down_proj.weight.shape[1]) for l in layers}
norms = column_norms(lm, layers)
p50 = estimate_p50(tok, model, lm, layers, device=dev)

W, M = {}, {}
for l in layers:
    Wt = lm.layers[l].mlp.down_proj.weight.detach().float().T.to(dev)      # 9216 x 2560
    Wt = Wt / Wt.norm(dim=1, keepdim=True).clamp(min=1e-12)
    W[l] = Wt
    M[l] = Wt @ Wt.T
torch.cuda.empty_cache()
torch.set_num_threads(1)
print(f'W/M on GPU: {sum(m.numel() * 4 for m in M.values()) / 1e9:.2f} GB', flush=True)


def chat_ids(prompt):
    ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True,
                                  enable_thinking=False, tokenize=True)
    return ids['input_ids'] if hasattr(ids, 'keys') else ids


def token_grads(ids_p, ids_t):
    """Per-token dose-space gradients h_t (9216 per layer) of log p(token_t | ...)."""
    ids = torch.tensor([list(ids_p) + list(ids_t)], device=dev)
    with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
        out = model(input_ids=ids, use_cache=False)
        lg = out.logits[0]
        lo = len(ids_p) - 1
        lp = torch.log_softmax(lg[lo: lo + len(ids_t)].float(), -1)
        lps = lp.gather(1, torch.tensor(ids_t, device=dev)[:, None]).squeeze(1)
        hs = []
        for t in range(len(ids_t)):
            gs = torch.autograd.grad(lps[t], [cap[l] for l in layers], retain_graph=True)
            h = {}
            for l, gr in zip(layers, gs):
                sc = gr.detach().double().sum(dim=tuple(range(gr.ndim - 1)))
                h[l] = (sc * (p50[l] / norms[l].to(sc.device).double().clamp(min=1e-12))).cpu()
            hs.append(h)
        del out, lg, lp
    return hs


def rank_by(vec_per_layer, K, coord_space=True):
    flat, off = [], 0
    for l in layers:
        flat.append(vec_per_layer[l].abs())
    if not coord_space:      # residual space: map back to coordinates first
        raise SystemExit('residual lists are not used directly')
    cat = torch.cat(flat)
    top = torch.argsort(cat, descending=True)[:K]
    out, off = [], 0
    for l in layers:
        n = dims[l]
        for i in top.tolist():
            if off <= i < off + n:
                out.append((l, i - off))
        off += n
    return out[:K]


recs = [r for r in json.load(open(a.recipes)) if r['model'] == a.model_name and r.get('neurons')]
recs = [r for r in recs if (r.get('examples') or [{}])[0].get('prompt')][:a.limit]
scales = [float(x) for x in a.mu_scale.split(',')]
print(f'recipes: {len(recs)}, mu scales {scales}', flush=True)
prog = Progress(len(recs) * a.max_target_tokens)
res = {'schema': 'qwentest-objective-second-order-v1', 'n_recipes': len(recs), 'k_order': a.k_order,
       'mu_scales': scales, 'per_recipe': {}, 'summary': {}}

for r in recs:
    ex = r['examples'][0]
    their = [(int(n['layer']), int(n['neuron']), float(n['dose'])) for n in r['neurons']]
    ts = {(l, n) for l, n, _ in their}
    ids_p = chat_ids(ex['prompt'])
    ids_t = tok(ex['output'], add_special_tokens=False).input_ids[:a.max_target_tokens]
    hs = []
    # one graph, retained; one backward per target token
    ids = torch.tensor([list(ids_p) + list(ids_t)], device=dev)
    with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
        out = model(input_ids=ids, use_cache=False)
        lg = out.logits[0]
        lo = len(ids_p) - 1
        lp = torch.log_softmax(lg[lo: lo + len(ids_t)].float(), -1)
        lps = lp.gather(1, torch.tensor(ids_t, device=dev)[:, None]).squeeze(1)
        for t in range(len(ids_t)):
            gs = torch.autograd.grad(lps[t], [cap[l] for l in layers],
                                     retain_graph=(t < len(ids_t) - 1))
            h = {}
            for l, gr in zip(layers, gs):
                sc = gr.detach().double().sum(dim=tuple(range(gr.ndim - 1)))
                h[l] = (sc * (p50[l] / norms[l].to(sc.device).double().clamp(min=1e-12))).cpu()
            hs.append(h)
            prog.step(f'{r["id"]} tok {t}')
        del out, lg, lp
    g = {l: sum(h[l] for h in hs) / len(hs) for l in layers}
    # residual-space objects
    gd = {l: (W[l].T.double() @ g[l].to(dev)) for l in layers}   # g_d = What^T g  (9216 -> 2560)
    Fd = {}
    for l in layers:
        A = torch.stack([(W[l].T.double() @ h[l].to(dev)) for h in hs])   # T x 2560: a_t = What^T h_t
        Fd[l] = (A.T @ A) / len(hs)
        del A
    entry = {'name': r['name'], 'n_their': len(their), 'their': their,
             'sign_agree': sum(1 for l, n, d in their if np.sign(float(g[l][n])) == np.sign(d))}
    order_abs = rank_by(g, a.k_order)
    uM = {}
    for l in layers:
        B = W[l].T.double() @ W[l].double()
        e = W[l].double().T @ g[l].to(dev)
        dstar = torch.linalg.solve(B, e)
        uM[l] = (W[l].double() @ torch.linalg.solve(B, dstar)).cpu()
    order_umag = rank_by(uM, a.k_order)
    def stats(order):
        rank = {c: i + 1 for i, c in enumerate(order)}
        return {'overlap_at_n': len(ts & set(order[:len(their)])),
                'n_in_top64': sum(1 for l, n, _ in their if (l, n) in rank),
                'ranks': [rank.get((l, n)) for l, n, _ in their]}
    entry['absg'] = stats(order_abs)
    entry['umag'] = stats(order_umag)
    for sc in scales:
        d = {}
        for l in layers:
            med = float(torch.diagonal(Fd[l]).median()) or 1.0
            mu = sc / med
            G = torch.eye(Fd[l].shape[0], dtype=torch.float64, device=dev) + mu * Fd[l]
            d[l] = torch.linalg.solve(G, gd[l])
        u = {l: (W[l].double() @ d[l]).cpu() for l in layers}
        entry[f'so|mu={sc:g}'] = stats(rank_by(u, a.k_order))
    res['per_recipe'][r['id']] = entry

for key in ['absg', 'umag'] + [f'so|mu={s:g}' for s in scales]:
    rows = [e[key] for e in res['per_recipe'].values() if key in e and e[key]]
    if not rows:
        continue
    ranks = [x for row in rows for x in row['ranks'] if x]
    res['summary'][key] = {
        'n': len(rows),
        'recipes_with_any_overlap_at_n': sum(1 for x in rows if x['overlap_at_n'] > 0) / len(rows),
        'mean_overlap_at_n': statistics.fmean(x['overlap_at_n'] for x in rows),
        'recipes_with_any_in_top64': sum(1 for x in rows if x['n_in_top64'] > 0) / len(rows),
        'mean_their_in_top64': statistics.fmean(x['n_in_top64'] for x in rows),
        'median_rank_when_present': statistics.median(ranks) if ranks else None,
    }
res['summary']['sign_agree_g'] = statistics.fmean(e['sign_agree'] / max(1, e['n_their'])
                                                 for e in res['per_recipe'].values())
Path(a.out).write_text(json.dumps(res, ensure_ascii=False))
print('\n=== summary ===')
for k, v in res['summary'].items():
    print(' ', k, v)
print('wrote', a.out)
