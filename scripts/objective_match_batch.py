#!/usr/bin/env python3
"""Scale the "same objective, our algorithm" test from one recipe to ALL of them, and separate
the two factors that plan 2.39 could not: the OBJECTIVE (plain vs contrast) and the LAYER SCALING
(raw |g| vs layer-equalised).

Why layer scaling is the missing knob. On one recipe (Cookie Monster) both objectives gave 0/4
overlap and 3 of our 4 picks sat in L31: a final-logit objective can always be moved most cheaply
in the last layer, so raw |g| hands the budget to L31. Two different remedies are therefore
confounded in that result:
  * the objective (plain vs contrast vs theirs), and
  * the cross-layer calibration of |g| (raw vs per-layer RMS).
This script varies both, on every recipe, and reports where their published coordinates land.

Per recipe and per setting we record: whether each of their coordinates appears in our top-K,
its rank, the sign agreement between sign(g_i) and their dose sign, and the Jaccard of the two
sets at K = their number of neurons. No generation and no readout of ours.

GPU: 2 forward+backward passes per recipe (plain J and contrast J, sharing one forward graph per
continuation) + one M build for all 32 layers; greedy is seconds per recipe.
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
ap.add_argument('--limit', type=int, default=0, help='0 = all recipes of this model')
ap.add_argument('--k-order', type=int, default=64, help='depth of our order used for ranking')
ap.add_argument('--max-target-tokens', type=int, default=96)
ap.add_argument('--out', default='/root/autodl-tmp/runs/objective_match_batch.json')
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
M = {}
for l in layers:
    W = lm.layers[l].mlp.down_proj.weight.detach().float().T.to(dev)
    W = W / W.norm(dim=1, keepdim=True).clamp(min=1e-12)
    M[l] = W @ W.T
    del W
torch.cuda.empty_cache()
torch.set_num_threads(1)
print(f'M on GPU: {sum(m.numel() * 4 for m in M.values()) / 1e9:.1f} GB', flush=True)


def mean_logp(ids_p, ids_t):
    ids = torch.tensor([list(ids_p) + list(ids_t)], device=dev)
    out = model(input_ids=ids, use_cache=False)
    lg = out.logits[0]
    lp = torch.log_softmax(lg[len(ids_p) - 1: len(ids_p) - 1 + len(ids_t)].float(), -1)
    return lp.gather(1, torch.tensor(ids_t, device=dev)[:, None]).mean()


def grads_of(obj, caps):
    gs = torch.autograd.grad(obj, [caps[l] for l in layers])
    g = {}
    for l, gr in zip(layers, gs):
        sc = gr.detach().double().sum(dim=tuple(range(gr.ndim - 1)))
        g[l] = (sc * (p50[l] / norms[l].to(sc.device).double().clamp(min=1e-12))).cpu()
    return g


def greedy(g, K):
    S = {l: [] for l in layers}
    P = {l: torch.zeros(0, 0, dtype=torch.float64, device=dev) for l in layers}
    y = {l: torch.zeros(0, dtype=torch.float64, device=dev) for l in layers}
    diag = {l: torch.diagonal(M[l]).double() for l in layers}
    gd = {l: g[l].to(dev) for l in layers}
    order = []
    for _ in range(K):
        best = (-float('inf'), None)
        for l in layers:
            k = len(S[l])
            if k >= dims[l]:
                continue
            if k:
                v = M[l].index_select(1, torch.tensor(S[l], device=dev)).double()
                gain = (gd[l] - v @ y[l]) ** 2 / (diag[l] - ((v @ P[l]) * v).sum(1)).clamp(min=1e-12)
                gain[torch.tensor(S[l], device=dev)] = -float('inf')
            else:
                gain = gd[l] ** 2 / diag[l]
            j = int(torch.argmax(gain))
            if float(gain[j]) > best[0]:
                best = (float(gain[j]), (l, j))
        _, (l, j) = best
        k = len(S[l])
        if k:
            vj = M[l].index_select(0, torch.tensor(S[l], device=dev))[:, j].double()
            Pv = P[l] @ vj
            dj = float(diag[l][j] - vj @ Pv)
            newP = torch.zeros(k + 1, k + 1, dtype=torch.float64, device=dev)
            newP[:k, :k] = P[l] + torch.outer(Pv, Pv) / dj
            newP[:k, k] = -Pv / dj
            newP[k, :k] = -Pv / dj
            newP[k, k] = 1.0 / dj
            P[l] = newP
        else:
            P[l] = (1.0 / diag[l][j]).reshape(1, 1)
        S[l].append(j)
        y[l] = P[l] @ gd[l][S[l]]
        order.append((l, j))
    return order


def layer_equalise(g):
    out = {}
    for l in layers:
        rms = float(torch.sqrt((g[l] ** 2).mean())) or 1.0
        out[l] = g[l] / rms
    return out


recs = [r for r in json.load(open(a.recipes)) if r['model'] == a.model_name]
if a.limit:
    recs = recs[:a.limit]
print(f'recipes: {len(recs)}', flush=True)
prog = Progress(len(recs))
res = {'schema': 'qwentest-objective-match-batch-v1', 'n_recipes': len(recs), 'k_order': a.k_order,
       'settings': ['plain|raw', 'plain|layer_rms', 'contrast|raw', 'contrast|layer_rms'],
       'per_recipe': {}, 'summary': {}}

for r in recs:
    ex = (r.get('examples') or [{}])[0]
    if not ex.get('prompt') or not ex.get('output'):
        continue
    their = [(int(n['layer']), int(n['neuron']), float(n['dose'])) for n in (r['neurons'] or [])]
    if not their:
        continue
    ids_p = tok.apply_chat_template([{'role': 'user', 'content': ex['prompt']}],
                                   add_generation_prompt=True, enable_thinking=False, tokenize=True)
    ids_p = ids_p['input_ids'] if hasattr(ids_p, 'keys') else ids_p
    ids_t = tok(ex['output'], add_special_tokens=False).input_ids[:a.max_target_tokens]
    ids_b = tok(ex.get('base_output') or '', add_special_tokens=False).input_ids[:a.max_target_tokens]
    # one graph per objective (two forward+backward passes per recipe)
    with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
        Jp = mean_logp(ids_p, ids_t)
        g_plain = grads_of(Jp, cap)
    g_contrast = None
    if ids_b:
        with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
            Jc = mean_logp(ids_p, ids_t) - mean_logp(ids_p, ids_b)
            g_contrast = grads_of(Jc, cap)
    entry = {'name': r['name'], 'n_their': len(their), 'their': their}
    for tag, g in (('plain', g_plain), ('contrast', g_contrast)):
        if g is None:
            continue
        for norm in ('raw', 'layer_rms'):
            gg = g if norm == 'raw' else layer_equalise(g)
            order = greedy(gg, a.k_order)
            rank = {c: i + 1 for i, c in enumerate(order)}
            ts = {(l, n) for l, n, _ in their}
            entry[f'{tag}|{norm}'] = {
                'overlap_at_n': len(ts & set(order[:len(their)])),
                'jaccard_at_n': (len(ts & set(order[:len(their)])) /
                                 max(1, len(ts | set(order[:len(their)])))),
                'their_ranks': {f'L{l}N{n}': rank.get((l, n)) for l, n, _ in their},
                'n_in_top64': sum(1 for l, n, _ in their if (l, n) in rank),
                'our_layer_hist_top64': {str(l): sum(1 for x, _ in order if x == l) for l in layers},
            }
    # sign agreement uses the raw gradients (scaling by a positive number cannot change signs)
    entry['sign_agree_plain'] = sum(1 for l, n, d in their
                                    if np.sign(float(g_plain[l][n])) == np.sign(d))
    if g_contrast is not None:
        entry['sign_agree_contrast'] = sum(1 for l, n, d in their
                                           if np.sign(float(g_contrast[l][n])) == np.sign(d))
    res['per_recipe'][r['id']] = entry
    prog.step(f'{r["id"]} {r["name"][:28]}')

for tag in res['settings']:
    rows = [e[tag] for e in res['per_recipe'].values() if tag in e]
    if not rows:
        continue
    res['summary'][tag] = {
        'n': len(rows),
        'recipes_with_any_overlap_at_n': sum(1 for x in rows if x['overlap_at_n'] > 0) / len(rows),
        'mean_overlap_at_n': statistics.fmean(x['overlap_at_n'] for x in rows),
        'mean_jaccard_at_n': statistics.fmean(x['jaccard_at_n'] for x in rows),
        'mean_their_in_top64': statistics.fmean(x['n_in_top64'] for x in rows),
        'recipes_with_any_in_top64': sum(1 for x in rows if x['n_in_top64'] > 0) / len(rows),
    }
sa = [e['sign_agree_plain'] / e['n_their'] for e in res['per_recipe'].values()]
res['summary']['sign_agree_plain'] = statistics.fmean(sa)
sb = [e['sign_agree_contrast'] / e['n_their'] for e in res['per_recipe'].values()
      if 'sign_agree_contrast' in e]
if sb:
    res['summary']['sign_agree_contrast'] = statistics.fmean(sb)
Path(a.out).write_text(json.dumps(res, ensure_ascii=False))
print('\n=== summary ===')
for k, v in res['summary'].items():
    print(' ', k, v)
print('wrote', a.out)
