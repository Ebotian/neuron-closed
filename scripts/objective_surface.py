#!/usr/bin/env python3
"""Is the remaining gap the SAMPLING SURFACE? Average the same behaviour's gradient over several
prompt surfaces and watch the overlap with the community's coordinates.

Plan 2.39.1 located the gap: with the same objective (their published prompt + output) the SIGN of
our gradient matches their behaviourally-tuned dose sign 93.4% of the time, but their exact
coordinates only reach our top-64 in about half the recipes, at median rank 13. The candidate
explanation is that they searched a behaviour over ~25 prompts while we used the single published
example. If that is right, averaging our gradient over more surfaces of the same behaviour should
raise the exact overlap monotonically, without hurting the sign agreement.

Surfaces (all built from published text only, no new annotation):
  S1  their chat prompt -> their output
  S2  their own generic prompt "Write whatever you like." -> their output
  S3  their own CHAT prompt "Who are you?" -> their output
  S4  their chat prompt -> their published base_output -> their output
      (the behaviour placed AFTER the unsteered answer, a different context for the same text)

For each recipe we compute g_S1..g_S4 and then score the selections from the cumulative averages
(S1), (S1+S2)/2, (S1+S2+S3)/3, all four. Reported per depth: exact overlap at K = their neuron
count, count of their coordinates inside our top-64, median rank of those that make it, and the
sign agreement. No generation, no readout of ours.
"""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--pkg', default='/root/autodl-tmp/scripts/j2space')
ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
ap.add_argument('--recipes', default='/root/autodl-tmp/scripts/neuronary_141.json')
ap.add_argument('--model-name', default='Qwen/Qwen3.5-4B')
ap.add_argument('--limit', type=int, default=60)
ap.add_argument('--k-order', type=int, default=64)
ap.add_argument('--max-target-tokens', type=int, default=96)
ap.add_argument('--out', default='/root/autodl-tmp/runs/objective_surface.json')
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


def chat_ids(prompt):
    ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True,
                                  enable_thinking=False, tokenize=True)
    return ids['input_ids'] if hasattr(ids, 'keys') else ids


def mean_logp(ids_p, ids_t):
    ids = torch.tensor([list(ids_p) + list(ids_t)], device=dev)
    out = model(input_ids=ids, use_cache=False)
    lg = out.logits[0]
    lp = torch.log_softmax(lg[len(ids_p) - 1: len(ids_p) - 1 + len(ids_t)].float(), -1)
    return lp.gather(1, torch.tensor(ids_t, device=dev)[:, None]).mean()


def grads(obj, caps):
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


recs = [r for r in json.load(open(a.recipes)) if r['model'] == a.model_name and (r.get('neurons'))]
recs = [r for r in recs if (r.get('examples') or [{}])[0].get('prompt')][:a.limit]
print(f'recipes: {len(recs)}', flush=True)
prog = Progress(len(recs) * 4)
res = {'schema': 'qwentest-objective-surface-v1', 'n_recipes': len(recs), 'k_order': a.k_order,
       'depths': ['S1', 'S1+2', 'S1+2+3', 'S1+2+3+4'], 'per_recipe': {}, 'summary': {}}

for r in recs:
    ex = r['examples'][0]
    their = [(int(n['layer']), int(n['neuron']), float(n['dose'])) for n in r['neurons']]
    ts = {(l, n) for l, n, _ in their}
    ids_out = tok(ex['output'], add_special_tokens=False).input_ids[:a.max_target_tokens]
    base = ex.get('base_output') or ''
    ids_base = tok(base, add_special_tokens=False).input_ids[:a.max_target_tokens] if base else []
    surf = {
        'S1': chat_ids(ex['prompt']),
        'S2': chat_ids('Write whatever you like.'),
        'S3': chat_ids('Who are you?'),
        'S4': chat_ids(ex['prompt']) + ids_base,
    }
    G = {}
    for tag, ids_p in surf.items():
        with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
            J = mean_logp(ids_p, ids_out)
            G[tag] = grads(J, cap)
        prog.step(f'{r["id"]} {tag}')
    entry = {'name': r['name'], 'n_their': len(their), 'their': their}
    for depth, tags in (('S1', ['S1']), ('S1+2', ['S1', 'S2']), ('S1+2+3', ['S1', 'S2', 'S3']),
                        ('S1+2+3+4', ['S1', 'S2', 'S3', 'S4'])):
        g = {l: sum(G[t][l] for t in tags) / len(tags) for l in layers}
        order = greedy(g, a.k_order)
        rank = {c: i + 1 for i, c in enumerate(order)}
        entry[depth] = {
            'overlap_at_n': len(ts & set(order[:len(their)])),
            'n_in_top64': sum(1 for l, n, _ in their if (l, n) in rank),
            'ranks': [rank.get((l, n)) for l, n, _ in their],
            'sign_agree': sum(1 for l, n, d in their
                              if np.sign(float(g[l][n])) == np.sign(d)),
            'max_layer_share': max(sum(1 for x, _ in order if x == l)
                                   for l in layers) / max(1, len(order)),
        }
    res['per_recipe'][r['id']] = entry

for depth in res['depths']:
    rows = [e[depth] for e in res['per_recipe'].values() if depth in e]
    ranks = [r for x in rows for r in x['ranks'] if r]
    res['summary'][depth] = {
        'n': len(rows),
        'recipes_with_any_overlap_at_n': sum(1 for x in rows if x['overlap_at_n'] > 0) / len(rows),
        'mean_overlap_at_n': statistics.fmean(x['overlap_at_n'] for x in rows),
        'recipes_with_any_in_top64': sum(1 for x in rows if x['n_in_top64'] > 0) / len(rows),
        'mean_their_in_top64': statistics.fmean(x['n_in_top64'] for x in rows),
        'median_rank_when_present': statistics.median(ranks) if ranks else None,
        'mean_sign_agree_frac': statistics.fmean(x['sign_agree'] / max(1, len(x['ranks'])) for x in rows),
        'median_max_layer_share': statistics.median(x['max_layer_share'] for x in rows),
    }
Path(a.out).write_text(json.dumps(res, ensure_ascii=False))
print('\n=== summary by number of surfaces ===')
for k in res['depths']:
    print(' ', k, json.dumps(res['summary'][k], ensure_ascii=False))
print('wrote', a.out)
