#!/usr/bin/env python3
"""Same OBJECTIVE, our ALGORITHM: take a published recipe (their prompt + their steered output),
define J = mean teacher-forced log p(their output | their prompt), and run our closed form on it.

Why the objective has to be their TEXT rather than their KL: their published per-coordinate
readout is KL(steered||base), which has a zero gradient at u=0 (its curvature is the Fisher we
measured in 2.34/2.35), so it cannot define a direction. Their recipes, by contrast, are signed:
a prompt and the output the behaviour produced.

Output, per recipe: their (coordinate, dose) rows, our order at the same K and at K=64, our
doses normalised to THEIR aggregate budget sqrt(sum dose^2), ranks of their coordinates in our
order, Jaccard of the two sets, and (for coordinates we do not select) the marginal dQ that
would come from adding them. No generation, no readout of ours.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--pkg', default='/root/autodl-tmp/scripts/j2space')
ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
ap.add_argument('--recipes', default='/root/autodl-tmp/scripts/neuronary_141.json')
ap.add_argument('--ids', default='qdzzppqy')
ap.add_argument('--k64', type=int, default=64)
ap.add_argument('--max-target-tokens', type=int, default=192)
ap.add_argument('--contrast', action='store_true',
                help='J = mean logp(their output) - mean logp(their published BASE output): the '
                     'contrast removes the generic "make this text likely" plumbing direction')
ap.add_argument('--out', default='/root/autodl-tmp/runs/objective_match.json')
a = ap.parse_args()

sys.path.insert(0, a.pkg)
from engine import capture_mlp_inputs, column_norms, estimate_p50, load_model, score_for_dose
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

layers = list(range(32))
tok, model, lm = load_model(a.model)
dev = 'cuda:0'
dims = {l: int(lm.layers[l].mlp.down_proj.weight.shape[1]) for l in layers}
norms = column_norms(lm, layers)
p50 = estimate_p50(tok, model, lm, layers, device=dev)

M, Wn = {}, {}
for l in layers:
    W = lm.layers[l].mlp.down_proj.weight.detach().float().T.to(dev)
    W = W / W.norm(dim=1, keepdim=True).clamp(min=1e-12)
    M[l] = W @ W.T
    Wn[l] = lm.layers[l].mlp.down_proj.weight.detach().float().norm(dim=0).cpu()
    del W
torch.cuda.empty_cache()
torch.set_num_threads(1)
print(f'M built on GPU: {sum(m.numel() * 4 for m in M.values()) / 1e9:.1f} GB', flush=True)


def _mean_logp(ids_p, ids_t):
    ids = torch.tensor([list(ids_p) + list(ids_t)], device=dev)
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits[0]
    lp = torch.log_softmax(logits[len(ids_p) - 1: len(ids_p) - 1 + len(ids_t)].float(), -1)
    return lp.gather(1, torch.tensor(ids_t, device=dev)[:, None]).mean()


def _g_of_scalar(obj):
    grads = torch.autograd.grad(obj, [cap[l] for l in layers])
    g = {}
    for l, gr in zip(layers, grads):
        sc = gr.detach().double().sum(dim=tuple(range(gr.ndim - 1)))
        g[l] = (sc * (p50[l] / norms[l].to(sc.device).double().clamp(min=1e-12))).cpu()
    return g


def gradient_of_J(prompt, target, max_tokens, base_target=None):
    """g (per layer, dose units) of J = mean log p(target | prompt), optionally CONTRASTED
    against a base continuation (their published unsteered output)."""
    ids_p = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                   add_generation_prompt=True, enable_thinking=False,
                                   tokenize=True)
    ids_p = ids_p['input_ids'] if hasattr(ids_p, 'keys') else ids_p
    ids_t = tok(target, add_special_tokens=False).input_ids[:max_tokens]
    with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
        J = _mean_logp(ids_p, ids_t)
        if base_target is not None:
            ids_b = tok(base_target, add_special_tokens=False).input_ids[:max_tokens]
            J = J - _mean_logp(ids_p, ids_b)
        g = _g_of_scalar(J)
    return g, float(J.detach()), len(ids_t)


def greedy(M, g, K):
    S = {l: [] for l in layers}
    P = {l: torch.zeros(0, 0, dtype=torch.float64, device=dev) for l in layers}
    y = {l: torch.zeros(0, dtype=torch.float64, device=dev) for l in layers}
    diag = {l: torch.diagonal(M[l]).double() for l in layers}
    order = []
    for _ in range(K):
        best = (-float('inf'), None)
        for l in layers:
            k = len(S[l])
            if k >= dims[l]:
                continue
            if k:
                v = M[l].index_select(1, torch.tensor(S[l], device=dev)).double()
                gain = (g[l].to(dev) - v @ y[l]) ** 2 / (diag[l] - ((v @ P[l]) * v).sum(1)).clamp(min=1e-12)
                gain[torch.tensor(S[l], device=dev)] = -float('inf')
            else:
                gain = g[l].to(dev) ** 2 / diag[l]
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
        y[l] = P[l] @ g[l].to(dev)[S[l]]
        order.append((l, j))
    return order


def doses(order, g, b):
    per = {}
    for l, j in order:
        per.setdefault(l, []).append(j)
    u = {}
    for l, js in per.items():
        idx = torch.tensor(js, device=dev)
        Ms = M[l].index_select(0, idx).index_select(1, idx).double()
        gl = g[l].to(dev).index_select(0, idx)
        u[l] = torch.linalg.solve(Ms, gl)
    b0 = math.sqrt(sum(float(g[l].to(dev).index_select(0, torch.tensor(js, device=dev)) @ u[l])
                       for l, js in per.items()))
    s = b / b0 if b0 > 0 else 0.0
    return {l: v * s for l, v in u.items()}, per, b0


recs = {r['id']: r for r in json.load(open(a.recipes))}
out = {'schema': 'qwentest-objective-match-v1', 'k64': a.k64, 'recipes': {}}
for rid in a.ids.split(','):
    r = recs[rid]
    their = [(int(n['layer']), int(n['neuron']), float(n['dose'])) for n in r['neurons']]
    b_theirs = math.sqrt(sum(d * d for _, _, d in their))
    ex = r['examples'][0]
    g, J, ntok = gradient_of_J(ex['prompt'], ex['output'], a.max_target_tokens,
                               ex.get('base_output') if a.contrast else None)
    order4 = greedy(M, g, len(their))
    order64 = greedy(M, g, a.k64)
    u4, per4, b0_4 = doses(order4, g, b_theirs)
    u64, per64, b0_64 = doses(order64, g, b_theirs)
    theirs_set = {(l, n) for l, n, _ in their}
    rk = {c: i + 1 for i, c in enumerate(order64)}
    rows = []
    for (l, n, d) in their:
        in64 = (l, n) in rk
        ouru = None
        if in64:
            j = per64[l].index(n)
            ouru = float(u64[l][j])
        rows.append({'layer': l, 'neuron': n, 'their_dose': d, 'our_rank_K64': rk.get((l, n)),
                     'our_dose_at_same_budget': ouru,
                     'our_alpha': None if ouru is None else ouru * p50[l] / float(Wn[l][n]),
                     'their_alpha': d * p50[l] / float(Wn[l][n])})
    out['recipes'][rid] = {
        'name': r['name'], 'prompt': ex['prompt'], 'J_mean_logp': J, 'n_target_tokens': ntok,
        'their_neurons': their, 'their_budget': b_theirs,
        'our_K_same_size': [[l, j] for l, j in order4],
        'jaccard_topK_vs_theirs': len(set(map(tuple, order4)) & theirs_set) / len(set(map(tuple, order4)) | theirs_set),
        'overlap_count': len(set(map(tuple, order4)) & theirs_set),
        'their_rows': rows}
    print(f'\n{rid} {r["name"]}')
    print(f'  J = {J:.4f} over {ntok} target tokens; their budget b = {b_theirs:.3f}')
    print(f'  their set: {sorted(theirs_set)}')
    print(f'  our K={len(their)} set: {[tuple(x) for x in order4]}  overlap {len(set(map(tuple, order4)) & theirs_set)}/{len(their)}')
    for row in rows:
        print(f'   L{row["layer"]}N{row["neuron"]}: theirs d={row["their_dose"]:+.2f} '
              f'(alpha {row["their_alpha"]:+.2f}) | ours rank {row["our_rank_K64"]} '
              f'dose {row["our_dose_at_same_budget"]} '
              f'alpha {row["our_alpha"]}')

Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
print('\nwrote', a.out)
