#!/usr/bin/env python3
"""Plan 2.42 re-run: a thin shell that assembles modules and writes results. No measurement logic lives here.
FROZEN CRITERIA: doc/plan-2.42-frozen.md (phi = path J2 in nats; 1 batched backward per context; 2K pairs in chunks
of 8; leave-one-out weights so E[(phi-b_-i)s_i]=g; ICC from the Gram)."""
from __future__ import annotations
import argparse
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
from core.sampling import generate_rollouts
from estimators.phi import phi_of, slot_ids
from estimators.score import per_rollout_scores
from estimators.pathwise import pathwise_batch
from estimators.teacher import teacher_grad
from analysis.split_half import split_half_curve
from analysis.report import summarize


def main():
    ap = argparse.ArgumentParser()
    A = ap.add_argument
    A('--model', default='/root/autodl-tmp/Qwen3.5-4B'); A('--probes', default='/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')
    A('--n-ctx', type=int, default=8); A('--rollouts', type=int, default=32); A('--max-new', type=int, default=24)
    A('--temp', type=float, default=1.0); A('--seed', type=int, default=0); A('--save-gram', default=None)
    A('--pathwise', action='store_true'); A('--fp32', action='store_true'); A('--out', default='/root/autodl-tmp/runs/p_a2.json')
    a = ap.parse_args()
    dev = 'cuda:0'
    torch.manual_seed(a.seed)

    ctx = [r for r in map(json.loads, open(a.probes)) if r['trait'] == 'hostility'][:a.n_ctx]
    tok, model, lm = load_model_fast(a.model, dev)
    if a.fp32:
        model = model.float()
        lm = E.model_lm(model)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = column_norms_dev(lm, layers, dev)
    SC = dose_scale(p50, norms, layers)
    for q in model.parameters():
        q.requires_grad_(False)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    OFFS, _o = [], 0
    for l in layers:
        d_ = int(lm.layers[l].mlp.down_proj.weight.shape[1])
        OFFS.append((l, _o, d_)); _o += d_
    g_path = {l: torch.zeros(lm.layers[l].mlp.down_proj.weight.shape[1], dtype=torch.float64, device=dev) for l in layers}
    g_path_in = {l: torch.zeros_like(g_path[l]) for l in layers}
    g_teach = {l: torch.zeros_like(g_path[l]) for l in layers}
    g_pw = {l: torch.zeros_like(g_path[l]) for l in layers}
    rec = {'ctx': [], 'phi': {}}
    S_all, W_all, pw_snap, allphi = [], [], {}, []

    for ci, r in enumerate(ctx):
        pre, seqs = generate_rollouts(model, tok, r['prompt'], a.rollouts, a.max_new, a.temp, pad, dev,
                                      a.seed * 1000 + ci)
        with torch.inference_mode():
            ph = [float(x) for x in phi_of(model, tok, seqs, r, pad, dev)]
        allphi += ph
        rec['ctx'].append(r['id']); rec['phi'][r['id']] = ph

        S = per_rollout_scores(model, lm, layers, p50, norms, seqs, pad, dev, len(pre), scale=SC)
        K = len(seqs); psum = sum(ph)
        w_loo = [(x - (psum - x) / max(K - 1, 1)) / (K * len(ctx)) for x in ph]
        w_in = [(x - psum / K) / (K * len(ctx)) for x in ph]
        blk, wl, wi = S.double(), torch.tensor(w_loo, device=dev).unsqueeze(1), torch.tensor(w_in, device=dev).unsqueeze(1)
        for l, off, d_ in OFFS:
            col = blk[:, off:off + d_]
            g_path[l] += (wl * col).sum(0)
            g_path_in[l] += (wi * col).sum(0)
        S_all.append(S); W_all.append(w_loo)

        if a.pathwise:
            ip, ineg = slot_ids(tok, r['positive']), slot_ids(tok, r['negative'])
            wt = 1.0 / (K * len(ctx))
            items = [(sq, ip, +1.0, wt) for sq in seqs] + [(sq, ineg, -1.0, wt) for sq in seqs]
            acc = pathwise_batch(model, lm, layers, p50, norms, items, pad, dev, scale=SC, chunk=8)
            for l in layers:
                g_pw[l] += acc[l]
            pw_snap[ci] = {l: g_pw[l].clone() for l in layers}

        for key, sgn in (('positive', +1.0), ('negative', -1.0)):
            v = teacher_grad(model, lm, tok, layers, p50, norms, r['prompt'],
                             r[key] if isinstance(r[key], str) else r[key][0], sgn, 1.0 / len(ctx), dev, scale=SC)
            for l in layers:
                g_teach[l] += v[l]

    rec.update(summarize(layers, g_path, g_teach, g_path_in, S_all, W_all, a.rollouts, a.seed, allphi,
                         pathwise=g_pw if a.pathwise else None))
    if a.pathwise:
        snaps = [pw_snap[ci] for ci in sorted(pw_snap)]
        inc = [snaps[0]] + [{l: snaps[i][l] - snaps[i - 1][l] for l in layers} for i in range(1, len(snaps))]
        rec['pw_additivity'] = {'contexts': len(snaps),
                                'max_dev_from_sum_of_increments': max(float((g_pw[l] - sum(x[l] for x in inc)).abs().max()) for l in layers),
                                'max_dev_from_last_increment': max(float((g_pw[l] - inc[-1][l]).abs().max()) for l in layers)}
    if a.save_gram:
        json.dump({'G': rec['gram'], 'group_sizes': rec['group_sizes'], 'k': a.rollouts, 'seed': a.seed},
                  open(a.save_gram, 'w'))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, 'w'))
    icc = rec['ICC']
    print('ICC  MSw=%.6f ICC=%.6f %s' % (icc['tr_sigma'], icc['icc'], [round(x, 4) for x in icc['icc_interval']]), flush=True)
    if a.pathwise:
        print('C2 cos(pw,teach)=%+.4f |pw|=%.2f ratio=%.3f' % (
            rec['C2_split']['cos_pathwise_teach'], rec['C2_split']['norm_pathwise'],
            rec['C2_split']['ratio_score_over_pathwise']), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
