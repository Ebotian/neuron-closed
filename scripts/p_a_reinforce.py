#!/usr/bin/env python3
"""Plan 2.40 re-run: the judge-based behaviour gradient, as a thin shell.

FROZEN CRITERIA: doc/plan-2.40-frozen.md. C0 = the judge has dynamic range on base rollouts (0 < mean phi < 0.5 and
fewer than half above 0.5); C1 = reproducibility of the score-function gradient; the ICC from the Gram is the
bias-immune reliability. No measurement logic lives here - see core/ and estimators/.
"""
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
from estimators.judge import load_gauges, phi_judge
from estimators.score import per_rollout_scores
from estimators.teacher import teacher_grad
from analysis.report import summarize


def main():
    ap = argparse.ArgumentParser()
    A = ap.add_argument
    A('--model', default='/root/autodl-tmp/Qwen3.5-4B'); A('--probes', default='/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')
    A('--gauges', default='/root/autodl-tmp/gauges'); A('--n-ctx', type=int, default=8); A('--rollouts', type=int, default=8)
    A('--max-new', type=int, default=32); A('--temp', type=float, default=1.0); A('--seed', type=int, default=0)
    A('--out', default='/root/autodl-tmp/runs/p_a.json')
    a = ap.parse_args()
    dev = 'cuda:0'
    torch.manual_seed(a.seed)

    ctx = [r for r in map(json.loads, open(a.probes)) if r['trait'] == 'hostility'][:a.n_ctx]
    tok, model, lm = load_model_fast(a.model, dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = column_norms_dev(lm, layers, dev)
    SC = dose_scale(p50, norms, layers)
    for q in model.parameters():
        q.requires_grad_(False)
    gauges = load_gauges(a.gauges, dev)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    g_re = {l: torch.zeros(lm.layers[l].mlp.down_proj.weight.shape[1], dtype=torch.float64, device=dev) for l in layers}
    g_teach = {l: torch.zeros_like(g_re[l]) for l in layers}
    rec, S_all, W_all, allphi = {'ctx': [], 'phi': {}}, [], [], []

    for ci, r in enumerate(ctx):
        pre, seqs = generate_rollouts(model, tok, r['prompt'], a.rollouts, a.max_new, a.temp, pad, dev, a.seed * 1000 + ci)
        texts = [tok.decode(s[len(pre):], skip_special_tokens=True) for s in seqs]
        phi, ph = phi_judge(gauges, texts, dev)
        allphi += phi
        rec['ctx'].append(r['id']); rec['phi'][r['id']] = ph
        K, psum = len(seqs), sum(phi)
        S = per_rollout_scores(model, lm, layers, p50, norms, seqs, pad, dev, len(pre), scale=SC)
        w_loo = [(x - (psum - x) / max(K - 1, 1)) / (K * len(ctx)) for x in phi]
        blk, wl = S.double(), torch.tensor(w_loo, device=dev).unsqueeze(1)
        off = 0
        for l in layers:
            d_ = int(lm.layers[l].mlp.down_proj.weight.shape[1])
            g_re[l] += (wl * blk[:, off:off + d_]).sum(0)
            off += d_
        S_all.append(S); W_all.append(w_loo)
        for key, sgn in (('positive', +1.0), ('negative', -1.0)):
            v = teacher_grad(model, lm, tok, layers, p50, norms, r['prompt'],
                             r[key] if isinstance(r[key], str) else r[key][0], sgn, 1.0 / len(ctx), dev, scale=SC)
            for l in layers:
                g_teach[l] += v[l]
        print(f'  [{ci}] {r["id"]} K={K} phi mean={psum/K:+.4f}', flush=True)

    mu = sum(allphi) / len(allphi)
    c0 = {'mean_phi': mu, 'frac_gt_0.5': sum(1 for x in allphi if x > 0.5) / len(allphi), 'n_rollouts': len(allphi)}
    print(f'C0 mean_phi={mu:.4f} frac>0.5={c0["frac_gt_0.5"]:.4f} n={len(allphi)}', flush=True)
    if not (0.0 < mu < 0.5 and c0['frac_gt_0.5'] < 0.5):
        json.dump({'C0': c0}, open(a.out, 'w'))
        print('C0 VIOLATED -> stop', flush=True)
        return 2
    rec.update(summarize(layers, g_re, g_teach, g_re, S_all, W_all, a.rollouts, a.seed, allphi))
    rec['C0'] = c0
    rec.pop('gram', None); rec.pop('group_sizes', None)
    icc = rec['ICC']
    print('ICC  MSw=%.4f ICC=%.6f %s' % (icc['tr_sigma'], icc['icc'], [round(x, 4) for x in icc['icc_interval']]), flush=True)
    print('C1   cos(score,teacher)=%+.4f' % rec['C1p']['cos_path_teach'], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, 'w'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
