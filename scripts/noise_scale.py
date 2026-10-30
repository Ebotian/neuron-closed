#!/usr/bin/env python3
"""Noise scale: how many rollouts would be needed for the score gradient to be measurable.

FROZEN: the sample requirement is K = 2*kappa, not kappa (r(K) = K/(K+2κ)); the independent unit is the CONTEXT, so a
different rollout seed gives independent rollouts inside the same contexts and cannot settle context-level
reproducibility. Thin shell: generation, phi, scores and the ICC all live in modules.
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
from core.sampling import generate_rollouts
from estimators.phi import phi_of
from estimators.score import per_rollout_scores
from analysis.report import summarize


def main():
    ap = argparse.ArgumentParser()
    A = ap.add_argument
    A('--model', default='/root/autodl-tmp/Qwen3.5-4B'); A('--probes', default='/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')
    A('--n-ctx', type=int, default=8); A('--rollouts', type=int, default=32); A('--max-new', type=int, default=24)
    A('--temp', type=float, default=1.0); A('--seed', type=int, default=0); A('--max-ctx-per-kappa', type=int, default=0)
    A('--out', default='/root/autodl-tmp/runs/noise_scale.json')
    a = ap.parse_args()
    dev = 'cuda:0'
    torch.manual_seed(a.seed)
    ctx = [r for r in map(json.loads, open(a.probes)) if r['trait'] == 'hostility'][:a.n_ctx]
    tok, model, lm = E.load_model(a.model, dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = column_norms_dev(lm, layers, dev)
    SC = dose_scale(p50, norms, layers)
    for q in model.parameters():
        q.requires_grad_(False)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    S_all, W_all, phi_all, rec = [], [], [], {'ctx': [], 'phi': {}}
    for ci, r in enumerate(ctx):
        pre, seqs = generate_rollouts(model, tok, r['prompt'], a.rollouts, a.max_new, a.temp, pad, dev, a.seed * 1000 + ci)
        with torch.inference_mode():
            phi = [float(x) for x in phi_of(model, tok, seqs, r, pad, dev)]
        phi_all += phi
        rec['ctx'].append(r['id']); rec['phi'][r['id']] = phi
        K, psum = len(seqs), sum(phi)
        S_all.append(per_rollout_scores(model, lm, layers, p50, norms, seqs, pad, dev, len(pre), scale=SC))
        W_all.append([(x - (psum - x) / max(K - 1, 1)) / (K * len(ctx)) for x in phi])
        print(f'  [{ci}] {r["id"]} phi mean={psum/K:+.3f}', flush=True)

    rec.update(summarize(layers, None, None, None, S_all, W_all, a.rollouts, a.seed, phi_all))
    rec.pop('gram', None); rec.pop('group_sizes', None)
    icc = rec['ICC']
    rec['K_for_r_half_via_2kappa'] = None if icc['g_sq'] <= 0 else 2 * (icc['tr_sigma'] / icc['g_sq'])
    print('trSigma=%.4f %s  g_sq=%.5f  ICC=%.6f %s  K(r=0.5)=2*kappa=%s' % (
        icc['tr_sigma'], icc['tr_sigma_interval'], icc['g_sq'], icc['icc'],
        icc['icc_interval'], rec['K_for_r_half_via_2kappa']), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, 'w'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
