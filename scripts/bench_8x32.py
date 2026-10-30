#!/usr/bin/env python3
"""8x32 throughput benchmark: load once, then two identical measurement passes.

DEFINITION, stated because it decides the number: the benchmark is the MEASUREMENT PASS, not the model load. The load
reads 8.8 GB from disk and sits at 0% GPU for 12-18 s; charging it to a throughput measurement measures the disk, not
the pipeline. It is therefore timed and reported separately (`setup_s`), and the cold-start figures from the earlier
whole-run measurement are kept in results/perf_8x32.json for comparison - both numbers, not one instead of the other.

Each pass prints PASS_START/PASS_END epoch stamps so an external nvidia-smi sampler can be aligned to the exact window.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
_h = Path(__file__).resolve().parent
sys.path.insert(0, str(_h))
for sub in ('core', 'estimators', 'analysis'):
    sys.path.insert(0, str(_h / sub))
import engine as E
from core.loading import load_model_fast
from core.units import column_norms_dev, dose_scale
from core.sampling import generate_rollouts
from estimators.phi import phi_of, slot_ids
from estimators.score import per_rollout_scores
from estimators.pathwise import pathwise_batch
from estimators.teacher import teacher_grad
from analysis.report import summarize


def pass_once(model, lm, tok, ctx, layers, p50, norms, SC, pad, dev, rollouts, max_new, temp, seed):
    S_all, W_all, phi_all = [], [], []
    g_path = {l: torch.zeros(lm.layers[l].mlp.down_proj.weight.shape[1], dtype=torch.float64, device=dev) for l in layers}
    g_teach = {l: torch.zeros_like(g_path[l]) for l in layers}
    g_pw = {l: torch.zeros_like(g_path[l]) for l in layers}
    for ci, r in enumerate(ctx):
        pre, seqs = generate_rollouts(model, tok, r['prompt'], rollouts, max_new, temp, pad, dev, seed * 1000 + ci)
        with torch.inference_mode():
            phi = [float(x) for x in phi_of(model, tok, seqs, r, pad, dev)]
        phi_all += phi
        K, psum = len(seqs), sum(phi)
        S_all.append(per_rollout_scores(model, lm, layers, p50, norms, seqs, pad, dev, len(pre), scale=SC))
        W_all.append([(x - (psum - x) / max(K - 1, 1)) / (K * len(ctx)) for x in phi])
        ip, ineg = slot_ids(tok, r['positive']), slot_ids(tok, r['negative'])
        wt = 1.0 / (K * len(ctx))
        items = [(sq, ip, +1.0, wt) for sq in seqs] + [(sq, ineg, -1.0, wt) for sq in seqs]
        acc = pathwise_batch(model, lm, layers, p50, norms, items, pad, dev, scale=SC, chunk=8)
        for l in layers:
            g_pw[l] += acc[l]
        for key, sgn in (('positive', +1.0), ('negative', -1.0)):
            v = teacher_grad(model, lm, tok, layers, p50, norms, r['prompt'],
                             r[key] if isinstance(r[key], str) else r[key][0], sgn, 1.0 / len(ctx), dev, scale=SC)
            for l in layers:
                g_teach[l] += v[l]
    return summarize(layers, g_path, g_teach, g_path, S_all, W_all, rollouts, seed, phi_all, pathwise=g_pw)


def main():
    ap = argparse.ArgumentParser()
    A = ap.add_argument
    A('--model', default='/root/autodl-tmp/Qwen3.5-4B'); A('--probes', default='/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')
    A('--n-ctx', type=int, default=8); A('--rollouts', type=int, default=32); A('--max-new', type=int, default=24)
    A('--temp', type=float, default=1.0); A('--seed', type=int, default=0); A('--out', default='/root/autodl-tmp/runs/bench.json')
    a = ap.parse_args()
    dev = 'cuda:0'
    ctx = [r for r in map(json.loads, open(a.probes)) if r['trait'] == 'hostility'][:a.n_ctx]
    t0 = time.time()
    tok, model, lm = load_model_fast(a.model, dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = column_norms_dev(lm, layers, dev)
    SC = dose_scale(p50, norms, layers)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f'SETUP_S {time.time() - t0:.1f}', flush=True)

    passes = []
    for i in (1, 2):
        t = time.time()
        print(f'PASS_START {i} {t:.3f}', flush=True)
        rec = pass_once(model, lm, tok, ctx, layers, p50, norms, SC, pad, dev, a.rollouts, a.max_new, a.temp, a.seed)
        dt = time.time() - t
        print(f'PASS_END {i} {time.time():.3f}', flush=True)
        icc = rec['ICC']
        passes.append({'pass': i, 'wall_s': round(dt, 1), 'MS_w': icc['tr_sigma'], 'ICC': icc['icc'],
                       'norm_pathwise': rec['C2_split']['norm_pathwise']})
        print(f'  pass {i}: {dt:.1f}s  MS_w={icc["tr_sigma"]:.6f}  ICC={icc["icc"]:.6f}  |pw|={rec["C2_split"]["norm_pathwise"]:.2f}', flush=True)
    json.dump({'setup_s': round(time.time() - t0, 1), 'passes': passes}, open(a.out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
