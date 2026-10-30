#!/usr/bin/env python3
"""Keep the frozen model resident and run tasks against it - the reason consecutive tasks can be fast.

WHY. Every script here loaded 8.5 GiB of weights, ran, and exited, so a rented box that runs ten tasks pays the load ten
times. The monitoring traces show it plainly: GPU memory climbs to ~28 GiB and drops back to 0 on each run. This worker
loads once, prints READY, then executes one task per stdin line and answers with one JSON line per task. Nothing about a
task changes: the pass is `bench_8x32.pass_once`, the same function the single-shot benchmark calls.

PROTOCOL. stdin: {"n_ctx":8,"rollouts":32,"seed":0} per line (all fields optional). stdout: a line starting 'PASS_START'
and 'PASS_END' with epoch stamps around each task, then 'RESULT <json>'. The stamps are what an external sampler aligns to.
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
import engine as E                                             # noqa: E402
from core.cache import p50_cached, column_norms_cached          # noqa: E402
from core.loading import load_model_fast                       # noqa: E402
from core.units import dose_scale                              # noqa: E402
from bench_8x32 import pass_once                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    A = ap.add_argument
    A('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    A('--probes', default='/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')
    A('--n-ctx', type=int, default=8)
    A('--rollouts', type=int, default=32)
    A('--max-new', type=int, default=24)
    A('--temp', type=float, default=1.0)
    a = ap.parse_args()
    dev = 'cuda:0'
    t0 = time.time()
    tok, model, lm = load_model_fast(a.model, dev)
    layers = list(range(32))
    C, tag = '/root/autodl-tmp/runs/cache_', Path(a.model).name
    p50 = p50_cached(tok, model, lm, layers, dev, f'{C}p50_{tag}.pt', model_path=a.model)
    norms = column_norms_cached(lm, layers, dev, f'{C}cn_{tag}.pt', model_path=a.model)
    SC = dose_scale(p50, norms, layers)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    probes = [r for r in map(json.loads, open(a.probes)) if r['trait'] == 'hostility']
    print(f'READY {time.time():.3f} load_s={time.time() - t0:.1f}', flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line in ('quit', 'exit'):
            break
        job = json.loads(line)
        ctx = probes[:int(job.get('n_ctx', a.n_ctx))]
        t = time.time()
        print(f'PASS_START {job.get("pass", 1)} {t:.3f}', flush=True)
        rec = pass_once(model, lm, tok, ctx, layers, p50, norms, SC, pad, dev,
                        int(job.get('rollouts', a.rollouts)), a.max_new, a.temp, int(job.get('seed', 0)))
        print(f'PASS_END {job.get("pass", 1)} {time.time():.3f}', flush=True)
        icc = rec['ICC']
        print('RESULT ' + json.dumps({'pass': job.get('pass', 1), 'wall_s': round(time.time() - t, 1),
                                      'MS_w': icc['tr_sigma'], 'ICC': icc['icc'],
                                      'icc_interval': icc['icc_interval'],
                                      'norm_pathwise': rec['C2_split']['norm_pathwise'],
                                      'cos_pathwise_teach': rec['C2_split']['cos_pathwise_teach']}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
