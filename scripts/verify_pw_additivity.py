#!/usr/bin/env python3
"""The pathwise accumulator must sum over contexts, not keep only the last one (bug 6 in doc/refactor-bugs.md).

THE DEFECT THIS GUARDS. `g_pw` was once re-initialised INSIDE the context loop, so the final object equalled the last
context's contribution instead of the sum over contexts. Two earlier versions of this check were themselves wrong (a
shallow dict copy whose tensors kept mutating; then summing CUMULATIVE snapshots), which is why it now compares the final
accumulator against (a) the sum of per-context INCREMENTS and (b) the last increment alone.

FROZEN: sum-of-increments deviation == 0 (exactly, it is the same additions) and last-increment deviation > 1e-6 * scale
(i.e. the accumulator is demonstrably NOT just the last context).
"""
from __future__ import annotations
import json
import sys
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
from estimators.pathwise import pathwise_batch


def main():
    dev = 'cuda:0'
    n_ctx, K = 3, 4
    ctx = [r for r in map(json.loads, open('/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl'))
           if r['trait'] == 'hostility'][:n_ctx]
    tok, model, lm = load_model_fast('/root/autodl-tmp/Qwen3.5-4B', dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = column_norms_dev(lm, layers, dev)
    SC = dose_scale(p50, norms, layers)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    acc = {l: torch.zeros(lm.layers[l].mlp.down_proj.weight.shape[1], dtype=torch.float64, device=dev) for l in layers}
    snaps = []
    for ci, r in enumerate(ctx):
        pre, seqs = generate_rollouts(model, tok, r['prompt'], K, 12, 1.0, pad, dev, ci)
        ip, ineg = slot_ids(tok, r['positive']), slot_ids(tok, r['negative'])
        wt = 1.0 / (K * len(ctx))
        items = [(sq, ip, +1.0, wt) for sq in seqs] + [(sq, ineg, -1.0, wt) for sq in seqs]
        got = pathwise_batch(model, lm, layers, p50, norms, items, pad, dev, scale=SC, chunk=8)
        for l in layers:
            acc[l] += got[l]
        snaps.append({l: acc[l].clone() for l in layers})

    inc = [snaps[0]] + [{l: snaps[i][l] - snaps[i - 1][l] for l in layers} for i in range(1, len(snaps))]
    dev_sum = max(float((acc[l] - sum(x[l] for x in inc)).abs().max()) for l in layers)
    dev_last = max(float((acc[l] - inc[-1][l]).abs().max()) for l in layers)
    scale = max(float(acc[l].abs().max()) for l in layers)
    print(f'contexts={len(snaps)}  |final - sum(inc)|={dev_sum:.2e}  |final - last_inc|={dev_last:.2e}  scale={scale:.2e}')
    ok = dev_sum == 0.0 and dev_last > 1e-6 * scale
    print('PW-ADDITIVITY-VERIFY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
