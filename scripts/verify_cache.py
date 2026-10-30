#!/usr/bin/env python3
"""The p50 / column-norm cache must be invisible: stored values == a fresh computation, exactly.

WHY THIS EXISTS. With `strict=True` inside the cache the comparison ran on EVERY use, which made the cache pointless (setup
stayed at 12.6 s on the benchmark); it now defaults to False and the check lives here instead. Run it after any change to
`core/units.py` or to the model snapshot, because a stale cache is otherwise undetectable downstream - nothing can tell
where a value came from.

FROZEN CRITERION: max |cached - fresh| == 0.0 over all 32 layers, for both quantities.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
_h = Path(__file__).resolve().parent
sys.path.insert(0, str(_h))
for sub in ('core', 'estimators', 'analysis'):
    sys.path.insert(0, str(_h / sub))
import engine as E                                    # noqa: E402
from core.units import column_norms_dev               # noqa: E402
from core.loading import load_model_fast              # noqa: E402


def compare(name, blob_path, compute, layers):
    p = Path(blob_path)
    if not p.exists():
        print(f'{name}: no cache at {p.name} - skipping')
        return True
    vals = torch.load(p)['values']
    if torch.load(p)['key'] != ','.join(str(l) for l in layers):
        print(f'{name}: FAIL key mismatch in {p.name}')
        return False
    fresh = compute()
    worst = max(float((torch.as_tensor(vals[l]).double() - torch.as_tensor(fresh[l]).double()).abs().max())
                for l in layers)
    print(f'{name}: max |cached - fresh| = {worst:.3e} over {len(layers)} layers')
    return worst == 0.0


def main():
    dev = 'cuda:0'
    tok, model, lm = load_model_fast('/dev/shm/qwen', dev)
    layers = list(range(32))
    C = '/root/autodl-tmp/runs/cache_'
    ok = compare('p50', f'{C}p50_qwen.pt', lambda: E.estimate_p50(tok, model, lm, layers, dev), layers)
    ok &= compare('column norms', f'{C}cn_qwen.pt', lambda: column_norms_dev(lm, layers, dev), layers)
    print('CACHE-VERIFY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
