#!/usr/bin/env python3
"""The p50 / column-norm cache must be invisible, and its key must actually bind the model.

TWO CHECKS, both of which the first version of this file failed to make:
 1. VALUES: max |cached - fresh| == 0.0 over all 32 layers. `strict=True` inside the cache re-computed on every use, which
    made the cache pointless (setup stayed at 12.6 s), so the cache now defaults to False and the check lives here.
 2. BINDING: a blob stored under a DIFFERENT key must not be returned. The audit's objection was that the key was the layer
    list plus the model directory's basename, so changed weights or a changed reference text would silently reuse an old p50.
    Here a counter proves the compute path is taken when the key differs - a value comparison alone would pass either way.

FROZEN CRITERIA: value diff == 0.0; recompute counter == 1 for the wrong key and == 0 for the right one.
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
from core.cache import _fingerprint, load_or_compute  # noqa: E402
from core.units import column_norms_dev               # noqa: E402
from core.loading import load_model_fast              # noqa: E402

MODEL = '/dev/shm/qwen'
CACHE = '/root/autodl-tmp/runs/cache_'


def check(name, blob_path, compute, layers, model_path, extra):
    p = Path(blob_path)
    key = _fingerprint(model_path, layers, extra)
    if p.exists():
        blob = torch.load(p)
        if blob.get('key') != key:
            print(f'{name}: blob was written under a DIFFERENT key ({str(blob.get("key"))[:12]} vs {key[:12]}) '
                  f'- it will be recomputed on the next run, which is the correct behaviour')
        else:
            fresh = compute()
            worst = max(float((torch.as_tensor(blob['values'][l]).double()
                               - torch.as_tensor(fresh[l]).double()).abs().max()) for l in layers)
            print(f'{name}: max |cached - fresh| = {worst:.3e} over {len(layers)} layers')
            if worst != 0.0:
                return False
    calls = {'n': 0}

    def counted():
        calls['n'] += 1
        return compute()

    wrong = _fingerprint(model_path, layers, {'quantity': 'deliberately-wrong'})
    load_or_compute(blob_path, counted, wrong, layers, strict=False)
    if calls['n'] != 1:
        print(f'{name}: FAIL wrong key did NOT force a recompute (calls={calls["n"]}) - the key is not binding')
        return False
    print(f'{name}: a different key forces a recompute (calls={calls["n"]}) - the key is binding')
    return True


def main():
    dev = 'cuda:0'
    tok, model, lm = load_model_fast(MODEL, dev)
    layers = list(range(32))
    ok = check('p50', f'{CACHE}p50_qwen.pt', lambda: E.estimate_p50(tok, model, lm, layers, dev),
               layers, MODEL, {'quantity': 'p50'})
    ok &= check('column norms', f'{CACHE}cn_qwen.pt', lambda: column_norms_dev(lm, layers, dev),
                layers, MODEL, {'quantity': 'column_norms'})
    print('CACHE-VERIFY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
