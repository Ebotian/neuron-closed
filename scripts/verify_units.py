#!/usr/bin/env python3
"""Per-value verification of the units refactor (not a cosine check).

COMPARES
  A  the inline expression used by the scripts before the refactor:
       {l: (p50[l] / norms[l].clamp(min=1e-12)).double()}
  B  core.units.dose_scale(p50, norms, layers)
     -> must be BIT-IDENTICAL (torch.equal), layer by layer.
  C  engine.column_norms (CPU reduction) vs core.units.column_norms_dev (GPU reduction)
     -> reported as a max relative difference; they need not be bit-identical (different reductions), but the
        SIZE of the difference must be recorded because it enters every budget.

FROZEN CRITERION: A and B must be exactly equal for all 32 layers; C is reported, not gated.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
import engine as E
import units as U


def main():
    dev = 'cuda:0'
    tok, model, lm = E.load_model('/root/autodl-tmp/Qwen3.5-4B', dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms_cpu = E.column_norms(lm, layers)
    norms_gpu = U.column_norms_dev(lm, layers, dev)

    old = {l: (p50[l] / norms_cpu[l].clamp(min=1e-12)).double() for l in layers}
    new = U.dose_scale(p50, norms_cpu, layers)
    bad = [l for l in layers if not torch.equal(old[l], new[l])]
    print(f'A vs B: bit-identical for {len(layers) - len(bad)}/{len(layers)} layers; mismatches={bad}')
    worst = max(float((norms_gpu[l] - norms_cpu[l]).abs().max()
                      / norms_cpu[l].clamp(min=1e-12).max()) for l in layers)
    print(f'C  CPU vs GPU column norms: max relative difference = {worst:.3e}')
    # the scale derived from the GPU norms must differ by exactly that much, no more
    new_gpu = U.dose_scale(p50, norms_gpu, layers)
    dr = max(float((new_gpu[l] - old[l]).abs().max() / old[l].abs().max()) for l in layers)
    print(f'C  resulting dose-scale drift = {dr:.3e}')
    ok = not bad
    print('UNITS-VERIFY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
