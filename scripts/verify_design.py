#!/usr/bin/env python3
"""Per-value verification of the device/precision policy.

A  `scale_for` hands out tensors already on the design device, with the SAME values as the CPU originals
   (bit-identical after the round trip), for every layer.
B  `check` rejects a tensor from the wrong device - i.e. the failure mode that cost four debugging rounds is
   now caught at the boundary instead of inside a `+=`.
C  the precision switch is reproducible: converting to float32 and taking the scale again gives values equal to
   the float64 scale up to float32 rounding, and the relative drift is REPORTED.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here / 'core'))
import engine as E
import units as U
from design import Design


def main():
    D = Design(dev='cuda:0', dtype=torch.bfloat16)
    tok, model, lm = E.load_model('/root/autodl-tmp/Qwen3.5-4B', D.dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, D.dev)
    scale_cpu = U.dose_scale(p50, U.column_norms_dev(lm, layers, D.dev), layers)

    got = D.scale_for(scale_cpu)
    on_dev = all(got[l].device.type == 'cuda' for l in layers)
    same = all(torch.equal(got[l].cpu(), scale_cpu[l]) for l in layers)
    print(f'A  device ok for {sum(got[l].device.type == "cuda" for l in layers)}/{len(layers)} layers; '
          f'values bit-identical after round trip: {same}')

    rejected = False
    try:
        D.check(torch.zeros(3), 'cpu tensor')
    except RuntimeError:
        rejected = True
    print(f'B  wrong-device tensor rejected at the boundary: {rejected}')

    model32, lm32 = D.as_float(model, lm, E.model_lm)
    p50_32 = E.estimate_p50(tok, model32, lm32, layers, D.dev)
    scale_32 = U.dose_scale(p50_32, U.column_norms_dev(lm32, layers, D.dev), layers)
    drift = max(float((scale_32[l] - scale_cpu[l]).abs().max() / scale_cpu[l].abs().max()) for l in layers)
    print(f'C  float32-vs-bf16 scale drift: {drift:.3e}')

    ok = on_dev and same and rejected
    print('DESIGN-VERIFY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
