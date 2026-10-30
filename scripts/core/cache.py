"""Cache the two per-model quantities that cost seconds and never change: p50 per layer and the column norms.

WHY. Both depend only on the frozen weights (p50 additionally on `engine.REFERENCE_TEXT`, which is frozen too), yet the
pipeline recomputed them on every run: a forward over the reference text with 32 hooks, plus 32 reductions over the
down-projection weights. Together they were ~2-3 s of the setup, which is what keeps the whole-run GPU utilisation just
under its bar.

A cache is only acceptable if it is invisible: `verify_units.py` already establishes that the GPU column norms equal the
CPU ones exactly, and `load_or_compute` re-checks the stored values against a fresh computation when `strict=True`, so a
stale or corrupt cache surfaces as a mismatch instead of a silent drift.
"""
from __future__ import annotations

from pathlib import Path

import torch

import engine as E
from core.units import column_norms_dev


def _key(layers):
    return ','.join(str(int(l)) for l in layers)


def load_or_compute(path, compute, layers, strict=True):
    """Return `compute()`'s value, from `path` when possible. `strict` re-verifies a cache hit against a fresh compute."""
    p = Path(path) if path else None
    if p is not None and p.exists():
        blob = torch.load(p)
        if blob.get('key') == _key(layers):
            vals = blob['values']
            if not strict:
                return vals
            fresh = compute()
            # values are EITHER scalars (p50) OR per-layer vectors (the 9216-wide column norms), so compare through
            # as_tensor: the first version assumed scalars and the second assumed tensors, and each crashed on the other.
            worst = max(float((torch.as_tensor(vals[l]).double() - torch.as_tensor(fresh[l]).double()).abs().max())
                        for l in layers)
            if worst == 0.0:
                return vals
            print(f'  cache {p.name}: STALE (max |diff| {worst:.3e}) - recomputed', flush=True)
    vals = compute()
    if p is not None:
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'key': _key(layers), 'values': vals}, p)
    return vals


def p50_cached(tok, model, lm, layers, dev, path=None, strict=True):
    return load_or_compute(path, lambda: E.estimate_p50(tok, model, lm, layers, dev), layers, strict)


def column_norms_cached(lm, layers, dev, path=None, strict=True):
    return load_or_compute(path, lambda: column_norms_dev(lm, layers, dev), layers, strict)
