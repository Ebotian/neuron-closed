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

import hashlib
import json
from pathlib import Path

import torch

import engine as E
from core.units import column_norms_dev

# Bump when the ALGORITHM that produces a cached value changes (p50 readout, reference text handling, precision).
CACHE_VERSION = 2
IDENTITY_FILES = ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'generation_config.json', 'vocab.json')


def _fingerprint(model_path, layers, extra=None, dtype=None):
    """Identity of everything a cached value depends on.

    The audit's objection was exact: the key was the layer list and the filename was the model directory's basename, so
    the same directory name with different weights, a changed tokenizer or a changed reference text would silently reuse
    an old p50. Here the key binds
      * the model path and the config/tokenizer file CONTENTS (small, hashed),
      * the weight FILES by name and size,
      * the reference text that p50 is computed from, the dtype, the layer list and CACHE_VERSION.
    STATED LIMIT: the 8.8 GB of weights are identified by name and size, not by content hash, because hashing them costs
    more than the computation being cached. Two snapshots with identical file names and sizes but different contents would
    collide; that is a deliberate trade and it is written down rather than assumed away.
    """
    h = hashlib.sha256()
    mp = Path(model_path)
    h.update(f'v{CACHE_VERSION}|{mp.resolve()}|{".".join(map(str, layers))}|'.encode())
    for name in IDENTITY_FILES:
        f = mp / name
        if f.exists():
            h.update(f'{name}:{hashlib.sha256(f.read_bytes()).hexdigest()}|'.encode())
    index = mp / 'model.safetensors.index.json'
    weights = sorted(mp.glob('*.safetensors'))
    if index.exists():
        h.update(f'index:{hashlib.sha256(index.read_bytes()).hexdigest()}|'.encode())
    for w in weights:
        h.update(f'{w.name}:{w.stat().st_size}|'.encode())
    h.update(f'ref:{hashlib.sha256(E.REFERENCE_TEXT.encode()).hexdigest()}|'.encode())
    # the dtype must be OBSERVED, not assumed: `--fp32` exists, and a hardcoded 'bf16' would let a float32 run reuse a
    # bfloat16 cache. Unknown dtype is recorded as 'unknown', which still separates it from any known one.
    h.update(f'dtype:{dtype or "unknown"}|extra:{json.dumps(extra, sort_keys=True) if extra else "-"}|'.encode())
    return h.hexdigest()


def load_or_compute(path, compute, key, layers, strict=False):
    """Return `compute()`'s value, from `path` when possible. `strict` re-verifies a cache hit against a fresh compute."""
    p = Path(path) if path else None
    if p is not None and p.exists():
        blob = torch.load(p)
        if blob.get('key') == key:
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
        torch.save({'key': key, 'cache_version': CACHE_VERSION, 'layers': list(layers), 'values': vals}, p)
    return vals


def p50_cached(tok, model, lm, layers, dev, path=None, strict=False, model_path=None, dtype=None):
    key = _fingerprint(model_path or '', layers, {'quantity': 'p50'}, dtype)
    return load_or_compute(path, lambda: E.estimate_p50(tok, model, lm, layers, dev), key, layers, strict)


def column_norms_cached(lm, layers, dev, path=None, strict=False, model_path=None, dtype=None):
    key = _fingerprint(model_path or '', layers, {'quantity': 'column_norms'}, dtype)
    return load_or_compute(path, lambda: column_norms_dev(lm, layers, dev), key, layers, strict)
