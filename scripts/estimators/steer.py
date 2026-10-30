"""Injection and readout for the closed-form comparison - the ONE implementation.

The readout is the teacher-forced J2 log-mass on the fixed prompt (engine.continuation_logp), with the injection added
once per layer. Vectorised on purpose: the vendor helper loops over coordinates one at a time, which is fine for a
32-coordinate support and far too slow at 294,912.
"""
from __future__ import annotations

import contextlib

import torch

import engine as E


@contextlib.contextmanager
def steer(lm, alpha):
    """alpha: {layer: vector on the layer's device}. One add per layer."""
    hs = []
    for l, v in alpha.items():
        def make(vec):
            def hook(_m, inp):
                act = inp[0].clone()
                return (act + vec.to(device=act.device, dtype=act.dtype),) + inp[1:]
            return hook
        hs.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(make(v)))
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def readout(model, lm, tok, ctx, layers, d, b, SC, dev):
    """Mean teacher-forced J2 over the contexts at budget b (d=None means no injection)."""
    # d lives on the CPU (it comes from the min-norm solve), SC may live on the device: bring them together FIRST.
    # This is the exact mismatch class core/design.py exists to prevent, and this shell hit it once.
    alpha = {} if d is None else {l: ((b * d[l]) * SC[l].to(d[l].device)).float().to(dev) for l in layers}
    tot = 0.0
    with (steer(lm, alpha) if d is not None else contextlib.nullcontext()):
        with torch.inference_mode():
            for r in ctx:
                for key, sgn in (('positive', +1.0), ('negative', -1.0)):
                    sl = r[key] if isinstance(r[key], str) else r[key][0]
                    rid, rpre, rsuf = E.candidate_ids(tok, r['prompt'], sl, dev, 384)
                    tot += sgn * float(E.continuation_logp(model, rid, rpre, rsuf))
    return tot / len(ctx)
