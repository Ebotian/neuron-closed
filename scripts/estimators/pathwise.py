"""The pathwise term - the ONE implementation.

`d/du of weight * sgn * (MEAN over the slot's own tokens of log p(t | x, seq))`, i.e. the derivative of the readout
taken along a given trajectory with the trajectory held fixed. It is obtained from ONE batched backward over all the
(sample, slot-sign) pairs of a context: each element of the delta leaf belongs to one sample, so per-sample
derivatives come from the same backward.

CONVENTIONS OWNED HERE
  * the readout is a MEAN over the slot's own tokens - `len(ids_slot)` appears exactly once, here, dividing once;
  * the caller passes the weight, so the 1/(K·C) averaging lives in one place (the caller), not twice;
  * the dose scale comes from core.units.
"""
from __future__ import annotations

import torch

from core.units import dose_scale
from mlp_capture import Capture


def slotted_mean_logp(model, seq, ids_slot, dev):
    """Mean log p over the slot's own tokens, teacher-forced after `seq`. The functional both branches share."""
    full = seq + ids_slot
    j = torch.tensor([full], dtype=torch.long, device=dev)
    out = model(input_ids=j, use_cache=False)
    lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
    pos = torch.arange(len(seq), len(full), device=dev)
    return lp[0, pos - 1, torch.tensor(ids_slot, device=dev)].mean()


def pathwise_grad(model, lm, layers, p50, norms, seq, ids_slot, sgn, weight, dev, scale=None):
    """Gradient of `weight * sgn * mean-slot-logp` w.r.t. the doses, dose-scaled, per layer (CPU float64)."""
    SC = scale if scale is not None else dose_scale(p50, norms, layers)
    j = torch.tensor([seq + ids_slot], dtype=torch.long, device=dev)
    with Capture(lm, layers, p50, norms) as cap:
        model.zero_grad(set_to_none=True)
        out = model(input_ids=j, use_cache=False)
        lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
        pos = torch.arange(len(seq), len(seq + ids_slot), device=dev)
        (lp[0, pos - 1, torch.tensor(ids_slot, device=dev)].mean() * (sgn * weight)).backward()
        raw = cap.grad_sum()
    return {l: raw[l].to(dev) * SC[l].to(dev) for l in layers}


def pathwise_batch(model, lm, layers, p50, norms, items, pad, dev, scale=None, chunk=8):
    """All (seq, ids_slot, sgn) pairs of a context in chunked batches; returns the accumulated per-layer gradient."""
    SC = scale if scale is not None else dose_scale(p50, norms, layers)
    acc = {l: torch.zeros(lm.layers[l].mlp.down_proj.weight.shape[1], dtype=torch.float64, device=dev)
           for l in layers}
    Tmax = max(len(seq) + len(slot) for seq, slot, _, _ in items)
    for c0 in range(0, len(items), chunk):
        part = items[c0:c0 + chunk]
        bid = torch.full((len(part), Tmax), pad, dtype=torch.long, device=dev)
        am = torch.zeros_like(bid); smask = torch.zeros_like(bid)
        w = torch.zeros(len(part), dtype=torch.float64, device=dev)
        for b, (seq, slot, sgn, weight) in enumerate(part):
            full = seq + slot
            bid[b, :len(full)] = torch.tensor(full, device=dev)
            am[b, :len(full)] = 1
            smask[b, len(seq):len(full)] = 1          # the slot's own tokens, once
            w[b] = sgn * weight
        with Capture(lm, layers, p50, norms) as cap:
            model.zero_grad(set_to_none=True)
            out = model(input_ids=bid, attention_mask=am, use_cache=False)
            lp = torch.log_softmax(out.logits[:, :-1].float(), -1).gather(
                -1, bid[:, 1:].unsqueeze(-1)).squeeze(-1)
            per = (lp * smask[:, 1:].float()).sum(1) / smask[:, 1:].float().sum(1).clamp(min=1)
            (per * w).sum().backward()
            gps = cap.grad_per_sample()
        for l in layers:
            acc[l] += gps[l].sum(0) * SC[l].to(dev)
    return acc
