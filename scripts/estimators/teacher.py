"""The teacher-forced readout term - the ONE implementation.

`d/du of weight * sgn * (MEAN over the SUFFIX tokens of log p(slot | prompt))`: the same log-mass construct as the
pathwise branch, but conditioned on the fixed prompt rather than on the model's own continuation. The mean over
suffix tokens is what makes it the same functional as engine.continuation_logp; summing instead was one of the
normalisation bugs the audit found.
"""
from __future__ import annotations

import torch

import engine as E
from core.units import dose_scale
from mlp_capture import Capture


def teacher_grad(model, lm, tok, layers, p50, norms, prompt, slot_txt, sgn, weight, dev, scale=None):
    """Gradient of `weight * sgn * mean-suffix-logp` w.r.t. the doses, dose-scaled, per layer."""
    SC = scale if scale is not None else dose_scale(p50, norms, layers)
    rid, rpre, rsuf = E.candidate_ids(tok, prompt, slot_txt, dev, 384)
    i1 = (rid.to(dev) if torch.is_tensor(rid) else torch.tensor(rid, device=dev)).view(1, -1).long()
    with Capture(lm, layers, p50, norms) as cap:
        model.zero_grad(set_to_none=True)
        out = model(input_ids=i1, use_cache=False)
        lp = torch.log_softmax(out.logits[:, :-1].float(), -1).gather(-1, i1[:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = torch.zeros_like(lp)
        mask[:, int(rpre) - 1:] = 1.0
        ((lp * mask).sum() / mask.sum() * (sgn * weight)).backward()
        raw = cap.grad_sum()
    return {l: raw[l].to(dev) * SC[l].to(dev) for l in layers}
