"""The legacy (WRONG) gradient readout, kept ONLY as the negative control for the finite-difference gate.

`x = x.detach().clone().requires_grad_(True)` cuts the residual-carrier paths: `x_L = act(h_L)`, and `h_L` also flows
onward into every later layer's MLP, so the true `dPhi/dx_L` contains that whole family of paths and this truncates it.
Measured against fp32 finite differences it deviates 13.97%-22.28% at layers 0 and 13 and matches only at the last layer.

It lives in its own file so that a grep for the pattern finds exactly one deliberate occurrence, and `make check`'s
noregress item whitelists this file and only this file.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def legacy_capture_grads(model, lm, want, ids, pos, slot, SC, dev):
    """Gradients of mean-slot-logp via the detach-then-clone readout. Deliberately wrong; control use only."""
    hs, cap_ = [], {}
    for L in list(want):
        def build(idx):
            def hook(_m, args):
                x = args[0]
                if idx in cap_:
                    return None
                xx = x.detach().clone().requires_grad_(True)     # the defect under test
                cap_[idx] = xx
                return (xx,) + tuple(args[1:])
            return hook
        hs.append(lm.layers[L].mlp.down_proj.register_forward_pre_hook(build(L)))
    try:
        model.zero_grad(set_to_none=True)
        o = model(input_ids=ids, use_cache=False)
        lp = F.log_softmax(o.logits[:, :-1].float(), -1)
        lp[0, pos - 1, torch.tensor(slot, device=dev)].mean().backward()
        out = {}
        for L in want:
            g = cap_[L].grad
            out[L] = g.sum(dim=tuple(range(g.dim() - 1))).double().cpu() * SC[L].cpu()
        return out
    finally:
        for h in hs:
            h.remove()
