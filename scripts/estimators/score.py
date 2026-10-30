"""The score term - the ONE implementation.

WHAT IT IS. For each rollout y_i of a context, the gradient of the log-density of that rollout with respect to
the injected doses, i.e. `∇_u log p(y_i)` in dose units. It is returned UNWEIGHTED and per-rollout, because the
baseline is applied afterwards by whoever needs it (leave-one-out or in-sample); weighting here would make the
two baselines indistinguishable, which is exactly the audit's section 4 objection.

TWO THINGS THAT THIS FUNCTION OWNS, so no caller can disagree:
  * the normalisation: the density is the WHOLE-sequence sum of log-probabilities over continuation tokens
    (a sum, not a mean - the measure is over complete sequences), and the dose scale comes from core.units;
  * the batching: ONE forward and ONE backward for all K rollouts of a context. Each element of the delta leaf
    belongs to one sample, so `grad_per_sample` returns per-rollout derivatives from that single backward.
"""
from __future__ import annotations

import torch

from core.units import dose_scale  # the only place the dose unit is formed
from mlp_capture import Capture


def loo_weights(phi, n_ctx):
    """Leave-one-out baseline: w_i = (phi_i - mean of the OTHER rollouts) / (K * contexts).

    The baseline is FREE in expectation (lean/Decomposition.lean, `baseline_free`), so which baseline is used must not
    change the estimate - but writing the formula in four places is how the two baselines became indistinguishable.
    """
    K, psum = len(phi), sum(phi)
    return [(x - (psum - x) / max(K - 1, 1)) / (K * n_ctx) for x in phi]


def insample_weights(phi, n_ctx):
    """In-sample baseline: w_i = (phi_i - mean of ALL rollouts) / (K * contexts). Kept as an explicit alternative
    because reporting BOTH is what shows the estimate does not depend on the baseline choice."""
    K, psum = len(phi), sum(phi)
    return [(x - psum / K) / (K * n_ctx) for x in phi]


def per_rollout_scores(model, lm, layers, p50, norms, seqs, pad, dev, n_prompt, scale=None):
    """Return S: (K, dim) float32 on `dev`, row i = ∇_u log p(y_i) in dose units (unweighted).

    `n_prompt` is passed in rather than inferred: inferring it from a common prefix was a silent hazard (two
    rollouts that happen to start alike would shorten it).
    """
    SC = scale if scale is not None else dose_scale(p50, norms, layers)
    K = len(seqs)
    Tmax = max(len(x) for x in seqs)
    bid = torch.full((K, Tmax), pad, dtype=torch.long, device=dev)
    am = torch.zeros_like(bid)
    sm = torch.zeros_like(bid)
    for i, x in enumerate(seqs):
        bid[i, :len(x)] = torch.tensor(x, device=dev)
        am[i, :len(x)] = 1
        sm[i, n_prompt:len(x)] = 1
    with Capture(lm, layers, p50, norms) as cap:
        model.zero_grad(set_to_none=True)
        out = model(input_ids=bid, attention_mask=am, use_cache=False)
        lp = torch.log_softmax(out.logits[:, :-1].float(), -1).gather(
            -1, bid[:, 1:].unsqueeze(-1)).squeeze(-1)
        ((lp * sm[:, 1:].float()).sum()).backward()   # SUM over continuation tokens, weight 1
        gps = cap.grad_per_sample()
    S = torch.zeros((K, sum(int(lm.layers[l].mlp.down_proj.weight.shape[1]) for l in layers)),
                    dtype=torch.float32, device=dev)
    off = 0
    for l in layers:
        d = int(lm.layers[l].mlp.down_proj.weight.shape[1])
        S[:, off:off + d] = (gps[l] * SC[l].to(dev)).float()
        off += d
    return S
