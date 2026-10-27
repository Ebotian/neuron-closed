"""Language-mass readout J2 (plan §2.2): exact weighted VJP over a slot language.

    J2(x) = log( (1/|L+|) Σ_{w∈L+} e^{S(w)} ) - log( (1/|L-|) Σ_{w∈L-} e^{S(w)} )
    S(w)  = (1/T) Σ_t log P(tokens(template(w))_t | x, ...)          (teacher-forced)

with the gradient

    ∂J2/∂d = Σ_w ω(w) ∂S(w)/∂d ,   ω = softmax over the language at d = 0 .

Because ∂/∂d log Σ e^S = Σ ω ∂S, the softmax weights are constants at the point
where the derivative is taken, so ONE grad-enabled batched pass is exact (no
sampling, no approximation). Slot words must be one token and the template must
give an equal-length continuation on both sides; both are asserted at run time.

No model weights are touched; same dose convention as engine.score_for_dose
(activation add of dose * p50 / ||W[:,n]|| at every position).
"""
from __future__ import annotations

import contextlib
import math

import torch
import torch.nn.functional as F

from engine import apply_steering, capture_mlp_inputs, prompt_prefix


def build_batch(tokenizer, prompt, template, pos_words, neg_words, device, max_seq=384):
    prefix = prompt_prefix(tokenizer, prompt)
    seqs, kinds = [], []
    for words, kind in ((pos_words, 1), (neg_words, -1)):
        for w in words:
            text = template.format(w=w)
            ids = tokenizer(text, add_special_tokens=False).input_ids
            if not ids or not isinstance(ids[0], int):
                raise ValueError(f'template did not tokenize to ids: {text!r}')
            seqs.append(prefix + ids)
            kinds.append(kind)
    if len({len(s) for s in seqs}) != 1:
        raise ValueError('slot sequences are not equal length; fix the template or the pool')
    T = len(seqs[0])
    if T > max_seq:
        raise ValueError(f'sequence length {T} > max_seq {max_seq}')
    ids = torch.tensor(seqs, dtype=torch.long, device=device)
    kinds = torch.tensor(kinds, dtype=torch.long, device=device)
    return ids, len(prefix), T - len(prefix), kinds


def batch_mean_logp(model, ids, prefix_len, suffix_len):
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits[:, prefix_len - 1: prefix_len - 1 + suffix_len]
    target = ids[:, prefix_len: prefix_len + suffix_len]
    if logits.shape[1] != suffix_len:
        raise AssertionError('token alignment failed')
    lp = -F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                          target.reshape(-1), reduction='none')
    return lp.reshape(ids.shape[0], suffix_len).mean(dim=1)


def language_log_mass(S):
    """log(mean e^{S}) and the within-language softmax weights (detached)."""
    w = torch.softmax(S, dim=0)
    return float(torch.logsumexp(S, dim=0) - math.log(S.numel())), w


def sem_gradient(tokenizer, model, lm, layers, prompt, template, pools, device, max_seq=384):
    """One grad-enabled batched pass; returns (J2, {layer: summed-over-positions grad})."""
    ids, pre, suf, kinds = build_batch(tokenizer, prompt, template, pools['pos'], pools['neg'],
                                       device, max_seq)
    with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
        S = batch_mean_logp(model, ids, pre, suf)
        with torch.no_grad():
            Sd = S.detach()
            j_pos, w_pos = language_log_mass(Sd[kinds > 0])
            j_neg, w_neg = language_log_mass(Sd[kinds < 0])
            coeff = torch.zeros_like(Sd)
            coeff[kinds > 0] = w_pos
            coeff[kinds < 0] = -w_neg
        loss = (coeff * S).sum()          # coeff is detached -> exact first derivative
        inputs = [cap[l] for l in layers]
        grads = torch.autograd.grad(loss, inputs)
        out = {l: g.detach().float().sum(dim=tuple(range(g.ndim - 1))).cpu()
               for l, g in zip(layers, grads)}
    return j_pos - j_neg, out


@torch.inference_mode()
def j2_value(tokenizer, model, lm, p50, norms, prompt, template, pools, interventions=None,
             device='cuda:0', max_seq=384):
    """J2 value with an optional steering intervention (for finite differences)."""
    ids, pre, suf, kinds = build_batch(tokenizer, prompt, template, pools['pos'], pools['neg'],
                                       device, max_seq)
    with apply_steering(lm, interventions or [], p50, norms):
        S = batch_mean_logp(model, ids, pre, suf)
    j_pos, _ = language_log_mass(S[kinds > 0])
    j_neg, _ = language_log_mass(S[kinds < 0])
    return j_pos - j_neg


def fd_check(tokenizer, model, lm, p50, norms, layers, prompt, template, pools, candidates,
             dose, device='cuda:0', max_seq=384):
    """Same-prompt central difference vs the analytic derivative, for a few coordinates."""
    j0 = j2_value(tokenizer, model, lm, p50, norms, prompt, template, pools, [], device, max_seq)
    rows = []
    for c in candidates:
        l, n, s = int(c['layer']), int(c['neuron']), int(c['sign'])
        # central difference along +dose (UNSIGNED direction), so it is directly
        # comparable to the analytic derivative; the candidate's sign is applied later.
        iv = [{'layer': l, 'neuron': n, 'dose': dose}]
        jp = j2_value(tokenizer, model, lm, p50, norms, prompt, template, pools, iv, device, max_seq)
        iv = [{'layer': l, 'neuron': n, 'dose': -dose}]
        jm = j2_value(tokenizer, model, lm, p50, norms, prompt, template, pools, iv, device, max_seq)
        fd = (jp - jm) / (2 * dose)
        rows.append({'layer': l, 'neuron': n, 'sign': s, 'analytic': float(c['gradient_per_dose']),
                     'fd': fd, 'J_plus': jp, 'J_minus': jm, 'J_zero': j0,
                     'curvature': (jp + jm - 2 * j0) / dose ** 2})
    return rows
