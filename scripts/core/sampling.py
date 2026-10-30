"""The sampling protocol - the ONE implementation.

Seeds are consumed in a fixed order (a global seed, then one per context) because changing the order changes the
trajectories and therefore every number downstream: two runs of the same script must differ only by the seed argument.
"""
from __future__ import annotations

import torch

import engine as E


def generate_rollouts(model, tok, prompt, n, max_new, temp, pad, dev, seed):
    """`n` sampled continuations of `prompt`, returned as full id lists (prompt + continuation)."""
    torch.manual_seed(seed)
    pre = E.prompt_prefix(tok, prompt)
    ids = torch.tensor([pre], dtype=torch.long, device=dev)
    with torch.inference_mode():
        o = model.generate(input_ids=ids, do_sample=True, temperature=temp, top_p=1.0,
                           max_new_tokens=max_new, use_cache=True, num_return_sequences=n,
                           pad_token_id=pad)
    seqs = []
    for g in o[:, ids.shape[1]:].tolist():
        if tok.eos_token_id in g:
            g = g[:g.index(tok.eos_token_id) + 1]
        seqs.append(pre + (g or [tok.eos_token_id]))
    return pre, seqs
