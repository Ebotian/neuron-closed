"""The path-J2 readout phi - the ONE implementation.

`phi(y) = mean over the POSITIVE slot's tokens of log p(t | x, y) - mean over the NEGATIVE slot's tokens`, teacher-forced
after the model's own continuation `y`. This is the functional whose on-policy gradient the estimator approximates; it is
in NATS, which is why it has usable dynamic range (a saturated judge gave 0.04 +- 0.01 and made the gradient unmeasurable).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def slot_ids(tok, s):
    txt = s if isinstance(s, str) else (s[0] if isinstance(s, (list, tuple)) else str(s))
    return tok(txt, add_special_tokens=False).input_ids


def slot_lp(model, seqs, ids_slot, pad, dev):
    """Per-row MEAN log p over the slot's own tokens, teacher-forced after each sequence. One division, here."""
    n_s = len(ids_slot)
    L = max(len(s) for s in seqs) + n_s
    ids = torch.full((len(seqs), L), pad, dtype=torch.long, device=dev)
    am = torch.zeros_like(ids)
    pos = torch.zeros((len(seqs), n_s), dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        full = s + ids_slot
        ids[i, :len(full)] = torch.tensor(full, device=dev)
        am[i, :len(full)] = 1
        for j in range(n_s):
            pos[i, j] = len(s) + j
    out = model(input_ids=ids, attention_mask=am, use_cache=False)
    lp = F.log_softmax(out.logits.float(), -1)
    vals = torch.stack([lp[torch.arange(len(seqs), device=dev), pos[:, j] - 1,
                           torch.tensor(ids_slot[j], device=dev)] for j in range(n_s)], dim=1)
    return vals.mean(dim=1)


def phi_of(model, tok, seqs, probe, pad, dev):
    ip, ineg = slot_ids(tok, probe['positive']), slot_ids(tok, probe['negative'])
    return slot_lp(model, seqs, ip, pad, dev) - slot_lp(model, seqs, ineg, pad, dev)
