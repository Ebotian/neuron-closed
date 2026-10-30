#!/usr/bin/env python3
"""Per-value verification of estimators/{pathwise,teacher}.py (task-3).

The reference is a verbatim copy of the code the extraction replaced (kept here, not in the live path). For one
sequence and one slot, the new module and the reference must give cos = 1.000000 and |‖new‖/‖old‖ - 1| < 1e-6.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here)); sys.path.insert(0, str(_here / 'core')); sys.path.insert(0, str(_here / 'estimators'))
import engine as E
from core.units import column_norms_dev, dose_scale
from mlp_capture import Capture
from estimators.pathwise import pathwise_grad
from estimators.teacher import teacher_grad


def old_pathwise(model, lm, layers, p50, norms, sq, ids_slot, sgn, weight, dev, SC):
    """Verbatim copy of the block that used to live in p_a2_pathj2.py."""
    full = sq + ids_slot
    j1 = torch.tensor([full], dtype=torch.long, device=dev)
    with Capture(lm, layers, p50, norms) as cap:
        model.zero_grad(set_to_none=True)
        out = model(input_ids=j1, use_cache=False)
        lp = torch.log_softmax(out.logits[:, :-1].float(), -1)
        pos = torch.arange(len(sq), len(full), device=dev)
        tgt = torch.tensor(ids_slot, device=dev)
        (lp[0, pos - 1, tgt].mean() * (sgn * weight)).backward()
        raw = cap.grad_sum()
    return {l: raw[l].to(dev) * SC[l].to(dev) for l in layers}


def old_teacher(model, lm, tok, layers, p50, norms, prompt, slot_txt, sgn, weight, dev, SC):
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


def cmp(a, b, layers):
    flat = lambda d: torch.cat([d[l].reshape(-1).double() for l in layers])
    x, y = flat(a), flat(b)
    return float(torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0))), abs(float(x.norm() / y.norm()) - 1)


def main():
    dev = 'cuda:0'
    rows = [__import__('json').loads(l) for l in open('/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')]
    r = [x for x in rows if x['trait'] == 'hostility'][0]
    tok, model, lm = E.load_model('/root/autodl-tmp/Qwen3.5-4B', dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = column_norms_dev(lm, layers, dev)
    SC = dose_scale(p50, norms, layers)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pre = E.prompt_prefix(tok, r['prompt'])
    torch.manual_seed(0)
    with torch.inference_mode():
        o = model.generate(input_ids=torch.tensor([pre], device=dev), do_sample=False, max_new_tokens=8,
                           use_cache=True, pad_token_id=pad)
    seq = pre + o[0, len(pre):].tolist()
    slot_txt = 'deeply hostile and cruel'
    slot = tok(slot_txt, add_special_tokens=False).input_ids
    w = 1.0 / 37.0
    c1, r1 = cmp(pathwise_grad(model, lm, layers, p50, norms, seq, slot, +1.0, w, dev, scale=SC),
                 old_pathwise(model, lm, layers, p50, norms, seq, slot, +1.0, w, dev, SC), layers)
    c2, r2 = cmp(teacher_grad(model, lm, tok, layers, p50, norms, r['prompt'], slot_txt, +1.0, w, dev, scale=SC),
                 old_teacher(model, lm, tok, layers, p50, norms, r['prompt'], slot_txt, +1.0, w, dev, SC), layers)
    print(f'pathwise cos={c1:.6f} rel_norm={r1:.2e}   teacher cos={c2:.6f} rel_norm={r2:.2e}')
    ok = min(c1, c2) >= 0.999999 and max(r1, r2) < 1e-6
    print('PATHWISE-VERIFY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
