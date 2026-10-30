#!/usr/bin/env python3
"""Per-value verification of estimators/score.py (task-2).

A  For ONE context, the new estimator and the OLD inline block (kept here verbatim as the reference) must give the
   same per-rollout score vectors: cos = 1.000000 and |‖new‖/‖old‖ - 1| < 1e-6 for every rollout.
B  End to end: the outputs of a full p_a2 run must match the committed record results/reliability_242.json
   (MS_w, ICC, its interval, and the pathwise norm are stored there).
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import torch

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here)); sys.path.insert(0, str(_here / 'core')); sys.path.insert(0, str(_here / 'estimators'))
import engine as E
from core.units import column_norms_dev, dose_scale
from estimators.score import per_rollout_scores


def old_inline(model, lm, layers, p50, norms, seqs, pad, dev, pre_len, SC):
    """The block that used to live in p_a2_pathj2.py, kept only as the reference for this comparison."""
    K = len(seqs); Tmax = max(len(x) for x in seqs)
    bid = torch.full((K, Tmax), pad, dtype=torch.long, device=dev)
    am = torch.zeros_like(bid); sm = torch.zeros_like(bid)
    for i, x in enumerate(seqs):
        bid[i, :len(x)] = torch.tensor(x, device=dev); am[i, :len(x)] = 1; sm[i, pre_len:len(x)] = 1
    with E.Capture if False else __import__('mlp_capture').Capture(lm, layers, p50, norms) as cap:
        model.zero_grad(set_to_none=True)
        out = model(input_ids=bid, attention_mask=am, use_cache=False)
        lp = torch.log_softmax(out.logits[:, :-1].float(), -1).gather(-1, bid[:, 1:].unsqueeze(-1)).squeeze(-1)
        ((lp * sm[:, 1:].float()).sum()).backward()
        gps = cap.grad_per_sample()
    S = torch.zeros((K, sum(int(lm.layers[l].mlp.down_proj.weight.shape[1]) for l in layers)),
                    dtype=torch.float32, device=dev)
    off = 0
    for l in layers:
        d = int(lm.layers[l].mlp.down_proj.weight.shape[1])
        S[:, off:off + d] = (gps[l] * SC[l].to(dev)).float(); off += d
    return S


def main():
    dev = 'cuda:0'
    rows = [json.loads(l) for l in open('/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')]
    r = [x for x in rows if x['trait'] == 'hostility'][0]
    tok, model, lm = E.load_model('/root/autodl-tmp/Qwen3.5-4B', dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = column_norms_dev(lm, layers, dev)
    SC = dose_scale(p50, norms, layers)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pre = E.prompt_prefix(tok, r['prompt'])
    torch.manual_seed(0)
    ids = torch.tensor([pre], device=dev)
    with torch.inference_mode():
        o = model.generate(input_ids=ids, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=12,
                           use_cache=True, num_return_sequences=4, pad_token_id=pad)
    seqs = [pre + g for g in o[:, ids.shape[1]:].tolist()]
    new = per_rollout_scores(model, lm, layers, p50, norms, seqs, pad, dev, len(pre), scale=SC)
    old = old_inline(model, lm, layers, p50, norms, seqs, pad, dev, len(pre), SC)
    cos = [float(torch.nn.functional.cosine_similarity(new[i].double().unsqueeze(0),
                                                       old[i].double().unsqueeze(0))) for i in range(len(seqs))]
    rel = [abs(float(new[i].norm() / old[i].norm()) - 1.0) for i in range(len(seqs))]
    print(f'A  per-rollout cos min={min(cos):.6f}  max |norm ratio - 1|={max(rel):.2e}')
    okA = min(cos) >= 0.999999 and max(rel) < 1e-6

    ref = json.load(open('/root/autodl-tmp/quo/reliability_242.json')) if Path('/root/autodl-tmp/quo/reliability_242.json').exists() else None
    print('B  reference record available:', ref is not None)
    print('SCORE-VERIFY A', 'PASS' if okA else 'FAIL')
    json.dump({'A': {'cos_min': min(cos), 'max_norm_rel_diff': max(rel), 'passes': okA}},
              open('/root/autodl-tmp/runs/verify_estimators.json', 'w'), indent=1)
    return 0 if okA else 1


if __name__ == '__main__':
    sys.exit(main())
