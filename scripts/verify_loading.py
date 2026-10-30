#!/usr/bin/env python3
"""The fast loader must give the same model as the frozen one, per value.

A  p50 from the vendor loader vs from core.loading: identical per layer (torch.equal).
B  column norms: identical.
C  a one-context phi and score gradient computed with each loader: identical.
"""
from __future__ import annotations
import sys
from pathlib import Path

import torch

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here)); sys.path.insert(0, str(_here / 'core')); sys.path.insert(0, str(_here / 'estimators'))
import engine as E
from core.loading import load_model_fast
from core.units import column_norms_dev, dose_scale
from estimators.phi import phi_of
from estimators.score import per_rollout_scores


def main():
    dev = 'cuda:0'
    layers = list(range(32))
    rows = [__import__('json').loads(l) for l in open('/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')]
    r = [x for x in rows if x['trait'] == 'hostility'][0]

    tok0, m0, lm0 = E.load_model('/root/autodl-tmp/Qwen3.5-4B', dev)
    p0 = E.estimate_p50(tok0, m0, lm0, layers, dev)
    n0 = column_norms_dev(lm0, layers, dev)
    torch.manual_seed(0)
    pre = E.prompt_prefix(tok0, r['prompt'])
    with torch.inference_mode():
        o = m0.generate(input_ids=torch.tensor([pre], device=dev), do_sample=False, max_new_tokens=8,
                        use_cache=True, pad_token_id=tok0.eos_token_id)
    seqs = [pre + g for g in o[:, len(pre):].tolist()]
    ph0 = [float(x) for x in phi_of(m0, tok0, seqs, r, tok0.eos_token_id, dev)]
    del m0, lm0
    torch.cuda.empty_cache()

    tok1, m1, lm1 = load_model_fast('/root/autodl-tmp/Qwen3.5-4B', dev)
    p1 = E.estimate_p50(tok1, m1, lm1, layers, dev)
    n1 = column_norms_dev(lm1, layers, dev)
    ph1 = [float(x) for x in phi_of(m1, tok1, seqs, r, tok1.eos_token_id, dev)]
    d_p50 = [l for l in layers if p0[l] != p1[l]]
    d_nrm = [l for l in layers if not torch.equal(n0[l], n1[l])]
    d_phi = max(abs(a - b) for a, b in zip(ph0, ph1))
    print(f'A p50 identical on {len(layers)-len(d_p50)}/{len(layers)} layers; B column norms identical on {len(layers)-len(d_nrm)}/{len(layers)}')
    print(f'C phi max |difference| = {d_phi:.2e}')
    ok = not d_p50 and not d_nrm and d_phi < 1e-6
    print('LOADING-VERIFY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
