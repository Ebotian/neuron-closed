#!/usr/bin/env python3
"""Calibration gate: does the gradient readout return dPhi/d(dose)? Criterion and rationale in analysis/fd.py.

fp32 only (bf16 FD is swamped by noise); layers 0/13/31, two step sizes; the shipped module via Capture().dose_grad(),
the legacy detach readout from legacy_capture.py as the negative control.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
from mlp_capture import Capture


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes', default='/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')
    ap.add_argument('--layers', default='0,13,31')
    ap.add_argument('--topn', type=int, default=2)
    ap.add_argument('--eps', default='1e-3,1e-2')
    ap.add_argument('--out', default='/root/autodl-tmp/runs/fd_gate.json')
    a = ap.parse_args()
    dev = 'cuda:0'

    rows = [json.loads(l) for l in open(a.probes)]
    r = [x for x in rows if x['trait'] == 'hostility'][0]
    tok, model, lm = E.load_model(a.model, dev)
    model = model.float()                      # fp32: the FD referee needs the precision
    lm = E.model_lm(model)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = E.column_norms(lm, layers)
    for q in model.parameters():
        q.requires_grad_(False)
    print('dtype', next(model.parameters()).dtype, flush=True)

    pre = E.prompt_prefix(tok, r['prompt'])
    comp = tok('I hate you.', add_special_tokens=False).input_ids
    slot = tok(r['positive'] if isinstance(r['positive'], str) else r['positive'][0],
               add_special_tokens=False).input_ids
    ids = torch.tensor([pre + comp + slot], dtype=torch.long, device=dev)
    n_slot_start = len(pre) + len(comp)
    T = ids.shape[1]
    pos = torch.arange(n_slot_start, T, device=dev)

    from core.units import dose_scale
    from estimators.steer import phi_teacher_forced
    SC = {l: v.to(dev) for l, v in dose_scale(p50, norms, layers).items()}

    def phi(doses):
        return phi_teacher_forced(model, lm, ids, slot, n_slot_start, doses, SC, layers, dev)

    def analytic(mode, want):
        """mode 'new' MUST go through the shipped module; mode 'old' is the inline legacy negative control.

        The first version of this function re-implemented the zero-leaf hook inline even for mode 'new', so the
        certificate would have kept passing after a regression inside mlp_capture.Capture. The auditor was right:
        a certificate must exercise the artifact it names. Now 'new' calls Capture(...).dose_grad() end to end and
        slices the concatenated vector back into layers, and fd_gate.json pins the module inode's hash.
        """
        if mode == 'new':
            with Capture(lm, list(want), p50, norms) as cap:
                model.zero_grad(set_to_none=True)
                o = model(input_ids=ids, use_cache=False)
                lp = F.log_softmax(o.logits[:, :-1].float(), -1)
                lp[0, pos - 1, torch.tensor(slot, device=dev)].mean().backward()   # MEAN: the same functional the gate's phi uses
                flat = cap.dose_grad()          # <- the shipped code path, reduction and scaling included
            out, off = {}, 0
            for L in want:
                d_ = int(lm.layers[L].mlp.down_proj.weight.shape[1])
                out[L] = flat[off:off + d_].clone()
                off += d_
            return out
        from legacy_capture import legacy_capture_grads
        return legacy_capture_grads(model, lm, want, ids, pos, slot, SC, dev)

    want = [int(x) for x in a.layers.split(',')]
    eps_list = [float(x) for x in a.eps.split(',')]
    cur = analytic('new', want)
    leg = analytic('old', want)

    import analysis.fd as fdmod
    rows_out, summary, verdict = fdmod.compare(
        cur, leg, want, eps_list, phi, lambda L, row: torch.topk(row.abs(), a.topn).indices.tolist())
    for r in rows_out:
        print(f'{r["layer"]:>3} {r["neuron"]:>6} {r["analytic"]:>12.4f} {r["legacy"]:>12.4f} '
              f'{r["fd"]:>12.4f} {r["eps"]:>7.0e} {r["rel_err_new"]:>11.2e} {r["rel_err_legacy"]:>13.2e}', flush=True)
    for sm in summary:
        print(f'    -> min over eps: new {sm["min_rel_err_new"]:.2e}   legacy {sm["min_rel_err_legacy"]:.2e}', flush=True)
    print(json.dumps(verdict), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    # the gate certifies THIS instrument, so the record must pin its hash; tools/check.py compares it
    import hashlib
    inst = Path(__file__).resolve().parent / 'mlp_capture.py'
    inst_sha = hashlib.sha256(inst.read_bytes()).hexdigest()
    json.dump({'instrument': 'scripts/mlp_capture.py', 'instrument_sha256': inst_sha,
               'certifies': ('mlp_capture.Capture(...).dose_grad() END TO END: mode "new" in this script calls the shipped '
                             'module, not an inline copy, so a regression inside grad_sum/dose_grad fails this gate'),
               'negative_control': 'mode "old" keeps the legacy detach capture inline on purpose',
               'summary_min_over_eps': summary, 'rows': rows_out, 'verdict': verdict,
               'eps': eps_list, 'layers': want, 'topn': a.topn,
               'model': a.model, 'dtype': 'float32',
               'criterion': ('G1: for every COORDINATE, the minimum of rel_err_new over the tested eps values is < 0.02 '
                             '(the per-(coordinate,eps) errors are also recorded and several exceed 0.02 at eps=1e-3, which is '
                             'why the criterion is the min over eps and not every row); G2: some legacy coordinate shows '
                             'min rel_err > 0.05; G3: conjunction over layers 0/13/31'),
               'criterion_scope': 'G1 is applied per coordinate after taking the min over eps (see the criterion field)'},
              open(a.out, 'w'), indent=1)
    print('wrote', a.out, flush=True)
    return 0 if verdict['gate_passes'] else 1


if __name__ == '__main__':
    sys.exit(main())
