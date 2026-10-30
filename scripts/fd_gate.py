#!/usr/bin/env python3
"""Instrument calibration gate: does the capture actually return dPhi/d(dose)?

WHY. audit-response-2026-10-30.md §12.1-2. The gradient readout was wrong in three scripts
(detach-then-clone cut the residual-carrier paths), which silently invalidated plans 2.40/2.42/2.43/2.46.
Before ANY science number may be reported again, this gate must pass, and its last-pass record is what
`make check` looks at.

WHAT IT DOES. In fp32 (bf16 finite differences are swamped by precision noise - measured: the same
coordinate gave -71.9 at eps=1e-3 and -4.7 at eps=1e-2), for layers 0, 13 and 31 and the top-2 |dose
gradient| coordinates of each:
  analytic = mlp_capture.Capture(...).dose_grad()          (the instrument under test)
  legacy   = the old detach-then-clone capture             (kept only to document the bug's signature)
  FD       = central difference of phi in dose units at eps = 1e-3 and 1e-2
phi is the teacher-forced sum log-probability of a fixed slot after a fixed prompt+completion, i.e. the
same kind of readout the plans use, with the injection applied at the coordinate under test.

FROZEN CRITERION (written before the run)
  G1  the instrument passes iff, for EVERY tested coordinate, min over eps of
      |FD - analytic| / max(|analytic|, 1e-6) < 0.02.
  G2  the legacy method must NOT pass: at least one coordinate must show > 0.05 relative error, otherwise
      this gate is not actually able to see the bug and the gate itself is uninformative.
  G3  layers 0/13/31 are reported separately; the recorded verdict is the conjunction over all three.
Output: results/fd_gate.json (and the same JSON on the box), consumed by tools/check.py.
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

    def phi(doses):
        hs = []
        for (L, n), d in doses.items():
            vec = torch.zeros(lm.layers[L].mlp.down_proj.weight.shape[1], device=dev, dtype=torch.float32)
            vec[n] = d * p50[L] / float(norms[L][n])
            def mk(v):
                def hook(_m, inp):
                    act = inp[0].clone()
                    return (act + v.to(device=act.device, dtype=act.dtype),) + inp[1:]
                return hook
            hs.append(lm.layers[L].mlp.down_proj.register_forward_pre_hook(mk(vec)))
        try:
            with torch.inference_mode():
                o = model(input_ids=ids, use_cache=False)
                lp = F.log_softmax(o.logits[:, :-1].float(), -1)
                return float(lp[0, pos - 1, torch.tensor(slot, device=dev)].sum())
        finally:
            for h in hs:
                h.remove()

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
                lp[0, pos - 1, torch.tensor(slot, device=dev)].sum().backward()
                flat = cap.dose_grad()          # <- the shipped code path, reduction and scaling included
            out, off = {}, 0
            for L in want:
                d_ = int(lm.layers[L].mlp.down_proj.weight.shape[1])
                out[L] = flat[off:off + d_].clone()
                off += d_
            return out
        hs, cap_, = [], {}
        for L in list(want):
            def build(idx):
                def hook(_m, args):
                    x = args[0]
                    if idx in cap_:
                        return None
                    xx = x.detach().clone().requires_grad_(True)   # the legacy (wrong) readout, deliberately kept
                    cap_[idx] = xx
                    return (xx,) + tuple(args[1:])
                return hook
            hs.append(lm.layers[L].mlp.down_proj.register_forward_pre_hook(build(L)))
        try:
            model.zero_grad(set_to_none=True)
            o = model(input_ids=ids, use_cache=False)
            lp = F.log_softmax(o.logits[:, :-1].float(), -1)
            lp[0, pos - 1, torch.tensor(slot, device=dev)].sum().backward()
            out = {}
            for L in want:
                g = cap_[L].grad
                out[L] = g.sum(dim=tuple(range(g.dim() - 1))).double().cpu() * \
                    (p50[L] / norms[L].clamp(min=1e-12)).double()
            return out
        finally:
            for h in hs:
                h.remove()

    want = [int(x) for x in a.layers.split(',')]
    eps_list = [float(x) for x in a.eps.split(',')]
    cur = analytic('new', want)
    leg = analytic('old', want)

    rows_out, ok_new, ok_old_visible = [], True, False
    summary = []
    print(f'{"L":>3} {"n":>6} {"analytic":>12} {"legacy":>12} {"FD(eps)":>12} {"eps":>7} {"relerr_new":>11} {"relerr_legacy":>13}')
    for L in want:
        for n in torch.topk(cur[L].abs(), a.topn).indices.tolist():
            best = best_leg = None
            for eps in eps_list:
                fd = (phi({(L, n): eps}) - phi({(L, n): -eps})) / (2 * eps)
                ra = abs(fd - float(cur[L][n])) / max(abs(float(cur[L][n])), 1e-6)
                rl = abs(fd - float(leg[L][n])) / max(abs(float(leg[L][n])), 1e-6)
                # the frozen criterion is the MINIMUM over eps: fp32 logp noise (~1e-3 absolute) makes the
                # smaller step unreliable, while the larger step leaves the linear regime, so neither single
                # step is a fair test and the criterion was written with min-over-eps for this reason.
                best = ra if best is None else min(best, ra)
                best_leg = rl if best_leg is None else min(best_leg, rl)
                rows_out.append({'layer': L, 'neuron': int(n), 'analytic': float(cur[L][n]),
                                 'legacy': float(leg[L][n]), 'fd': fd, 'eps': eps,
                                 'rel_err_new': ra, 'rel_err_legacy': rl})
                print(f'{L:>3} {int(n):>6} {float(cur[L][n]):>12.4f} {float(leg[L][n]):>12.4f} '
                      f'{fd:>12.4f} {eps:>7.0e} {ra:>11.2e} {rl:>13.2e}', flush=True)
            summary.append({'layer': L, 'neuron': int(n), 'min_rel_err_new': best,
                            'min_rel_err_legacy': best_leg})
            print(f'    -> min over eps: new {best:.2e}   legacy {best_leg:.2e}', flush=True)
            if best >= 0.02:
                ok_new = False
            if best_leg > 0.05:
                ok_old_visible = True
    verdict = {'G1_instrument_passes': ok_new,
               'G2_legacy_signature_visible': ok_old_visible,
               'G3_all_layers_ok': ok_new,
               'gate_passes': bool(ok_new and ok_old_visible)}
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
               'criterion': 'G1 rel_err_new<0.02 for every row; G2 some legacy rel_err>0.05; G3 conjunction'},
              open(a.out, 'w'), indent=1)
    print('wrote', a.out, flush=True)
    return 0 if verdict['gate_passes'] else 1


if __name__ == '__main__':
    sys.exit(main())
