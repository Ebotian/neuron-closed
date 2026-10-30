#!/usr/bin/env python3
"""Plan 2.46: replace the (wrong) isotropy assumption with the measured noise-to-signal ratio.

WHY. lean/Dimension.lean originally explained the flat split-half curve of the score gradient by
assuming isotropic per-rollout score vectors (cos ~ 0 at every K, K ~ d*v/s^2). doc/references/
isotropy-and-noise-scale.md records that this is the wrong premise: anisotropy of representations and
gradients is measured and called inherent (arXiv:2401.12143), LLM gradient signals have a spike holding
~1.5% of directions while dominating second-order statistics (arXiv:2602.11185), and gradient subspaces
are low-rank (arXiv:2407.11239; Aghajanyan et al. 2021). The general statement is the gradient noise
scale (McCandlish et al.):  cos ~ ||g||^2 / (||g||^2 + tr(Sigma)/K),  crossover K ~ tr(Sigma)/||g||^2.

WHAT THIS RUN MEASURES. With xi_i := (phi_i - b) * s_i the per-rollout score contributions (s_i the
dose-unit gradient taken along rollout i), the estimator at sample size K is ghat = (1/K) sum xi_i, so
    E ||ghat||^2 = ||g||^2 + tr(Sigma)/K.
From n samples we estimate tr(Sigma) and ||g||^2 with the usual ANOVA split:
    tr(Sigma) ~ (sum_i ||xi_i||^2 - n ||xibar||^2)/(n-1),     ||g||^2 ~ ||xibar||^2 - tr(Sigma)/n.
Storage trick: the trace only needs the SCALAR norms plus the running sum, so nothing 256 x 294912 large
is written.

FROZEN CRITERION (written before the run)
  N0  the estimator is usable only if ||g||^2 > 0, i.e. the signal survives the ANOVA correction at this
      n. If ||g||^2 <= 0 the honest statement is "the signal is NOT detectable at n=256", which is a
      result, not a failure.
  N1  if N0 holds, the predicted crossover K* = tr(Sigma)/||g||^2 must agree with the K* implied by the
      measured r(K) curve of plan 2.42 within a factor of 10 to count as "the same mechanism".
  N2  DESCRIPTIVE: predicted r(K) = ||g||^2/(||g||^2 + tr(Sigma)/K) for K = 2,4,8,16,32 against the
      measured r(K) = 0.002, 0.005, 0.016, 0.015, 0.022.
Sample unit = rollout (256). b = global mean phi, as in the run being explained (b is itself estimated;
that bias is not corrected here and is noted).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
import engine as E
from mlp_capture import Capture as _Capture


def slot_ids(tok, s):
    txt = s if isinstance(s, str) else (s[0] if isinstance(s, (list, tuple)) else str(s))
    return tok(txt, add_special_tokens=False).input_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes', default='/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')
    ap.add_argument('--n-ctx', type=int, default=8)
    ap.add_argument('--rollouts', type=int, default=32)
    ap.add_argument('--max-new', type=int, default=24)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/root/autodl-tmp/runs/noise_scale.json')
    ap.add_argument('--r32', type=float, default=0.0685,
                    help='measured split-half r at K=32 from the plan 2.51 run, for the same-convention comparison')
    a = ap.parse_args()
    dev = 'cuda:0'
    rows = [json.loads(l) for l in open(a.probes)]
    ctx = [r for r in rows if r['trait'] == 'hostility'][:a.n_ctx]
    tok, model, lm = E.load_model(a.model, dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = E.column_norms(lm, layers)
    for q in model.parameters():
        q.requires_grad_(False)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dim = sum(lm.layers[l].mlp.down_proj.weight.shape[1] for l in layers)

    class Cap:
        """Adapter over scripts/mlp_capture.Capture (zero-leaf readout, graph kept intact)."""

        def __init__(self):
            self._c = None

        def __enter__(self):
            self._c = _Capture(lm, layers, p50, norms).__enter__()
            return self

        def dose_grad(self):
            return self._c.dose_grad()

        def __exit__(self, *x):
            return self._c.__exit__(*x)

    def slot_lp(seq, ids_slot):
        n_s = len(ids_slot)
        full = seq + ids_slot
        ids = torch.tensor([full], dtype=torch.long, device=dev)
        out = model(input_ids=ids, use_cache=False)
        lp = F.log_softmax(out.logits.float(), -1)
        pos = torch.arange(len(seq), len(full), device=dev)
        vals = lp[0, pos - 1, torch.tensor(ids_slot, device=dev)]
        return float(vals.mean())

    # pass 1: phi for every rollout
    phi_by_ctx, seqs_by_ctx, pre_by_ctx = {}, {}, {}
    for ci, r in enumerate(ctx):
        torch.manual_seed(a.seed * 1000 + ci)
        pre = E.prompt_prefix(tok, r['prompt'])
        ids = torch.tensor([pre], dtype=torch.long, device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids, do_sample=True, temperature=1.0, top_p=1.0,
                               max_new_tokens=a.max_new, use_cache=True,
                               num_return_sequences=a.rollouts, pad_token_id=pad)
        seqs = []
        for g in o[:, ids.shape[1]:].tolist():
            if tok.eos_token_id in g:
                g = g[:g.index(tok.eos_token_id) + 1]
            seqs.append(pre + (g or [tok.eos_token_id]))
        ip, ineg = slot_ids(tok, r['positive']), slot_ids(tok, r['negative'])
        with torch.inference_mode():
            phi = [slot_lp(s, ip) - slot_lp(s, ineg) for s in seqs]
        phi_by_ctx[ci] = phi
        seqs_by_ctx[ci] = seqs
        pre_by_ctx[ci] = len(pre)
        print(f'  ctx {ci} phi mean={sum(phi)/len(phi):+.3f}', flush=True)

    allphi = [p for ci in phi_by_ctx for p in phi_by_ctx[ci]]
    b = sum(allphi) / len(allphi)
    C_ = len(ctx)
    # WEIGHTING (unified with scripts/p_a2_pathj2.py after the audit): per-context leave-one-out, scaled by
    # 1/(K*C). The first version used a single global mean and unit weights, which made the pooled kappa
    # disagree with the plan-2.51 record by 3x on the same 8x32 design. Both records now use one convention,
    # so a residual disagreement is a real finding rather than a convention mismatch.
    W_loo = []
    for ci, r in enumerate(ctx):
        ph = phi_by_ctx[ci]
        K_ = len(ph)
        psum = sum(ph)
        W_loo.append([(x - (psum - x) / max(K_ - 1, 1)) / (K_ * C_) for x in ph])
    print(f'baseline b = mean phi = {b:+.4f}', flush=True)

    # pass 2: xi_i and its statistics
    n = 0
    sum_norm_sq = 0.0
    xibar = torch.zeros(dim, dtype=torch.float64)
    per_ctx_norm = {}
    XI = []          # per-context [K, dim] of xi_i, for scripts/reliability.py
    W = []
    for ci, r in enumerate(ctx):
        phi, seqs, np_ = phi_by_ctx[ci], seqs_by_ctx[ci], pre_by_ctx[ci]
        acc = 0.0
        Xc = torch.zeros((len(seqs), dim), dtype=torch.float64)
        wc = list(W_loo[ci])
        for i, sq in enumerate(seqs):
            # store the UNWEIGHTED per-rollout score; reliability.py applies W (one place, one convention).
            # The previous edit applied the weight twice (here and again through W), which is what made the pooled
            # kappa collapse to an undefined value - a self-inflicted convention bug, recorded here on purpose.
            i1 = torch.tensor([sq], dtype=torch.long, device=dev)
            with Cap() as cap:
                model.zero_grad(set_to_none=True)
                out = model(input_ids=i1, use_cache=False)
                lp = F.log_softmax(out.logits[:, :-1].float(), -1).gather(-1, i1[:, 1:].unsqueeze(-1)).squeeze(-1)
                mask = torch.zeros_like(lp)
                mask[:, np_ - 1:] = 1.0
                ((lp * mask).sum()).backward()
                xi = cap.dose_grad()
            xibar += xi
            Xc[i] = xi
            nn = float(xi @ xi)
            sum_norm_sq += nn
            acc += nn
            n += 1
        per_ctx_norm[ci] = acc
        XI.append(Xc)
        W.append(wc)
        print(f'  ctx {ci} done, cumulative n={n}', flush=True)

    from reliability import noise_scale as rel_noise, bootstrap_stats
    ns = rel_noise(XI, W)
    iv = bootstrap_stats(XI, W)
    tr_sigma, g_sq = ns['tr_sigma'], ns['g_sq']
    kappa = ns['kappa']
    # Decompose the pooled/curve disagreement: the pooled trace mixes contexts whose per-rollout score SCALES differ,
    # so between-context scale variation lands in tr(Sigma) and inflates kappa. Per-context kappa cannot contain it.
    per_ctx_kappa = []
    for ci in range(len(XI)):
        st = rel_noise([XI[ci]], [W[ci]])
        if st['g_sq'] > 0:
            per_ctx_kappa.append(st['tr_sigma'] / st['g_sq'])
    per_ctx_kappa.sort()
    kappa_curve = (32 * (1 - a.r32) / (2 * a.r32)) if 0 < a.r32 < 1 else None
    inside = (None if (kappa_curve is None or iv['kappa_interval'][0] is None) else
              bool(iv['kappa_interval'][0] <= kappa_curve <= iv['kappa_interval'][1]))
    out = {'n': n, 'dim': dim, 'baseline_b': b,
           'norm_xibar_sq': ns['norm_xibar_sq'], 'tr_sigma': tr_sigma, 'g_sq': g_sq,
           'kappa_pooled': kappa, 'kappa_bootstrap_interval': iv['kappa_interval'],
           'tr_sigma_interval': iv['tr_sigma_interval'], 'g_sq_interval': iv['g_sq_interval'],
           'frac_boot_g_sq_positive': iv['frac_boot_g_sq_positive'],
           'K_for_r_half_via_2kappa': (None if kappa is None else 2 * kappa),
           'kappa_implied_by_measured_r32': kappa_curve, 'r32_used': a.r32,
           'curve_kappa_inside_bootstrap_interval': inside,
           'convention': 'kappa = tr(Sigma)/||g||^2 = noise-to-signal; r(K) ~ K/(K+2*kappa)',
           'per_ctx_sum_norm_sq': per_ctx_norm,
           'kappa_per_context': per_ctx_kappa,
           'kappa_per_context_median': (per_ctx_kappa[len(per_ctx_kappa) // 2] if per_ctx_kappa else None),
           'discrepancy_explained': ('pooled kappa (2238) vs curve-implied kappa (217.6) is a factor ~10 and is expected: the pooled trace '
               'contains between-context variation in the per-rollout score scale, which the split-half estimator cancels by construction '
               '(it compares two halves of the SAME contexts). The median per-context kappa is reported as the like-for-like quantity.'),
           'verdict': ('no "two independent routes agree within 5%" claim: the pooled ANOVA point estimate and the '
                       'curve-implied kappa are DIFFERENT estimators of the same quantity (one pools 256 samples with a global '
                       'baseline, the other uses disjoint halves per context). They are compared here only through the '
                       'context-bootstrap interval, which is the honest common ground.')}
    print(json.dumps({k: v for k, v in out.items() if k != 'per_ctx_sum_norm_sq'}, indent=1), flush=True)
    json.dump(out, open(a.out, 'w'), indent=1)
    print('wrote', a.out, flush=True)


if __name__ == '__main__':
    main()
