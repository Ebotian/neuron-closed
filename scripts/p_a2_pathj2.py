#!/usr/bin/env python3
"""P-A' (plan 2.42): the path version of J2. Same construct, only the measure changes.

THE METHOD PROBLEM THIS FIXES (not a knob). Plan 2.40 differentiated a saturated toxicity judge:
phi came out at 0.0405 +- 0.01 on base rollouts, so the REINFORCE weights (phi - b) were ~1e-4 and the
gradient was pure sampling noise (split-half cos -0.073). That is not "K too small": a functional whose
dynamic range is 1e-4 cannot supply a first-order structure at any affordable K. The mistake is the
SCALE of the functional, not the estimator, the dose, the baseline or the judge.

THE FIX. Our validated construct is J2 = the log-mass margin between a hostility slot and a praise slot.
Teacher-forced J2 conditions that margin on the FIXED prompt prefix. The path version conditions it on
the model's OWN continuation:

    phi(y) = (1/|s|) sum_{t in slot} [ log p(t+ | x, y) - log p(t- | x, y) ]      (teacher-forced on y)

so phi is in NATS with real spread, not a saturated probability. Then

    d*_path = E_{x~D} E_{y~p^u}[ (phi(y) - E phi) * grad_u log p^u(y) ]

and the closed-form algebra is untouched (u = WB^-1 d*, budget sqrt(u'Mu), Q(S), greedy). The ONLY
difference between g_path and g_teacher is the measure change (teacher-forced prefix vs on-policy
rollouts) - no judge, no threshold, no dose convention. So cos(g_path, g_teacher) is a direct
measurement of "how much does the measure change cost", which is exactly the trait-vs-behaviour gap
in its cleanest available form.

UNIFIED FUNCTIONAL AND NORMALISATION (plan 2.48, task-3 of audit-response §12).
Every quantity in this file is an average over rollouts i = 1..K and over contexts c = 1..C, and every
readout is a mean over the slot's own tokens. Written once, here, and nothing else is allowed:

    phi(y)          = (1/|s+|) sum_{t in s+} log p(t | x, y)  -  (1/|s-|) sum_{t in s-} log p(t | x, y)
                      ------------------------------------        ------------------------------------
                          mean over the POSITIVE slot tokens              mean over the NEGATIVE slot tokens
    g_path          = (1/C) sum_c (1/K) sum_i grad_u phi(y_ci)
    g_score         = (1/C) sum_c (1/K) sum_i ( phi(y_ci) - b ) grad_u log p(y_ci)
    g_teach         = (1/C) sum_c grad_u [ mean over SUFFIX tokens of ( log p(s+|x) - log p(s-|x) ) ]

Two distinctions that look like inconsistencies and are NOT, so nobody "fixes" them later:
  * g_score uses grad_u log p(y) with log p the density of the WHOLE sequence (a SUM over tokens), because
    the measure is over complete sequences; the readouts phi and g_teach divide by their own token counts.
  * b is a single scalar baseline shared by all contexts, so that (1/C)sum_c(1/K)sum_i(phi-b)s is the
    estimator of the same 1/C sum_c grad E[phi].

WHAT WAS WRONG BEFORE THIS (audit-response §2-3): the pathwise branch divided by |slot| a second time on top
of .mean() (so L^-2 while phi is L^-1), the teacher branch summed instead of averaging over suffix tokens,
the score branch carried 1/K but not 1/C, and g_pw was re-initialised inside the context loop so only the
last context survived. All four are fixed here.

FROZEN CRITERIA (written 2026-10-30 before the run)
  C0  phi must have dynamic range: total sd(phi) >= 0.05 nat AND within-context sd >= 0.01 nat.
      Violated -> the functional is again near-constant, STOP and say so (no knob turning).
  R   reliability curve: split-half cos over K = 2,4,8,16,32 rollouts per context, 200 splits each.
      Report r(K) and the K* implied for r = 0.5 by Spearman-Brown r(K) = K/(K+K0).
      PRE-REGISTERED FAILURE CONDITION: if r(32) < 0.5 the path gradient is NOT a usable direction at
      this budget, and the finding is "not estimable at this scale" - do not tune further.
  C1' DESCRIPTIVE, and only readable if R passes: cos(g_path, g_teacher), plus per-layer.
      Prediction: < 0.5, i.e. the measure change moves the direction substantially.
  SELF batch-vs-single gate on the slot log-prob (|d| <= 5e-2, per-layer cos >= 0.999) before any number.
Sample unit = context (8). Error scale = split-half spread; phi spread in nats.
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
import engine as E
from mlp_capture import Capture as _Capture
from core.units import column_norms_dev, dose_scale


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
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='/root/autodl-tmp/runs/p_a2_path.json')
    ap.add_argument('--selftest', action='store_true')
    ap.add_argument('--pathwise', action='store_true')
    ap.add_argument('--save-gram', default=None, help='write the n x n Gram to this JSON path')
    ap.add_argument('--fp32', action='store_true', help='run the model in float32: removes the shared bf16 batch bias that moves r(K)')
    ap.add_argument('--inject-double-division', action='store_true',
                    help='sensitivity test: put the L^-2 slot normalisation back, so --branch-selftest must FAIL')
    ap.add_argument('--branch-selftest', action='store_true',
                    help='task-3 gate: FD must reproduce the SHIPPED pathwise and teacher terms (fp32)')
    a = ap.parse_args()
    dev = 'cuda:0'
    torch.manual_seed(a.seed)

    rows = [json.loads(l) for l in open(a.probes)]
    ctx = [r for r in rows if r['trait'] == 'hostility'][:a.n_ctx]
    print(f'prompts: {len(ctx)} hostility', flush=True)
    tok, model, lm = E.load_model(a.model, dev)
    if a.fp32:
        model = model.float()
        lm = E.model_lm(model)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = column_norms_dev(lm, layers, dev)
    SC = dose_scale(p50, norms, layers)   # the ONE place the dose unit is formed
    for q in model.parameters():
        q.requires_grad_(False)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    class Cap:
        """Adapter over scripts/mlp_capture.Capture (zero-leaf readout, graph kept intact)."""

        def __init__(self):
            self._c = None

        def __enter__(self):
            self._c = _Capture(lm, layers, p50, norms).__enter__()
            return self

        def grad_sum(self):
            return self._c.grad_sum()

        def __exit__(self, *x):
            return self._c.__exit__(*x)

    def slot_lp(seqs, slot_starts, ids_slot):
        """Teacher-forced log p of the slot tokens appended after each sequence. seqs: list of id lists
        (prompt+continuation). Returns per-row mean logp over slot tokens."""
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

    def phi_of(seqs, r):
        ip, ineg = slot_ids(tok, r['positive']), slot_ids(tok, r['negative'])
        return slot_lp(seqs, None, ip) - slot_lp(seqs, None, ineg)

    # ---------------- SELF gate on the slot log-prob path
    if a.selftest:
        r = ctx[0]
        pre = E.prompt_prefix(tok, r['prompt'])
        seqs = [pre + [40, 14952, 678]]
        with torch.inference_mode():
            b = phi_of(seqs, r)
            singles = []
            for s in seqs:
                singles.append(phi_of([s], r))
        d = abs(float(b[0]) - float(singles[0]))
        # batched vs single over 2 rows
        seqs2 = [pre + [40, 14952, 678], pre + [40, 14952, 678, 46253]]
        with torch.inference_mode():
            b2 = phi_of(seqs2, r)
            s2 = torch.stack([phi_of([x], r)[0] for x in seqs2])
        d2 = float((b2 - s2).abs().max())
        cos = float(F.cosine_similarity(b2.unsqueeze(0), s2.unsqueeze(0)))
        ok = d <= 5e-2 and d2 <= 5e-2 and cos >= 0.999
        print(f'SELF1 |d|={d:.2e}  SELF2 |d|max={d2:.2e} cos={cos:.6f} -> {"PASS" if ok else "FAIL"}', flush=True)
        print(f'SELF phi values batch={[round(float(x),4) for x in b2]} single={[round(float(x),4) for x in s2]}', flush=True)
        return 0 if ok else 1

    from estimators.pathwise import pathwise_grad, slotted_mean_logp
    from estimators.teacher import teacher_grad

    def pathwise_term(sq, ids_slot, sgn, weight):
        return pathwise_grad(model, lm, layers, p50, norms, sq, ids_slot, sgn, weight, dev, scale=SC)

    def teacher_term(prompt, slot_txt, sgn, weight):
        return teacher_grad(model, lm, tok, layers, p50, norms, prompt, slot_txt, sgn, weight, dev, scale=SC)

    if a.branch_selftest:
        """REAL gate for the four defects fixed in plan 2.48, replacing the vacuous one.

        The deleted version compared phi_of against slot_lp with identical arguments - the same function called twice,
        so 0.00e+00 was guaranteed and it could not have detected anything. This one differentiates the SHIPPED helpers
        (pathwise_term / teacher_term, the very functions the main loop calls) against fp32 finite differences of the
        functional they claim to differentiate, using a deliberately MULTI-TOKEN slot so that a second division by |slot|
        would show up as a factor |slot|, and it separately checks that g_pw accumulates across contexts.
        """
        model = model.float()
        lm = E.model_lm(model)
        p50 = E.estimate_p50(tok, model, lm, layers, dev)
        norms = column_norms_dev(lm, layers, dev)
        p_ = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        r = ctx[0]
        pre = E.prompt_prefix(tok, r['prompt'])
        SLOT_TXT = 'deeply hostile and cruel'
        slot_t = tok(SLOT_TXT, add_special_tokens=False).input_ids
        print(f'BRANCH-SELFTEST slot tokens={len(slot_t)}', flush=True)
        torch.manual_seed(0)
        ids0 = torch.tensor([pre], dtype=torch.long, device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids0, do_sample=True, temperature=1.0, top_p=1.0,
                               max_new_tokens=10, use_cache=True, num_return_sequences=1, pad_token_id=p_)
        seq = pre + o[0, ids0.shape[1]:].tolist()

        def steer_phi(fn):
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
                        return fn()
                finally:
                    for h in hs:
                        h.remove()
            return phi

        def read_path():
            full = seq + slot_t
            j1 = torch.tensor([full], dtype=torch.long, device=dev)
            out = model(input_ids=j1, use_cache=False)
            lp = F.log_softmax(out.logits[:, :-1].float(), -1)
            pos = torch.arange(len(seq), len(full), device=dev)
            return float(lp[0, pos - 1, torch.tensor(slot_t, device=dev)].mean())

        def read_teach():
            rid, rpre, rsuf = E.candidate_ids(tok, r['prompt'], SLOT_TXT, dev, 384)
            i1 = (rid.to(dev) if torch.is_tensor(rid) else torch.tensor(rid, device=dev)).view(1, -1).long()
            out = model(input_ids=i1, use_cache=False)
            lp = F.log_softmax(out.logits[:, :-1].float(), -1).gather(-1, i1[:, 1:].unsqueeze(-1)).squeeze(-1)
            mask = torch.zeros_like(lp)
            mask[:, int(rpre) - 1:] = 1.0
            return float((lp * mask).sum() / mask.sum())

        rows_s, ok = [], True
        v_pw = pathwise_term(seq, slot_t, +1.0, 1.0)
        v_t = teacher_term(r['prompt'], SLOT_TXT, +1.0, 1.0)
        for tag, read, vv in (('pathwise', read_path, v_pw), ('teacher', read_teach, v_t)):
            phi = steer_phi(read)
            for L in (0, 13, 31):
                n = int(torch.argmax(vv[L].abs()))
                best = None
                for eps in (1e-3, 1e-2):
                    fd = (phi({(L, n): eps}) - phi({(L, n): -eps})) / (2 * eps)
                    rel = abs(fd - float(vv[L][n])) / max(abs(float(vv[L][n])), 1e-6)
                    best = rel if best is None else min(best, rel)
                    rows_s.append({'term': tag, 'layer': L, 'neuron': n, 'analytic': float(vv[L][n]),
                                   'fd': fd, 'eps': eps, 'rel_err': rel})
                print(f'  {tag:9s} L{L:<2d} n={n:<6d} analytic={float(vv[L][n]):+11.4f} '
                      f'min rel_err over eps={best:.2e}', flush=True)
                if best >= 0.02:
                    ok = False
        print('BRANCH-SELFTEST', 'PASS' if ok else 'FAIL', flush=True)
        Path(a.out.replace('.json', '_branch_selftest.json')).parent.mkdir(parents=True, exist_ok=True)
        json.dump({'rows': rows_s, 'passed': ok, 'slot_tokens': len(slot_t),
                   'injected_double_division': bool(a.inject_double_division),
                   'criterion': 'FD (fp32, min over eps) must reproduce the shipped helper to <2%'},
                  open(a.out.replace('.json', '_branch_selftest.json'), 'w'), indent=1)
        return 0 if ok else 1

    g_path = {l: torch.zeros(lm.layers[l].mlp.down_proj.weight.shape[1], dtype=torch.float64, device=dev) for l in layers}
    g_teach = {l: torch.zeros_like(g_path[l]) for l in layers}
    g_path_in = {l: torch.zeros_like(g_path[l]) for l in layers}  # in-sample baseline (biased by 1-1/K)
    g_pw = {l: torch.zeros_like(g_path[l]) for l in layers}
    rec = {'ctx': [], 'phi': {}, 'texts': {}}
    splits = {}
    w_loo_ctx = {}
    pw_snap = {}
    allphi = []
    for ci, r in enumerate(ctx):
        torch.manual_seed(a.seed * 1000 + ci)
        pre = E.prompt_prefix(tok, r['prompt'])
        ids = torch.tensor([pre], dtype=torch.long, device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids, do_sample=True, temperature=a.temp, top_p=1.0,
                               max_new_tokens=a.max_new, use_cache=True,
                               num_return_sequences=a.rollouts, pad_token_id=pad)
        seqs = []
        for g in o[:, ids.shape[1]:].tolist():
            if tok.eos_token_id in g:
                g = g[:g.index(tok.eos_token_id) + 1]
            seqs.append(pre + (g or [tok.eos_token_id]))
        with torch.inference_mode():
            phi = phi_of(seqs, r)
        ph = [float(x) for x in phi]
        allphi += ph
        rec['ctx'].append(r['id'])
        rec['phi'][r['id']] = ph
        rec['texts'][r['id']] = [tok.decode(s[len(pre):], skip_special_tokens=True)[:160] for s in seqs]
        print(f'  [{ci}] {r["id"]} phi mean={sum(ph)/len(ph):+.3f} sd={(sum((x-sum(ph)/len(ph))**2 for x in ph)/len(ph))**.5:.3f}', flush=True)

        base = sum(ph) / len(ph)
        # Per-rollout score vectors: needed for the reliability curve r(K). 302 MB total, fine.
        from estimators.score import per_rollout_scores   # the ONE implementation of the score term
        S = per_rollout_scores(model, lm, layers, p50, norms, seqs, pad, dev, len(pre), scale=SC)
        # --- task-4: two baselines from the SAME per-rollout scores
        K_ = len(seqs)
        w_in = [(x - base) / (K_ * len(ctx)) for x in ph]
        phi_sum = sum(ph)
        w_loo = [(x - (phi_sum - x) / max(K_ - 1, 1)) / (K_ * len(ctx)) for x in ph]
        w_loo_ctx[ci] = w_loo
        off = 0
        for l in layers:
            d = lm.layers[l].mlp.down_proj.weight.shape[1]
            blk = S[:, off:off + d].double()
            g_path[l] += (torch.tensor(w_loo, device=dev).double().unsqueeze(1) * blk).sum(0)
            g_path_in[l] += (torch.tensor(w_in, device=dev).double().unsqueeze(1) * blk).sum(0)
            off += d
        if a.pathwise:
            from estimators.pathwise import pathwise_batch
            ip, ineg = slot_ids(tok, r['positive']), slot_ids(tok, r['negative'])
            items = []
            for sq in seqs:
                items.append((sq, ip, +1.0, 1.0 / (len(seqs) * len(ctx))))
                items.append((sq, ineg, -1.0, 1.0 / (len(seqs) * len(ctx))))
            acc = pathwise_batch(model, lm, layers, p50, norms, items, pad, dev, scale=SC, chunk=8)
            for l in layers:
                g_pw[l] += acc[l]
            pw_snap[ci] = {l: g_pw[l].clone() for l in layers}
        splits[ci] = S
        print(f'  [{ci}] {len(seqs)} per-rollout scores stored', flush=True)

        # teacher-forced J2 gradient on the same context (fixed prefix)
        for key, sgn in (('positive', +1.0), ('negative', -1.0)):
            v = teacher_term(r['prompt'], r[key] if isinstance(r[key], str) else r[key][0], sgn, 1.0 / len(ctx))
            for l in layers:
                g_teach[l] += v[l]
        print(f'  [{ci}] grads done', flush=True)

    mu = sum(allphi) / len(allphi)
    sd = (sum((x - mu) ** 2 for x in allphi) / len(allphi)) ** .5
    wsd = []
    for r in ctx:
        v = rec['phi'][r['id']]
        m = sum(v) / len(v)
        wsd.append((sum((x - m) ** 2 for x in v) / len(v)) ** .5)
    within = sum(wsd) / len(wsd)
    rec['C0'] = {'phi_mean': mu, 'phi_sd_total': sd, 'phi_sd_within_mean': within}
    print(f'C0 phi mean={mu:+.4f} sd_total={sd:.4f} sd_within={within:.4f}', flush=True)

    # Reliability via scripts/reliability.py: one convention for the whole repo.
    #   split-half compares TWO DISJOINT HALVES of size K/2  =>  r(K) ~ K/(K+2*kappa)
    #   kappa = tr(Sigma)/||g||^2 with an interval from resampling CONTEXTS (the independent unit).
    from reliability import (split_half_curve_fast, noise_scale, bootstrap_stats_fast,
                            gram_from_scores, icc_from_gram)
    W_all = [w_loo_ctx[ci] for ci in range(len(ctx))]
    curve = split_half_curve_fast([splits[ci] for ci in range(len(ctx))], W_all,
                             ks=tuple(k for k in (2, 4, 8, 16, 32) if k <= a.rollouts))
    ns = noise_scale([splits[ci] for ci in range(len(ctx))], W_all)
    iv = bootstrap_stats_fast([splits[ci] for ci in range(len(ctx))], W_all)
    # TENSORISED, ROBUST estimator: one Gram over all rollouts, then sums of squares -> ICC + F interval.
    # SS_w uses only DIFFERENCES of rollouts, so the batched bf16 shared bias cancels (lean: ssw_shift_invariant).
    G = gram_from_scores([splits[ci] for ci in range(len(ctx))], W_all)
    icc = icc_from_gram(G, [int(splits[ci].shape[0]) for ci in range(len(ctx))], k=a.rollouts)
    rec['ICC'] = icc
    if a.save_gram:
        json.dump({'G': G.cpu().tolist(), 'group_sizes': [int(splits[ci].shape[0]) for ci in range(len(ctx))],
                   'k': a.rollouts, 'seed': a.seed}, open(a.save_gram, 'w'))
    print('ICC  MSw=%.6f [%s]  MSb-derived ||g||^2=%.6f  ICC=%s interval=%s' % (
        icc['tr_sigma'], icc['tr_sigma_interval'], icc['g_sq'],
        None if icc['icc'] is None else round(icc['icc'], 4),
        None if icc['icc_interval'] is None else [round(x, 4) for x in icc['icc_interval']]), flush=True)
    r32 = curve.get(a.rollouts, {}).get('r_mean')
    rec['R'] = {'r_of_K': curve, 'noise_scale': ns, 'kappa_interval': iv,
                'convention': 'r(K) ~ K/(K+2*kappa), so r=0.5 needs K = 2*kappa, NOT kappa. kappa is the noise-to-signal ratio.',
                'tr_sigma_interval': iv['tr_sigma_interval'], 'g_sq_interval': iv['g_sq_interval'],
                'tr_sigma_bca': iv['tr_sigma_bca'], 'g_sq_bca': iv['g_sq_bca'], 'kappa_bca': iv['kappa_bca'],
                'tr_sigma_bca': iv['tr_sigma_bca'], 'g_sq_bca': iv['g_sq_bca'], 'kappa_bca': iv['kappa_bca'],
                'frac_boot_g_sq_positive': iv['frac_boot_g_sq_positive'],
                'r_at_Kmax': r32}
    print('R r(K)=%s' % {k: round(v['r_mean'], 4) for k, v in curve.items()}, flush=True)
    print('R trSigma=%.4e [%s, %s]  ||g||^2=%.4e [%s, %s]  kappa=%s  K(r=0.5)=2*kappa=%s' % (
        ns['tr_sigma'], iv['tr_sigma_interval'][0], iv['tr_sigma_interval'][1],
        ns['g_sq'], iv['g_sq_interval'][0], iv['g_sq_interval'][1],
        None if ns['kappa'] is None else round(ns['kappa'], 1),
        None if ns['kappa'] is None else round(2 * ns['kappa'], 1)), flush=True)
    print('R bootstrap: fraction of resamples with g_sq>0 = %.3f' % iv['frac_boot_g_sq_positive'], flush=True)
    if r32 is not None:
        # solve r = K/(K+2k) for k  =>  k = K(1-r)/(2r).  (The first version had it inverted: 1.2 instead of 217.)
        kappa_from_curve = (a.rollouts * (1 - r32) / (2 * r32)) if 0 < r32 < 1 else None
        rec['R']['kappa_implied_by_curve'] = kappa_from_curve
        rec['R']['consistency'] = (
            'the curve-implied kappa and the pooled ANOVA kappa are different estimators (per-context halves vs one '
            'pool of 256 samples with a global baseline); report both, and treat the pooled point estimate as '
            'unreliable when it falls outside the context-bootstrap interval')
        print('R kappa implied by r(Kmax)=%.4f is %s (so K(r=0.5)=%s); pooled says %s; kappa interval [%s, %s]' % (
            r32, None if kappa_from_curve is None else round(kappa_from_curve, 1),
            None if kappa_from_curve is None else round(2 * kappa_from_curve, 1),
            None if ns['kappa'] is None else round(ns['kappa'], 1),
            iv['kappa_interval'][0], iv['kappa_interval'][1]), flush=True)

    def flat(d):
        return torch.cat([d[l] for l in layers])

    if a.pathwise and len(pw_snap) >= 2:
        # The bug this must catch (audit §2) was g_pw being re-initialised INSIDE the context loop, which makes the
        # final object equal to the LAST context's contribution instead of the sum over contexts. Cumulative snapshots
        # therefore cannot be summed (that was my first, wrong, check); the discriminating comparison is
        #   final  vs  the last context's INCREMENT, and  final  vs  sum of increments.
        snaps = [pw_snap[ci] for ci in sorted(pw_snap)]
        inc = []
        for i in range(len(snaps)):
            prev = snaps[i - 1] if i > 0 else {l: torch.zeros_like(snaps[0][l]) for l in layers}
            inc.append({l: snaps[i][l] - prev[l] for l in layers})
        dev_sum = max(float((g_pw[l] - sum(inc[i][l] for i in range(len(inc)))).abs().max()) for l in layers)
        dev_last = max(float((g_pw[l] - inc[-1][l]).abs().max()) for l in layers)
        scale = max(float(g_pw[l].abs().max()) for l in layers)
        rec['pw_additivity'] = {'contexts': len(snaps),
                                'max_dev_from_sum_of_increments': dev_sum,
                                'max_dev_from_last_increment': dev_last,
                                'scale': scale,
                                'passes': dev_sum < 1e-9 and dev_last > 1e-6 * scale}
        print('PW-ADDITIVITY ctx=%d |final-sum(inc)|=%.2e |final-last_inc|=%.2e scale=%.2e -> %s' % (
            len(snaps), dev_sum, dev_last, scale, 'PASS' if rec['pw_additivity']['passes'] else 'FAIL'), flush=True)

    gp, gt = flat(g_path), flat(g_teach)
    cos = float(F.cosine_similarity(gp.unsqueeze(0), gt.unsqueeze(0)))
    per = {l: float(F.cosine_similarity(g_path[l].unsqueeze(0), g_teach[l].unsqueeze(0))) for l in layers}
    gpi = flat(g_path_in)
    rec['baseline_compare'] = {
        'cos_loo_vs_insample': float(F.cosine_similarity(gp.unsqueeze(0), gpi.unsqueeze(0))),
        'norm_ratio_loo_over_insample': float(gp.norm() / gpi.norm()) if float(gpi.norm()) > 0 else None,
        'K': a.rollouts, 'n_ctx': len(ctx),
        'predicted_bias_factor_insample': 1.0 - 1.0 / a.rollouts,
        'note': 'the in-sample baseline b=mean(phi) yields E[(phi-b)s]=(1-1/K) g, so g_in underestimates g by that factor; the leave-one-out b_{-i} is unbiased'}
    print('BASE cos(loo,in-sample)=%+.6f  |loo|/|in|=%.6f  predicted(1-1/K)=%.6f' % (
        rec['baseline_compare']['cos_loo_vs_insample'],
        rec['baseline_compare']['norm_ratio_loo_over_insample'] or 0.0,
        rec['baseline_compare']['predicted_bias_factor_insample']), flush=True)
    rec['C1p'] = {'cos_path_teach': cos, 'cos_per_layer': per,
                  'norm_path': float(gp.norm()), 'norm_teach': float(gt.norm())}
    if a.pathwise:
        gw = flat(g_pw)
        rec['C2_split'] = {
            'cos_pathwise_teach': float(F.cosine_similarity(gw.unsqueeze(0), gt.unsqueeze(0))),
            'cos_pathwise_score': float(F.cosine_similarity(gw.unsqueeze(0), gp.unsqueeze(0))),
            'norm_pathwise': float(gw.norm()), 'norm_score': float(gp.norm()),
            'ratio_score_over_pathwise': float(gp.norm() / gw.norm()) if float(gw.norm()) > 0 else None,
            'cos_per_layer_pathwise_teach': {l: float(F.cosine_similarity(g_pw[l].unsqueeze(0), g_teach[l].unsqueeze(0))) for l in layers},
        }
        print('C2 cos(pw,teach)=%+.4f cos(pw,score)=%+.4f |pw|=%.2f |score|=%.2f ratio=%.3f' % (
            rec['C2_split']['cos_pathwise_teach'], rec['C2_split']['cos_pathwise_score'],
            rec['C2_split']['norm_pathwise'], rec['C2_split']['norm_score'],
            rec['C2_split']['ratio_score_over_pathwise'] or 0), flush=True)
    print(f'C1p cos(g_path,g_teach)={cos:+.4f} per-layer min/max {min(per.values()):+.4f}/{max(per.values()):+.4f}', flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, 'w'))
    blob = {'g_path': g_path, 'g_teach': g_teach, 'p50': p50}
    if a.pathwise:
        blob['g_pw'] = g_pw
    torch.save(blob, a.out.replace('.json', '.pt'))
    print('wrote', a.out, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
