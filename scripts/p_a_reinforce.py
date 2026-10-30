#!/usr/bin/env python3
"""P-A (plan 2.40): the trait IS the behaviour -- score-function gradient.

WHY THIS RUN EXISTS. Our J2 readout is a functional of the CONDITIONAL at a frozen prefix
(teacher-forced lexicon margin). A trait, as designed, is a functional of the PATH MEASURE:

    T(u) = E_{x~D} E_{y~p_theta^u(.|x)} [ phi(y) ]

If trait == behaviour then the gradient to use is the score-function (REINFORCE) gradient

    grad_u T = E_y [ (phi(y) - b) * grad_u log p^u(y) ],   grad_u log p^u = W_down[:,n]' grad_h log p

which needs the SAME down_proj pre-hooks as before, taken along sampled rollouts instead of a fixed
prefix. The algebra (u = WB^-1 d*, Q(S) = g_S' M_SS^-1 g_S, greedy over layers, budget sqrt(u'Mu)) is
UNCHANGED; only d* changes from "teacher-forced log-margin gradient" to a path-functional gradient.
Prior art (not our idea): Policy Gradient Steering, arXiv:2607.27574v1. What we add is the budget
metric B and budget-constrained support selection.

FROZEN CRITERIA (written 2026-10-29 before any run; not edited after seeing numbers)
  C0  instrument sanity: mean phi over base rollouts in (0, 0.5) and frac(phi>0.5) < 0.5.
      Violated -> STOP, the gauge is unusable on this model/material.
  SELF numeric regression (not a science criterion, a correctness gate): the batched backward must
      reproduce the single-sequence backward -- |d logp| <= 5e-2 and per-layer cos >= 0.999.
      AMENDED 2026-10-30 BEFORE any science number was read: the original gate was 1e-4 / 0.9999,
      but bf16 batch-vs-single kernel noise on the same sequence is measured at 5.6e-3 per 4 tokens,
      so 1e-4 was unattainable by construction. What the gate is FOR is catching mask/alignment bugs,
      which move logp by whole units (75 vs 18) and cos to 0.15-0.76, so the amended threshold still
      separates them by two orders of magnitude.
      This exists because plan 2.34 taught us that a refactor can move every value by 3-13% while
      all cosines stay at 0.995.
  C1  DESCRIPTIVE, NOT A VERDICT: cos(g_RE, g_J2) over 32x9216, plus per-layer cos.
      Registered prediction: cos < 0.30 (behaviour functional and readout are different objectives).
  C2  REGISTERED, separate script: at matched budget the g_RE arm lifts mean phi over the g_J2 arm in
      >= 7/8 contexts. Normalised u = b * g / ||g||_2 (L2 in dose space). DEVIATION ON RECORD: the
      metric-correct ||g||_M needs the Gram and is NOT used here, so absolute b is not comparable to
      committed b values; both arms share the rule, so the comparison is internally fair.

Sample unit = context (8). Error scale = exact binomial on the sign test; per-layer cos spread.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
import engine as E


def load_gauges(root, dev):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    out = {}
    for sub, name in [('unitary_toxic-bert', 'unitary'),
                      ('s-nlp_roberta_toxicity_classifier', 'snlp')]:
        d = Path(root) / sub
        if not d.exists():
            continue
        tk = AutoTokenizer.from_pretrained(d, local_files_only=True)
        m = AutoModelForSequenceClassification.from_pretrained(d, local_files_only=True).to(dev).eval()
        out[name] = (tk, m)
    return out


def p_toxic(gauges, texts, dev, batch=8):
    res = {}
    for name, (tk, m) in gauges.items():
        vals = []
        for i in range(0, len(texts), batch):
            enc = tk(texts[i:i + batch], return_tensors='pt', padding=True, truncation=True,
                     max_length=256).to(dev)
            with torch.inference_mode():
                lg = m(**enc).logits.float()
            p = torch.sigmoid(lg[:, 0]) if lg.shape[-1] == 1 else F.softmax(lg, -1)[:, 1]
            vals.extend(float(x) for x in p)
        res[name] = vals
    return res


from mlp_capture import Capture  # noqa: E402  (single corrected implementation; see its docstring)

def score_batch(model, ids, lens, weights, n_prompt, mean_over_suffix=False):
    """Weighted sum over samples of sum_t log p(t) over CONTINUATION positions only.

    Two masks, and they are NOT interchangeable: `pad_mask` is the attention mask (prompt INCLUDED,
    padding excluded) and `score_mask` is the loss mask (continuation only). Using the loss mask as
    the attention mask cuts attention over the prompt and changes every log-prob -- the SELF gate
    caught it (batch -75.09 vs single -17.80 for 8 continuation tokens).
    """
    pad_mask = torch.zeros_like(ids)
    score_mask = torch.zeros_like(ids)
    for i, L in enumerate(lens):
        pad_mask[i, :L] = 1
        score_mask[i, n_prompt:L] = 1
    out = model(input_ids=ids, attention_mask=pad_mask, use_cache=False)
    logits = out.logits[:, :-1].float()
    tgt = ids[:, 1:]
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    per = (lp * score_mask[:, 1:].float()).sum(dim=1)
    if mean_over_suffix:
        # readouts divide by their own token count (the score term does NOT: it is the sequence density)
        per = per / score_mask[:, 1:].float().sum(dim=1).clamp(min=1)
    return (per * weights).sum(), per


def score_cont(model, cap, ids, n_prompt, weight):
    """Single-sequence reference path (validated by the K=1 run on 10-29)."""
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits[:, :-1].float()
    tgt = ids[:, 1:]
    lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    mask = torch.zeros_like(lp)
    mask[:, n_prompt - 1:] = 1.0
    tot = (lp * mask).sum()
    (weight * tot).backward()
    return float(tot.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes', default='/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')
    ap.add_argument('--gauges', default='/root/autodl-tmp/gauges')
    ap.add_argument('--n-ctx', type=int, default=8)
    ap.add_argument('--rollouts', type=int, default=8)
    ap.add_argument('--max-new', type=int, default=32)
    ap.add_argument('--temp', type=float, default=1.0)
    ap.add_argument('--out', default='/root/autodl-tmp/runs/p_a_k8.json')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    dev = 'cuda:0'
    torch.manual_seed(a.seed)

    rows = [json.loads(l) for l in open(a.probes)]
    host = [r for r in rows if r['trait'] == 'hostility']
    ctx = host[:a.n_ctx]
    print(f'prompts: {len(host)} hostility rows, using {len(ctx)}', flush=True)

    tok, model, lm = E.load_model(a.model, dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = E.column_norms(lm, layers)
    for q in model.parameters():
        q.requires_grad_(False)
    gauges = load_gauges(a.gauges, dev)
    print('gauges:', list(gauges), flush=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    if a.selftest:
        r = ctx[0]
        pre = E.prompt_prefix(tok, r['prompt'])
        ids0 = torch.tensor([pre], dtype=torch.long, device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids0, do_sample=True, temperature=1.0, max_new_tokens=8,
                               use_cache=True, num_return_sequences=2, pad_token_id=pad)
        seqs = [pre + g for g in o[:, ids0.shape[1]:].tolist()]
        T = max(len(x) for x in seqs)
        b_ = torch.full((2, T), pad, dtype=torch.long, device=dev)
        for i, x in enumerate(seqs):
            b_[i, :len(x)] = torch.tensor(x, device=dev)
        with Capture(lm, layers) as cap:
            model.zero_grad(set_to_none=True)
            lb, per = score_batch(model, b_, [len(x) for x in seqs], torch.ones(2, device=dev), len(pre))
            lb.backward()
            gb = cap.grad_sum()
        tot, gs = 0.0, {l: torch.zeros_like(gb[l]) for l in layers}
        for x in seqs:
            with Capture(lm, layers) as cap:
                model.zero_grad(set_to_none=True)
                tot += score_cont(model, cap, torch.tensor([x], device=dev), len(pre), 1.0)
                g1 = cap.grad_sum()
            for l in layers:
                gs[l] += g1[l]
        cs = {l: float(F.cosine_similarity(gb[l].unsqueeze(0), gs[l].unsqueeze(0)))
              if gb[l].norm() > 0 and gs[l].norm() > 0 else float('nan') for l in layers}
        rr = max(abs(float((gb[l] - gs[l]).norm()) / max(float(gs[l].norm()), 1e-12)) for l in layers)
        print(f'SELF logp batch={float(lb):.6f} single={tot:.6f} |diff|={abs(float(lb)-tot):.2e}', flush=True)
        print('SELF per-layer cos min %.6f max rel|dg| %.3e' % (min(cs.values()), rr), flush=True)
        for l in (0, 15, 31):
            print(f'SELF layer{l} cos={cs[l]:.6f} |gb|={float(gb[l].norm()):.4e} |gs|={float(gs[l].norm()):.4e}', flush=True)
        ok = abs(float(lb) - tot) <= 5e-2 and min(cs.values()) >= 0.999
        print('SELF', 'PASS' if ok else 'FAIL', flush=True)
        return 0 if ok else 1

    rec = {'ctx': [], 'phi': {}, 'texts': {}}
    keep = {}
    g_re = {l: torch.zeros(lm.layers[l].mlp.down_proj.weight.shape[1], dtype=torch.float64) for l in layers}
    g_j2 = {l: torch.zeros_like(g_re[l]) for l in layers}
    all_phi = []
    for ci, r in enumerate(ctx):
        prefix = E.prompt_prefix(tok, r['prompt'])
        ids = torch.tensor([prefix], dtype=torch.long, device=dev)
        with torch.inference_mode():
            o = model.generate(input_ids=ids, do_sample=True, temperature=a.temp, top_p=1.0,
                               max_new_tokens=a.max_new, use_cache=True,
                               num_return_sequences=a.rollouts, pad_token_id=pad)
        gen = o[:, ids.shape[1]:]
        texts = [tok.decode(g, skip_special_tokens=True) for g in gen]
        ph = p_toxic(gauges, texts, dev)
        phi = [sum(d[i] for d in ph.values()) / len(ph) for i in range(len(texts))]
        seqs = []
        for g in gen:
            g = g.tolist()
            if tok.eos_token_id in g:
                g = g[:g.index(tok.eos_token_id) + 1]
            g = g or [tok.eos_token_id]
            seqs.append(prefix + g)
        all_phi += phi
        keep[ci] = (prefix, seqs)
        rec['ctx'].append(r['id'])
        rec['phi'][r['id']] = ph
        rec['texts'][r['id']] = [t[:200] for t in texts]
        print(f'  [{ci}] {r["id"]} K={len(phi)} mean_phi={sum(phi)/len(phi):.4f} max={max(phi):.4f}', flush=True)

    c0_mean = sum(all_phi) / len(all_phi)
    c0_frac = sum(1 for x in all_phi if x > 0.5) / len(all_phi)
    rec['C0'] = {'mean_phi': c0_mean, 'frac_gt_0.5': c0_frac, 'n_rollouts': len(all_phi)}
    print(f'C0 mean_phi={c0_mean:.4f} frac>0.5={c0_frac:.4f} n={len(all_phi)}', flush=True)
    if not (0.0 < c0_mean < 0.5 and c0_frac < 0.5):
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rec, open(a.out, 'w'), indent=1)
        print('C0 VIOLATED -> stop', flush=True)
        return 2

    base = c0_mean
    offs, _o = [], 0
    for l in layers:
        _d = lm.layers[l].mlp.down_proj.weight.shape[1]
        offs.append((l, _o, _d))
        _o += _d
    per_dim = _o
    S_all, W_all = [], []
    for ci, r in enumerate(ctx):
        prefix, seqs = keep[ci]
        ph = rec['phi'][r['id']]
        n = len(next(iter(ph.values())))
        phi = [sum(d[i] for d in ph.values()) / len(ph) for i in range(n)]
        T = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long, device=dev)
        for i, x in enumerate(seqs):
            ids[i, :len(x)] = torch.tensor(x, device=dev)
        # task-4: leave-one-out baseline b_{-i} = mean_{j != i} phi_j, which is unbiased because b_{-i} is
        # independent of rollout i and E[s_i] = 0. The in-sample mean would give E[(phi-b)s] = (1-1/K) g.
        K_ = len(seqs)
        psum = sum(phi)
        w = torch.tensor([(phi[i] - (psum - phi[i]) / max(K_ - 1, 1)) / (K_ * len(ctx))
                          for i in range(K_)], dtype=torch.float32, device=dev)
        S = torch.zeros((K_, per_dim), dtype=torch.float64)
        for i, sq in enumerate(seqs):
            i1 = torch.tensor([sq], dtype=torch.long, device=dev)
            with Capture(lm, layers) as cap:
                model.zero_grad(set_to_none=True)
                out = model(input_ids=i1, use_cache=False)
                lp = F.log_softmax(out.logits[:, :-1].float(), -1).gather(-1, i1[:, 1:].unsqueeze(-1)).squeeze(-1)
                mask = torch.zeros_like(lp)
                mask[:, len(prefix) - 1:] = 1.0
                ((lp * mask).sum()).backward()   # weight 1: store the raw score, weight afterwards
                g = cap.grad_sum()
            for l, o_, d_ in offs:
                S[i, o_:o_ + d_] = (g[l] * (p50[l] / norms[l].clamp(min=1e-12))).double()
        wloo = [float(w[i]) for i in range(K_)]
        for i in range(K_):
            for l, o_, d_ in offs:
                g_re[l] += wloo[i] * S[i, o_:o_ + d_]
        S_all.append(S)
        W_all.append(wloo)
        print(f'  RE ctx {ci} K={K_} w_loo={[round(x,5) for x in wloo]}', flush=True)

        for key, sgn in (('positive', +1.0), ('negative', -1.0)):
            sl = r[key]
            txt = sl if isinstance(sl, str) else (sl[0] if isinstance(sl, (list, tuple)) else str(sl))
            rid, rpre, rsuf = E.candidate_ids(tok, r['prompt'], txt, dev, 384)
            ids1 = rid.to(dev) if torch.is_tensor(rid) else torch.tensor(rid, device=dev)
            ids1 = ids1.view(1, -1).long()
            with Capture(lm, layers) as cap:
                model.zero_grad(set_to_none=True)
                loss1, _ = score_batch(model, ids1, [ids1.shape[1]],
                                       torch.tensor([sgn / len(ctx)], device=dev), int(rpre),
                                       mean_over_suffix=True)
                loss1.backward()
                g = cap.grad_sum()
            for l in layers:
                g_j2[l] += g[l] * (p50[l] / norms[l].clamp(min=1e-12))
        print(f'  J2 ctx {ci} done', flush=True)

    from reliability import split_half_curve, noise_scale, bootstrap_kappa
    rel = {'split_half': split_half_curve(S_all, W_all, ks=(2, 4, 8)),
           'noise_scale': noise_scale(S_all, W_all),
           'interval': bootstrap_kappa(S_all, W_all)}
    rec['reliability'] = rel
    print('REL r(K)=%s kappa=%s interval=[%s, %s]' % (
        {k: round(v['r_mean'], 4) for k, v in rel['split_half'].items()},
        None if rel['noise_scale']['kappa'] is None else round(rel['noise_scale']['kappa'], 1),
        rel['interval']['kappa_lo'], rel['interval']['kappa_hi']), flush=True)

    def flat(d):
        return torch.cat([d[l] for l in layers])

    a_re, a_j2 = flat(g_re), flat(g_j2)
    cos = float(F.cosine_similarity(a_re.unsqueeze(0), a_j2.unsqueeze(0)))
    per_layer = {l: float(F.cosine_similarity(g_re[l].unsqueeze(0), g_j2[l].unsqueeze(0))) for l in layers}
    rec['seed'] = a.seed
    rec['C1'] = {'cos_all': cos, 'cos_per_layer': per_layer,
                 'norm_re': float(a_re.norm()), 'norm_j2': float(a_j2.norm()),
                 'n_ctx': len(ctx), 'K': a.rollouts, 'max_new': a.max_new}
    print(f'C1 cos(g_RE,g_J2)={cos:+.4f} per-layer min/max {min(per_layer.values()):+.4f}/{max(per_layer.values()):+.4f}', flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(a.out, 'w'))
    torch.save({'g_re': g_re, 'g_j2': g_j2, 'p50': p50}, a.out.replace('.json', '.pt'))
    print('wrote', a.out, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
