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
    a = ap.parse_args()
    dev = 'cuda:0'
    torch.manual_seed(a.seed)

    rows = [json.loads(l) for l in open(a.probes)]
    ctx = [r for r in rows if r['trait'] == 'hostility'][:a.n_ctx]
    print(f'prompts: {len(ctx)} hostility', flush=True)
    tok, model, lm = E.load_model(a.model, dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = E.column_norms(lm, layers)
    for q in model.parameters():
        q.requires_grad_(False)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    class Cap:
        def __init__(self):
            self.cap, self.h = {}, []

        def __enter__(self):
            for l in layers:
                def build(idx):
                    def hook(_m, args):
                        x = args[0]
                        if idx in self.cap:
                            return None
                        x = x.detach().clone().requires_grad_(True)
                        self.cap[idx] = x
                        return (x,) + tuple(args[1:])
                    return hook
                self.h.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(build(l)))
            return self

        def grad_sum(self):
            out = {}
            for l, t in self.cap.items():
                g = t.grad
                out[l] = (g.sum(dim=tuple(range(g.dim() - 1))).double().cpu() if g is not None
                          else torch.zeros(t.shape[-1], dtype=torch.float64))
            return out

        def __exit__(self, *x):
            for h in self.h:
                h.remove()

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

    g_path = {l: torch.zeros(lm.layers[l].mlp.down_proj.weight.shape[1], dtype=torch.float64) for l in layers}
    g_teach = {l: torch.zeros_like(g_path[l]) for l in layers}
    rec = {'ctx': [], 'phi': {}, 'texts': {}}
    splits = {}
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
        S = torch.zeros((len(seqs), sum(lm.layers[l].mlp.down_proj.weight.shape[1] for l in layers)),
                        dtype=torch.float32)
        g_pw = {l: torch.zeros_like(g_path[l]) for l in layers}
        for i, sq in enumerate(seqs):
            off = 0
            w = (ph[i] - base) / len(seqs)
            i1 = torch.tensor([sq], dtype=torch.long, device=dev)
            with Cap() as cap:
                model.zero_grad(set_to_none=True)
                out = model(input_ids=i1, use_cache=False)
                lp = F.log_softmax(out.logits[:, :-1].float(), -1).gather(-1, i1[:, 1:].unsqueeze(-1)).squeeze(-1)
                mask = torch.zeros_like(lp)
                mask[:, len(pre) - 1:] = 1.0
                ((lp * mask).sum() * w).backward()
                gg = cap.grad_sum()
            for l in layers:
                v = gg[l] * (p50[l] / norms[l].clamp(min=1e-12))
                g_path[l] += v
                d = lm.layers[l].mlp.down_proj.weight.shape[1]
                S[i, off:off + d] = v.float()
                off += d
        if a.pathwise:
            for i, sq in enumerate(seqs):
                i1 = torch.tensor([sq], dtype=torch.long, device=dev)
                ip, ineg = slot_ids(tok, r['positive']), slot_ids(tok, r['negative'])
                acc = None
                for ids_slot, sgn in ((ip, +1.0), (ineg, -1.0)):
                    full = sq + ids_slot
                    j1 = torch.tensor([full], dtype=torch.long, device=dev)
                    with Cap() as cap:
                        model.zero_grad(set_to_none=True)
                        out = model(input_ids=j1, use_cache=False)
                        lp = F.log_softmax(out.logits[:, :-1].float(), -1)
                        pos = torch.arange(len(sq), len(full), device=dev)
                        tgt = torch.tensor(ids_slot, device=dev)
                        term = lp[0, pos - 1, tgt].mean() * (sgn / len(ids_slot) / len(seqs))
                        term.backward()
                        gg = cap.grad_sum()
                    for l in layers:
                        g_pw[l] += gg[l] * (p50[l] / norms[l].clamp(min=1e-12))
                print(f'    pw rollout {i} done', flush=True) if i % 8 == 0 else None
        splits[ci] = S
        print(f'  [{ci}] {len(seqs)} per-rollout scores stored', flush=True)

        # teacher-forced J2 gradient on the same context (fixed prefix)
        for key, sgn in (('positive', +1.0), ('negative', -1.0)):
            rid, rpre, rsuf = E.candidate_ids(tok, r['prompt'],
                                              r[key] if isinstance(r[key], str) else r[key][0], dev, 384)
            i1 = (rid.to(dev) if torch.is_tensor(rid) else torch.tensor(rid, device=dev)).view(1, -1).long()
            with Cap() as cap:
                model.zero_grad(set_to_none=True)
                out = model(input_ids=i1, use_cache=False)
                lp = F.log_softmax(out.logits[:, :-1].float(), -1).gather(-1, i1[:, 1:].unsqueeze(-1)).squeeze(-1)
                mask = torch.zeros_like(lp)
                mask[:, int(rpre) - 1:] = 1.0
                ((lp * mask).sum() * (sgn / len(ctx))).backward()
                gg = cap.grad_sum()
            for l in layers:
                g_teach[l] += gg[l] * (p50[l] / norms[l].clamp(min=1e-12))
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

    # Reliability curve r(K): split-half cos over K rollouts per context, 200 random disjoint splits.
    import random
    random.seed(0)
    curve = {}
    for K in (2, 4, 8, 16, 32):
        if K > a.rollouts:
            continue
        vals = []
        for ci in range(len(ctx)):
            S = splits[ci]
            if S.shape[0] < K:
                continue
            for _ in range(200):
                idx = list(range(S.shape[0]))
                random.shuffle(idx)
                A = torch.stack([S[i] for i in idx[:K // 2]]).sum(0)
                B = torch.stack([S[i] for i in idx[K // 2:K]]).sum(0)
                if A.norm() > 0 and B.norm() > 0:
                    vals.append(float(F.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0))))
        vals.sort()
        curve[K] = {'r_mean': sum(vals) / len(vals) if vals else None,
                    'r_p10': vals[int(0.1 * len(vals))] if vals else None,
                    'n': len(vals)}
    r32 = curve.get(32, {}).get('r_mean')
    kstar = None
    if r32 is not None and 0 < r32 < 1:
        kstar = 32 * (1 - r32) / r32
    rec['R'] = {'r_of_K': curve, 'K_star_for_0.5': kstar,
                'gate': 'r(32) >= 0.5' if r32 is not None else 'n/a'}
    print(f'R r(K)={ {k: round(v["r_mean"], 3) for k, v in curve.items()} } K*={kstar}', flush=True)

    def flat(d):
        return torch.cat([d[l] for l in layers])

    gp, gt = flat(g_path), flat(g_teach)
    cos = float(F.cosine_similarity(gp.unsqueeze(0), gt.unsqueeze(0)))
    per = {l: float(F.cosine_similarity(g_path[l].unsqueeze(0), g_teach[l].unsqueeze(0))) for l in layers}
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
