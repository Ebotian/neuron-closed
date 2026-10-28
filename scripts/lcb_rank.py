#!/usr/bin/env python3
"""Does ranking by a LOWER CONFIDENCE BOUND beat ranking by mean |g|? (test of RACE's CAM idea)

Why. Record 12 section 6.2 registered this test before running it. RACE (arXiv 2608.24758 v2) ranks
neurons by a posterior mean with a "conservative alignment magnitude" that penalises uncertain
estimates, at roughly 1/100 of gradient cost. Our own reproducibility numbers say the uncertainty
is real: the per-context rank correlation between disjoint prompt halves is only r = 0.4722 median
(pooled it reaches 0.5469), and top-1 by |g| captures only 17.3% of the full-support effect.

Budget and readout are exactly those of plan 2.20, so the two rankings are compared at equal cost
and on the same nested-support protocol:

    budget(u) = sqrt(sum_l ||u_l @ What_l||^2)      (unit-normalised down_proj columns)

Two rankings over the same 18,432 coordinates (layers 13-14):
  A  mean   - descending |mean over contexts of g|          (what we have always used)
  B  lcb    - descending lower confidence bound: sign(mean) * q05(|g|) where q05 is the 5th
              percentile of a context-level bootstrap of |g| (2000 resamples of the 8 contexts).
              A coordinate whose sign flips across contexts gets a *smaller* bound; one that is
              large only on average does too. This is the closest cheap analogue of CAM.

FROZEN reading (registered in record 12 before this run):
  report K at 73% / 89% / 97% of the full-support effect for both rankings, at b = 0.05.
  PREDICTION TO FALSIFY: ranking B reaches 73% with a SMALLER K than ranking A (i.e. K@73%(B) < 256,
  the value ranking A needs per plan 2.20).
  Also report Spearman between the two rankings and the top-64 overlap, so "the ranking changed" and
  "the effect changed" are separable.
  Error discipline (plan 2.18.4): differences below 1e-02 on a single measurement are not claimed;
  capture percentages are ratios of means over 8 contexts, whose error is about 9e-04 in dJ2.

    [i/N] elapsed, eta ...   per measurement (package progress.Progress)
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, random, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # qwentest/


def resolve_pkg(given):
    cands = [Path(given)] if given else []
    cands += [ROOT / '03_unpacked/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
              Path('/root/autodl-tmp/scripts/j2space'),
              Path('/root/autodl-tmp/qwen-gradient-discovery-v6')]
    for c in cands:
        if (c / 'engine.py').is_file() and (c / 'semantic.py').is_file():
            return c
    raise SystemExit('package not found; pass --pkg <dir with engine.py AND semantic.py>')


def load_pools(path, variant='full'):
    lex = json.loads(Path(path).read_text(encoding='utf8'))
    fams = {}
    for fam, d in lex['families'].items():
        key = 'pos' if variant == 'full' else f'pos_{variant}'
        tmpl = d.get('template') or lex.get('templates', {}).get(fam)
        if not tmpl:
            raise ValueError(f'family {fam} has no template')
        fams[fam] = {'pos': d[key], 'neg': d['neg'], 'template': tmpl}
    return lex, fams


def lcb_ranking(G, z_draws=2000, q=0.05, seed=20261028):
    """sign(mean) * q-quantile of the bootstrap distribution of |mean over resampled contexts|.

    Resampling CONTEXTS (not coordinates) is the honest unit: the 8 contexts are the repeated
    measurements of the same coordinate's gradient.
    """
    n_ctx = len(G)
    absg = [[abs(v) for v in row] for row in G]
    dim = len(absg[0])
    rng = random.Random(seed)
    draws = []
    for _ in range(z_draws):
        pick = [rng.randrange(n_ctx) for _ in range(n_ctx)]
        draws.append([sum(absg[i][j] for i in pick) / n_ctx for j in range(dim)])
    lcb = []
    for j in range(dim):
        col = sorted(d[j] for d in draws)
        lcb.append(col[int(q * (z_draws - 1))])
    mean = [sum(absg[i][j] for i in range(n_ctx)) / n_ctx for j in range(dim)]
    sign = []
    for j in range(dim):
        s = sum(G[i][j] for i in range(n_ctx))
        sign.append(1.0 if s >= 0 else -1.0)
    # Return the MAGNITUDE bound and the sign separately. CAM is a conservative magnitude, not a
    # signed value: ranking a signed quantity would push every negative coordinate to the bottom
    # and make the comparison against |mean g| meaningless (first attempt: Spearman 0.0016 while
    # the top-64 overlap was 0.5, which is self-contradictory and exposed exactly this).
    return lcb, sign, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--layers', default='13-14')
    ap.add_argument('--ks', default='1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,18432')
    ap.add_argument('--budget', type=float, default=0.05)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--contexts', type=int, default=8,
                    help='keep this equal to plan 2.20 (8 contexts) so K@73 pct is comparable with its 256')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--out', default='out/lcb_rank')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
    ks = [int(x) for x in a.ks.split(',')]
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    ctx = [r for r in rows if r.get('split') == a.split][:a.contexts]
    if not ctx:
        raise SystemExit(f'no {a.split} contexts in {a.probes}')

    print('Question : does ranking by a LOWER CONFIDENCE BOUND beat ranking by mean |g|?')
    print(f'Rankings : A = mean |g| (current practice) | B = 5% bootstrap LCB of |g| x sign(mean)')
    print(f'Readout  : J2 = mean over families {sorted(fams)}, pool variant {a.pool_variant}')
    print(f'Protocol : nested top-K by each ranking, budget {a.budget} (plan 2.20 protocol)')
    print(f'Contexts : {len(ctx)} -> {[c["id"] for c in ctx]}   bootstrap draws {a.boot}')
    n_eval = 2 * len(ks) * len(ctx)
    print(f'Work     : {len(ctx)}x{len(fams)} grad passes + 2 rankings x {len(ks)} K x {len(ctx)} '
          f'contexts = {n_eval} J2 evaluations = {n_eval * len(fams)} forwards')
    print('Frozen   : report K at 73% / 89% / 97% of full-support effect for both rankings.')
    print('PREDICTION TO FALSIFY (registered before the run): K@73%(B) < 256 = K@73%(A).')
    if a.dry_run:
        print(f'[dry-run] would write {a.out}/lcb_rank.json and .md (+ the raw g matrix)')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import apply_steering, column_norms, estimate_p50, model_lm, score_for_dose
    from progress import Progress
    from semantic import batch_mean_logp, build_batch, language_log_mass, sem_gradient

    dev = 'cuda:0'
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to(dev).eval()
    model.requires_grad_(False)
    lm = model_lm(model)
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device=dev)
    print('p50:', {l: round(p50[l], 6) for l in layers})
    What = {}
    for l in layers:
        Wt = lm.layers[l].mlp.down_proj.weight.detach().float().T
        What[l] = Wt / Wt.norm(dim=1, keepdim=True).clamp(min=1e-12)
    npl = What[layers[0]].shape[0]
    dim = len(layers) * npl

    prog = Progress(len(ctx) * len(fams) + 2 * len(ks) * len(ctx))
    G, batches, base = [], {}, {}
    for c in ctx:
        gsum = None
        for fam, f in sorted(fams.items()):
            _j2, gg = sem_gradient(tok, model, lm, layers, c['prompt'], f['template'], f, dev, a.max_seq)
            sc = {l: score_for_dose(gg[l].float(), p50[l], norms[l].to(gg[l].device)) for l in layers}
            gsum = sc if gsum is None else {l: gsum[l] + sc[l] for l in layers}
            prog.step(f'{c["id"]} g {fam}')
        G.append(torch.cat([(gsum[l] / len(fams)).double().cpu() for l in layers]).tolist())
        batches[c['id']] = {fam: build_batch(tok, c['prompt'], f['template'], f['pos'], f['neg'],
                                            dev, a.max_seq) for fam, f in sorted(fams.items())}
        with torch.inference_mode():
            vals = []
            for fam, (ids, pre, suf, kinds) in batches[c['id']].items():
                S = batch_mean_logp(model, ids, pre, suf)
                jp, _ = language_log_mass(S[kinds > 0].detach().float())
                jn, _ = language_log_mass(S[kinds < 0].detach().float())
                vals.append(jp - jn)
        base[c['id']] = sum(vals) / len(vals)
        prog.step(f'{c["id"]} base J2={base[c["id"]]:+.6f}')

    print(f'computing the two rankings over {dim} coordinates (bootstrap {a.boot}) ...', flush=True)
    lcb_mag, lcb_sign, mean_abs = lcb_ranking(G, z_draws=a.boot)
    gmean = [sum(G[i][j] for i in range(len(G))) / len(G) for j in range(dim)]
    rankA = sorted(range(dim), key=lambda j: -abs(gmean[j]))
    rankB = sorted(range(dim), key=lambda j: -lcb_mag[j])

    def spearman(x, y):
        rx = [[j, x[j]] for j in range(len(x))]
        rx.sort(key=lambda t: -t[1])
        rankx = {t[0]: i for i, t in enumerate(rx)}
        ry = [[j, y[j]] for j in range(len(y))]
        ry.sort(key=lambda t: -t[1])
        ranky = {t[0]: i for i, t in enumerate(ry)}
        n = len(x)
        d2 = sum((rankx[j] - ranky[j]) ** 2 for j in range(n))
        return 1 - 6 * d2 / (n * (n * n - 1))

    sp = spearman(lcb_mag, [abs(v) for v in gmean])
    top_overlap = len(set(rankA[:64]) & set(rankB[:64])) / 64
    print(f'  Spearman(lcb, |mean g|) = {sp:.4f}   top-64 overlap = {top_overlap:.3f}')

    def j2_of(cid, iv):
        with apply_steering(lm, iv, p50, norms), torch.inference_mode():
            vals = []
            for fam, (ids, pre, suf, kinds) in batches[cid].items():
                S = batch_mean_logp(model, ids, pre, suf)
                jp, _ = language_log_mass(S[kinds > 0].detach().float())
                jn, _ = language_log_mass(S[kinds < 0].detach().float())
                vals.append(jp - jn)
        return sum(vals) / len(vals)

    def curve(order):
        out = {}
        for K in ks:
            per_ctx = []
            for c in ctx:
                cid = c['id']
                idx = order[:K]
                u_by_layer = {}
                off = 0
                for l in layers:
                    sel = [j - off for j in idx if off <= j < off + npl]
                    u = torch.zeros(npl, dtype=torch.float64)
                    for j in sel:
                        u[j] = gmean[off + j]
                    u_by_layer[l] = u.to(dev)
                    off += npl
                bu = math.sqrt(sum(float((u.float() @ What[l]).pow(2).sum())
                                   for l, u in u_by_layer.items()))
                scale = a.budget / bu
                iv = []
                off = 0
                for l in layers:
                    for j in torch.nonzero(u_by_layer[l].abs() > 1e-15).flatten().tolist():
                        iv.append({'layer': l, 'neuron': int(j),
                                   'dose': float(u_by_layer[l][j]) * scale})
                    off += npl
                per_ctx.append(j2_of(cid, iv) - base[cid])
            out[str(K)] = {'mean_dJ2': statistics.fmean(per_ctx), 'per_context': per_ctx,
                           'n_nonzero': len(iv)}
            prog.step(f'K={K} mean dJ2={out[str(K)]["mean_dJ2"]:+.5f}')
        return out

    curA = curve(rankA)
    curB = curve(rankB)
    full = curA[str(max(ks))]['mean_dJ2']
    rec = {'schema': 'qwentest-lcb-rank-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'layers': layers,
           'budget': a.budget, 'ks': ks, 'contexts': [c['id'] for c in ctx], 'bootstrap': a.boot,
           'spearman_lcb_vs_mean': sp, 'top64_overlap': top_overlap,
           'lcb_is_magnitude_with_sign_separate': True,
           'baseline_J2': base, 'curve_mean': curA, 'curve_lcb': curB,
           'full_support_dJ2': full,
           'prediction_registered': 'K@73%(lcb) < 256 = K@73%(mean)',
           'ranking_order_top32_mean': rankA[:32], 'ranking_order_top32_lcb': rankB[:32]}

    def k_at(frac):
        target = frac * full
        for K in ks:
            if curA[str(K)]['mean_dJ2'] >= target:
                return K
        return None
    rec['K_at'] = {name: {str(int(f * 100)): k_at(f) for f in (0.73, 0.89, 0.97)}
                   for name in ('mean',)}
    rec['K_at']['lcb'] = {}
    for f in (0.73, 0.89, 0.97):
        target = f * full
        rec['K_at']['lcb'][str(int(f * 100))] = next(
            (K for K in ks if curB[str(K)]['mean_dJ2'] >= target), None)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'lcb_rank.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf8')
    lines = ['# Lower-confidence-bound ranking vs mean |g| ranking', '',
             f'- machine `{rec["machine"]}` / {rec["gpu"]} · layers {layers} · budget {a.budget} · '
             f'{len(ctx)} contexts · bootstrap {a.boot}',
             f'- Spearman(LCB, |mean g|) = **{sp:.4f}**, top-64 overlap **{top_overlap:.3f}** '
             f'(so the two rankings really differ)',
             f'- full-support dJ2 (mean ranking) = {full:+.5f}', '',
             '| K | mean-rank dJ2 | lcb-rank dJ2 | mean-rank capture | lcb-rank capture |',
             '|---:|---:|---:|---:|---:|']
    for K in ks:
        a_ = curA[str(K)]['mean_dJ2']
        b_ = curB[str(K)]['mean_dJ2']
        lines.append(f'| {K} | {a_:+.5f} | {b_:+.5f} | {a_ / full:.1%} | {b_ / full:.1%} |')
    lines += ['', '## K needed to reach a share of the full-support effect (same denominator)', '',
              '| share | mean ranking | LCB ranking |', '|---|---:|---:|']
    for f in ('73', '89', '97'):
        lines.append(f'| {f}% | {rec["K_at"]["mean"][f]} | {rec["K_at"]["lcb"][f]} |')
    lines += ['', f'**registered prediction**: {rec["prediction_registered"]} -> '
              f'{"HELD" if (rec["K_at"]["lcb"]["73"] or 10**9) < 256 else "FALSIFIED"}']
    (out / 'lcb_rank.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/lcb_rank.json and .md')


if __name__ == '__main__':
    main()
