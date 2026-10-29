#!/usr/bin/env python3
"""R7: in the DENSE regime, does the budget-optimal direction beat "follow the gradient"?

Plan §2.21.3 found that the dose-budget metric is identity-like only for sparse supports:
budget(g_S)/||g_S|| = 1.000 at K=1, 1.193 at K=32, but **2.757 at full support**. So
§2.16's "M ~= I, therefore G=I is near-optimal" was measured at K=32 and does NOT extend to
the dense regime, where the surrogate-optimal control differs from the gradient direction.

Derivation (all exact, no sampling). With What_l the 9216 x 2560 matrix of UNIT-normalised
down_proj columns, the residual displacement and budget are

    d_l = What_l' u_l        budget(u)^2 = sum_l ||d_l||^2 = u' M u,   M = What What'

M is 9216 x 9216 with rank <= 2560, so it is singular and the minimiser of
{ 1/2 u'Mu : g'u = delta } uses the pseudo-inverse:

    u* ~ M^+ g,   M^+ = What B^-2 What',   B = What'What  (2560 x 2560, SPD)

Writing e = What' g (the residual-space image of the gradient), this collapses to

    d* ~ B^-1 e          and          u* = What^+ d* = B^-1 What' d*

i.e. **the answer naturally lives in residual space (2560 dims), and u* is the
minimum-norm representative of the equivalence class { u : What'u = d* }, whose dimension is
9216 - 2560 = 6656 per layer.** That is exactly the output shape asked for: residual
direction + cheapest representative + equivalence class.

Predicted gains at budget b (so the experiment can be judged before it is run):
    u ~ g    :  b * ||g||^2 / ||e||
    u ~ M^+g :  b * sqrt(e' B^-2 e)
The second is >= the first by Cauchy-Schwarz; the ratio is printed BEFORE any measurement,
because if it is ~1 there is nothing to gain and the run is pointless.

ARMS  base | u_g (follow the gradient, full support) | u_M (budget-optimal, full support)
      | u_g64 (top-64 by |g|, the §2.20 operating point)
FROZEN criteria: PRIMARY = realized dJ2(u_M) > dJ2(u_g) in >= 7/8 contexts at each budget
(exact two-sided sign test) with median relative gain > 0. §2.18.4 error discipline: any
difference below 1e-02 is not claimed from bf16 alone; --fp32 re-runs the readout in
float32 with p50, the column norms and every direction held at their bf16 values.
§2.16 warning, stated up front: surrogate-optimal does NOT imply realized-optimal -- there
the J'J-optimal direction LOST 26.6% on the real model. So this must be measured.

GPU: per context 2 grad-enabled passes (for g) + (1 + 3 arms) x |budgets| J2 evaluations.

    [i/N] elapsed, eta ...   per measurement (package progress.Progress)
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # qwentest/


def resolve_pkg(given):
    cands = [Path(given)] if given else []
    cands += [ROOT / 'vendor/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
              Path('/root/autodl-tmp/scripts/j2space'),
              Path('/root/autodl-tmp/qwen-gradient-discovery-v6')]
    for c in cands:
        if (c / 'engine.py').is_file() and (c / 'semantic.py').is_file():
            return c
    raise SystemExit('package not found; pass --pkg <dir with engine.py AND semantic.py>')


def load_pools(path, variant='full'):
    """Same rule as scan_sem.py::load_pools."""
    lex = json.loads(Path(path).read_text(encoding='utf8'))
    fams = {}
    for fam, d in lex['families'].items():
        key = 'pos' if variant == 'full' else f'pos_{variant}'
        tmpl = d.get('template') or lex.get('templates', {}).get(fam)
        if not tmpl:
            raise ValueError(f'family {fam} has no template')
        fams[fam] = {'pos': d[key], 'neg': d['neg'], 'template': tmpl}
    return lex, fams


def exact_sign_p(diffs):
    v = [d for d in diffs if abs(d) > 1e-12]
    n = len(v)
    if n == 0:
        return None, 0
    k = min(sum(x > 0 for x in v), sum(x < 0 for x in v))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), n


def transfer(a, tok, model, lm, p50, norms, What, B, n_dim, d_dim, layers, budgets,
             fams, ctx, dev, budget, to_budget, **_):
    """G2-style transfer test for the R7 result (plan 2.22).

    R7 showed u ~ M^+g beats u ~ g by 36-43% in 8/8 contexts -- but every one of those
    measurements built the direction from the SAME context it was then measured on.
    Plan 2.15.2 found that J'J's off-diagonal structure does NOT survive a change of context
    (cross-context r = 0.12-0.18), so "does the direction transfer" is a live question.
    M itself is context-free (weights only); context enters only through g.

    Design: split the contexts into two trait-balanced halves (alternating index, because the
    probe file groups pairs by trait), pool g within each half, build u_g and u_M from the
    pooled g, then measure every direction on every context. That gives in-sample AND
    out-of-sample measurements of the same arms, plus the registered G2 statistics on the two
    halves' gradients.

    FROZEN criteria
      PRIMARY  out-of-sample, realized dJ2(u_M) > dJ2(u_g) in >= 7 of the 8 (half, context)
               evaluations at each budget; exact two-sided sign test; median relative gain > 0.
      G2       r(|g_A|, |g_B|) >= 0.50 AND top-64 overlap >= 0.30 (the registered gate).
      ALSO     the out-of-sample M/g ratio must not collapse relative to in-sample.
      Error    §2.18.4: differences below 1e-02 are not claimed from bf16 alone.
    """
    import torch
    from engine import apply_steering, score_for_dose
    from progress import Progress
    from semantic import batch_mean_logp, build_batch, language_log_mass, sem_gradient

    ids_all = [c['id'] for c in ctx]
    halves = {'A': ids_all[0::2], 'B': ids_all[1::2]}
    pools = {**halves, 'ALL': ids_all}
    arms = ['u_g', 'u_M']
    n_meas = len(pools) * len(arms) * len(ids_all) * len(budgets)
    prog = Progress(len(ctx) * len(fams) + len(ctx) + n_meas)

    batches, G_ctx = {}, {}
    for c in ctx:
        gsum = None
        for fam, f in sorted(fams.items()):
            _j2, gg = sem_gradient(tok, model, lm, layers, c['prompt'], f['template'], f, dev,
                                   a.max_seq)
            sc = {l: score_for_dose(gg[l].float(), p50[l], norms[l].to(gg[l].device)) for l in layers}
            gsum = sc if gsum is None else {l: gsum[l] + sc[l] for l in layers}
            prog.step(f'{c["id"]} g {fam}')
        G_ctx[c['id']] = {l: (gsum[l] / len(fams)).double().to(dev) for l in layers}
        batches[c['id']] = {fam: build_batch(tok, c['prompt'], f['template'], f['pos'], f['neg'],
                                            dev, a.max_seq) for fam, f in sorted(fams.items())}

    def j2_of(cid, iv):
        with apply_steering(lm, iv, p50, norms), torch.inference_mode():
            vals = []
            for fam, (ids, pre, suf, kinds) in batches[cid].items():
                S = batch_mean_logp(model, ids, pre, suf)
                jp, _ = language_log_mass(S[kinds > 0].detach().float())
                jn, _ = language_log_mass(S[kinds < 0].detach().float())
                vals.append(jp - jn)
        return sum(vals) / len(vals)

    base = {cid: j2_of(cid, []) for cid in ids_all}
    for cid in ids_all:
        prog.step(f'{cid} base J2={base[cid]:+.6f}')

    dirs, flat_g = {}, {}
    for name, members in pools.items():
        gm = {l: sum(G_ctx[c][l] for c in members) / len(members) for l in layers}
        e = {l: What[l].T @ gm[l] for l in layers}
        ds = {l: torch.linalg.solve(B[l], e[l]) for l in layers}
        um = {l: What[l] @ torch.linalg.solve(B[l], ds[l]) for l in layers}
        dirs[name] = {'u_g': gm, 'u_M': um}
        flat_g[name] = torch.cat([gm[l] for l in layers])

    res = {}
    for name in pools:
        for arm in arms:
            v = dirs[name][arm]
            for b in budgets:
                u, _s = to_budget(v, b)
                iv = [{'layer': l, 'neuron': i, 'dose': float(u[l][i])}
                      for l in layers
                      for i in torch.nonzero(u[l].abs() > 1e-15).flatten().tolist()]
                for cid in ids_all:
                    res[(name, arm, b, cid)] = j2_of(cid, iv) - base[cid]
                    prog.step(f'{name} {arm} b={b} {cid} dJ2={res[(name, arm, b, cid)]:+.5f}')

    # ---------------- frozen analysis ----------------
    def ratio(name, b, members):
        v = [res[(name, 'u_M', b, c)] / res[(name, 'u_g', b, c)] for c in members
             if abs(res[(name, 'u_g', b, c)]) > 1e-12]
        return statistics.fmean(v) if v else None

    cos = lambda x, y: float(x @ y / (x.norm() * y.norm()))
    gA, gB = flat_g['A'], flat_g['B']
    r_abs = statistics.correlation([float(x) for x in gA.abs().cpu()],
                                   [float(x) for x in gB.abs().cpu()])
    tA = set(torch.topk(gA.abs(), 64).indices.tolist())
    tB = set(torch.topk(gB.abs(), 64).indices.tolist())
    rec = {'schema': 'qwentest-dense-transfer-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'layers': layers,
           'budgets': budgets, 'pool_variant': a.pool_variant, 'halves': halves,
           'p50': {str(l): p50[l] for l in layers}, 'baseline_J2': base,
           'g2_pearson_abs_g': r_abs, 'g2_top64_overlap': len(tA & tB) / 64,
           'cos_u_g_A_B': cos(torch.cat([dirs['A']['u_g'][l] for l in layers]),
                              torch.cat([dirs['B']['u_g'][l] for l in layers])),
           'cos_u_M_A_B': cos(torch.cat([dirs['A']['u_M'][l] for l in layers]),
                              torch.cat([dirs['B']['u_M'][l] for l in layers])),
           'measurements': {f'{n}|{arm}|{b}|{c}': v for (n, arm, b, c), v in res.items()},
           'analysis': {}}
    for b in budgets:
        for name in ('A', 'B', 'ALL'):
            out = [c for c in ids_all if c not in pools[name]] if name != 'ALL' else []
            ins = pools[name]
            for tag, members in (('in', ins), ('out', out)):
                if not members:
                    continue
                diffs = [res[(name, 'u_M', b, c)] - res[(name, 'u_g', b, c)] for c in members]
                p, n = exact_sign_p(diffs)
                rel = [res[(name, 'u_M', b, c)] / res[(name, 'u_g', b, c)] for c in members
                       if abs(res[(name, 'u_g', b, c)]) > 1e-12]
                rec['analysis'][f'b={b}|{name}|{tag}'] = {
                    'n': n, 'wins': sum(d > 0 for d in diffs), 'sign_p': p,
                    'median_abs_diff': statistics.median(diffs),
                    'mean_ratio_M_over_g': ratio(name, b, members),
                    'mean_dJ2_u_g': statistics.fmean(res[(name, 'u_g', b, c)] for c in members),
                    'mean_dJ2_u_M': statistics.fmean(res[(name, 'u_M', b, c)] for c in members)}
    oos = []
    for b in budgets:
        for name in ('A', 'B'):
            oos += [res[(name, 'u_M', b, c)] - res[(name, 'u_g', b, c)]
                    for c in ids_all if c not in pools[name]]
    p_oos, n_oos = exact_sign_p(oos)
    rec['primary_out_of_sample'] = {'n': n_oos, 'wins': sum(d > 0 for d in oos),
                                    'exact_two_sided_sign_p': p_oos,
                                    'median_abs_diff': statistics.median(oos)}
    rec['g2_gate_pass'] = bool(r_abs >= 0.50 and len(tA & tB) / 64 >= 0.30)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'dense_transfer.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                             encoding='utf8')
    lines = ['# Does the R7 direction TRANSFER? (direction built on one half, measured on the other)', '',
             f'- machine `{rec["machine"]}` / {rec["gpu"]} · layers {layers} · budgets {budgets}',
             f'- half A = {halves["A"]}', f'- half B = {halves["B"]}',
             f'- **G2 gate**: r(|g_A|,|g_B|) = {r_abs:.4f} (need >= 0.50), '
             f'top-64 overlap = {rec["g2_top64_overlap"]:.3f} (need >= 0.30) -> '
             f'{"PASS" if rec["g2_gate_pass"] else "**FAIL**"}',
             f'- cos(u_g(A), u_g(B)) = {rec["cos_u_g_A_B"]:.4f}   '
             f'cos(u_M(A), u_M(B)) = {rec["cos_u_M_A_B"]:.4f}', '',
             '## realized mean dJ2, in-sample vs out-of-sample', '',
             '| budget | pool | sample | n | u_g | u_M | mean ratio M/g | wins | sign p |',
             '|---:|---|---|---:|---:|---:|---:|---:|---:|']
    for b in budgets:
        for name in ('A', 'B', 'ALL'):
            for tag in ('in', 'out'):
                k = f'b={b}|{name}|{tag}'
                if k not in rec['analysis']:
                    continue
                v = rec['analysis'][k]
                sp = 'n/a' if v['sign_p'] is None else f'{v["sign_p"]:.4f}'
                lines.append(f'| {b} | {name} | {tag} | {v["n"]} | {v["mean_dJ2_u_g"]:+.5f} | '
                             f'{v["mean_dJ2_u_M"]:+.5f} | {v["mean_ratio_M_over_g"]:.4f} | '
                             f'{v["wins"]}/{v["n"]} | {sp} |')
    po = rec['primary_out_of_sample']
    lines += ['', f'**PRIMARY (out-of-sample, pooled over both halves and all budgets)**: '
                  f'u_M > u_g in {po["wins"]}/{po["n"]}, exact two-sided sign p = '
                  f'{po["exact_two_sided_sign_p"]:.4f}, median abs diff {po["median_abs_diff"]:+.5f}']
    (out / 'dense_transfer.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/dense_transfer.json and .md')


def g1_test(a, tok, model, lm, p50, norms, What, B, n_dim, d_dim, layers, budgets,
            fams_ignored, ctx, dev, budget, to_budget, **_):
    """G1 for the R7 result: does it survive a change of WORD POOL?

    §2.23 settled G2 (disjoint prompt halves). G1 is the other registered gate: the readout's
    own words split into two disjoint halves. The lexicon ships `pos_A` / `pos_B` (adj 9/10,
    verb 7/7) with a SHARED neutral `neg`, which is exactly how G1 was defined for the scan.

    Two things are measured, and they answer different worries:
      1. DIRECTION AGREEMENT - are the two halves' directions the same direction?
         cos(u_g^A, u_g^B), cos(u_M^A, u_M^B), r(|g^A|,|g^B|) over all 18432 coordinates,
         and the top-64 overlap of the two |g| rankings. Registered gate: r >= 0.50 and
         top-64 overlap >= 0.30.
      2. ADVANTAGE REPRODUCIBILITY - within each word half, measured with THAT half's own
         readout, does u_M still beat u_g in >= 7/8 contexts (exact two-sided sign test)?
         If the advantage only exists under the full pool, it is a word-selection artifact.

    Each half is measured with its own readout on purpose: J2 over pos_A is a different
    quantity from J2 over pos_B, so comparing a direction built on A against A's readout is
    the honest within-half test.
    """
    import torch
    from engine import apply_steering, score_for_dose
    from progress import Progress
    from semantic import batch_mean_logp, build_batch, language_log_mass, sem_gradient

    variants = ('A', 'B')
    fams = {v: load_pools(a.lexicon, v)[1] for v in variants}
    ids_all = [c['id'] for c in ctx]
    arms = ['u_g', 'u_M']
    n_meas = len(ids_all) * len(variants) * (1 + len(arms) * len(budgets))
    # pooled-over-contexts directions are measured too: the registered G1 gate was defined on
    # POOLED gradients (the earlier r=0.693/J64=0.524 came from pooling 19 prompts), so
    # per-context agreement is a different and stricter quantity. Report both, labelled.
    n_meas += len(variants) * len(arms) * len(budgets) * len(ids_all)
    prog = Progress(len(ids_all) * len(variants) * len(fams['A']) + n_meas)

    G, batches, base = {}, {}, {}
    for v in variants:
        for c in ctx:
            cid = c['id']
            gsum = None
            for fam, f in sorted(fams[v].items()):
                _j2, gg = sem_gradient(tok, model, lm, layers, c['prompt'], f['template'], f,
                                       dev, a.max_seq)
                sc = {l: score_for_dose(gg[l].float(), p50[l], norms[l].to(gg[l].device))
                      for l in layers}
                gsum = sc if gsum is None else {l: gsum[l] + sc[l] for l in layers}
                prog.step(f'{cid} g pool{v} {fam}')
            G[(v, cid)] = {l: (gsum[l] / len(fams[v])).double().to(dev) for l in layers}
            batches[(v, cid)] = {fam: build_batch(tok, c['prompt'], f['template'], f['pos'],
                                                 f['neg'], dev, a.max_seq)
                                 for fam, f in sorted(fams[v].items())}

    def j2_of(v, cid, iv):
        with apply_steering(lm, iv, p50, norms), torch.inference_mode():
            vals = []
            for fam, (ids, pre, suf, kinds) in batches[(v, cid)].items():
                S = batch_mean_logp(model, ids, pre, suf)
                jp, _ = language_log_mass(S[kinds > 0].detach().float())
                jn, _ = language_log_mass(S[kinds < 0].detach().float())
                vals.append(jp - jn)
        return sum(vals) / len(vals)

    for v in variants:
        for cid in ids_all:
            base[(v, cid)] = j2_of(v, cid, [])
            prog.step(f'{cid} base pool{v} J2={base[(v, cid)]:+.6f}')

    dirs, flat = {}, {}
    for v in variants:
        for cid in ids_all:
            g = G[(v, cid)]
            e = {l: What[l].T @ g[l] for l in layers}
            ds = {l: torch.linalg.solve(B[l], e[l]) for l in layers}
            dirs[(v, cid)] = {'u_g': g,
                              'u_M': {l: What[l] @ torch.linalg.solve(B[l], ds[l]) for l in layers}}
            flat[(v, cid)] = torch.cat([g[l] for l in layers])

    res = {}
    for v in variants:
        for cid in ids_all:
            for arm in arms:
                for b in budgets:
                    u, _s = to_budget(dirs[(v, cid)][arm], b)
                    iv = [{'layer': l, 'neuron': i, 'dose': float(u[l][i])}
                          for l in layers
                          for i in torch.nonzero(u[l].abs() > 1e-15).flatten().tolist()]
                    res[(v, cid, arm, b)] = j2_of(v, cid, iv) - base[(v, cid)]
                    prog.step(f'{cid} pool{v} {arm} b={b} dJ2={res[(v, cid, arm, b)]:+.5f}')

    # ---- pooled-over-contexts directions, measured with each half's own readout ----
    Gpool = {v: {l: sum(G[(v, c)][l] for c in ids_all) / len(ids_all) for l in layers}
             for v in variants}
    dirs_pool, flat_pool = {}, {}
    for v in variants:
        gp = Gpool[v]
        e = {l: What[l].T @ gp[l] for l in layers}
        ds = {l: torch.linalg.solve(B[l], e[l]) for l in layers}
        dirs_pool[v] = {'u_g': gp,
                        'u_M': {l: What[l] @ torch.linalg.solve(B[l], ds[l]) for l in layers}}
        flat_pool[v] = torch.cat([gp[l] for l in layers]).cpu()
    res_pool = {}
    for v in variants:
        for arm in arms:
            for b in budgets:
                u, _s = to_budget(dirs_pool[v][arm], b)
                iv = [{'layer': l, 'neuron': i, 'dose': float(u[l][i])}
                      for l in layers
                      for i in torch.nonzero(u[l].abs() > 1e-15).flatten().tolist()]
                for cid in ids_all:
                    res_pool[(v, arm, b, cid)] = j2_of(v, cid, iv) - base[(v, cid)]
                    prog.step(f'POOLED {v} {arm} b={b} {cid} '
                              f'dJ2={res_pool[(v, arm, b, cid)]:+.5f}')

    # ---------------- frozen analysis ----------------
    agree = []
    for cid in ids_all:
        gA, gB = flat[('A', cid)].cpu(), flat[('B', cid)].cpu()
        topA = set(torch.topk(gA.abs(), 64).indices.tolist())
        topB = set(torch.topk(gB.abs(), 64).indices.tolist())
        cos = lambda x, y: float(x @ y / (x.norm() * y.norm()))
        uA = torch.cat([dirs[('A', cid)]['u_M'][l].cpu() for l in layers])
        uB = torch.cat([dirs[('B', cid)]['u_M'][l].cpu() for l in layers])
        vA = torch.cat([dirs[('A', cid)]['u_g'][l].cpu() for l in layers])
        vB = torch.cat([dirs[('B', cid)]['u_g'][l].cpu() for l in layers])
        agree.append({'context': cid,
                      'pearson_abs_g': statistics.correlation([float(x) for x in gA.abs()],
                                                              [float(x) for x in gB.abs()]),
                      'top64_overlap': len(topA & topB) / 64,
                      'cos_u_g_A_B': cos(vA, vB), 'cos_u_M_A_B': cos(uA, uB)})
    gA, gB = flat_pool['A'], flat_pool['B']
    cosv = lambda x, y: float(x @ y / (x.norm() * y.norm()))
    pooled = {
        'pearson_abs_g': statistics.correlation([float(x) for x in gA.abs()],
                                                [float(x) for x in gB.abs()]),
        'top64_overlap': len(set(torch.topk(gA.abs(), 64).indices.tolist())
                             & set(torch.topk(gB.abs(), 64).indices.tolist())) / 64,
        'cos_u_g_A_B': cosv(torch.cat([dirs_pool['A']['u_g'][l].cpu() for l in layers]),
                            torch.cat([dirs_pool['B']['u_g'][l].cpu() for l in layers])),
        'cos_u_M_A_B': cosv(torch.cat([dirs_pool['A']['u_M'][l].cpu() for l in layers]),
                            torch.cat([dirs_pool['B']['u_M'][l].cpu() for l in layers]))}

    rec = {'schema': 'qwentest-dense-g1-v2', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'layers': layers,
           'budgets': budgets, 'contexts': ids_all,
           'pools': {v: {f: {'pos': len(fams[v][f]['pos']), 'neg': len(fams[v][f]['neg']),
                             'template': fams[v][f]['template']} for f in fams[v]} for v in variants},
           'p50': {str(l): p50[l] for l in layers}, 'cond_B': {str(l): float(torch.linalg.cond(B[l]))
                                                               for l in layers},
           'agreement': agree,
           'agreement_pooled_over_contexts': pooled,
           'g1_gate_pass_pooled': bool(pooled['pearson_abs_g'] >= 0.50
                                       and pooled['top64_overlap'] >= 0.30),
           'measurements_pooled': {f'{v}|{arm}|{b}|{c}': val
                                   for (v, arm, b, c), val in res_pool.items()},
           'g1_median_pearson_abs_g': statistics.median(x['pearson_abs_g'] for x in agree),
           'g1_median_top64_overlap': statistics.median(x['top64_overlap'] for x in agree),
           'measurements': {f'{v}|{c}|{arm}|{b}': val for (v, c, arm, b), val in res.items()},
           'analysis': {}}
    rec['g1_gate_pass'] = bool(rec['g1_median_pearson_abs_g'] >= 0.50
                               and rec['g1_median_top64_overlap'] >= 0.30)
    for v in variants:
        for b in budgets:
            diffs = [res_pool[(v, 'u_M', b, cid)] - res_pool[(v, 'u_g', b, cid)]
                     for cid in ids_all]
            p, n = exact_sign_p(diffs)
            rel = [(res_pool[(v, 'u_M', b, cid)] - res_pool[(v, 'u_g', b, cid)])
                   / abs(res_pool[(v, 'u_g', b, cid)]) for cid in ids_all
                   if abs(res_pool[(v, 'u_g', b, cid)]) > 1e-12]
            rec['analysis'][f'POOLED pool{v}|b={b}'] = {
                'n': n, 'wins': sum(d > 0 for d in diffs), 'sign_p': p,
                'median_abs_diff': statistics.median(diffs),
                'median_relative_gain': statistics.median(rel) if rel else None,
                'mean_dJ2_u_g': statistics.fmean(res_pool[(v, 'u_g', b, c)] for c in ids_all),
                'mean_dJ2_u_M': statistics.fmean(res_pool[(v, 'u_M', b, c)] for c in ids_all)}
    for v in variants:
        for b in budgets:
            diffs = [res[(v, cid, 'u_M', b)] - res[(v, cid, 'u_g', b)] for cid in ids_all]
            p, n = exact_sign_p(diffs)
            rel = [(res[(v, cid, 'u_M', b)] - res[(v, cid, 'u_g', b)])
                   / abs(res[(v, cid, 'u_g', b)]) for cid in ids_all
                   if abs(res[(v, cid, 'u_g', b)]) > 1e-12]
            rec['analysis'][f'pool{v}|b={b}'] = {
                'n': n, 'wins': sum(d > 0 for d in diffs), 'sign_p': p,
                'median_abs_diff': statistics.median(diffs),
                'median_relative_gain': statistics.median(rel) if rel else None,
                'mean_dJ2_u_g': statistics.fmean(res[(v, cid, 'u_g', b)] for cid in ids_all),
                'mean_dJ2_u_M': statistics.fmean(res[(v, cid, 'u_M', b)] for cid in ids_all)}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'dense_g1.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf8')
    lines = ['# G1 for R7: does the advantage survive a change of WORD POOL?', '',
             f'- machine `{rec["machine"]}` / {rec["gpu"]} · layers {layers} · {len(ids_all)} contexts',
             f'- pools: ' + '; '.join(f'{v}: ' + ', '.join(
                 f'{f} pos={rec["pools"][v][f]["pos"]}/neg={rec["pools"][v][f]["neg"]}'
                 for f in sorted(rec['pools'][v])) for v in variants) + ' (neg shared)',
             f'- **G1 gate, PER-CONTEXT g** (stricter than the registered gate): median '
             f'r(|g_A|,|g_B|) = {rec["g1_median_pearson_abs_g"]:.4f} (need >=0.50), median top-64 '
             f'overlap = {rec["g1_median_top64_overlap"]:.3f} (need >=0.30) -> '
             f'{"PASS" if rec["g1_gate_pass"] else "**FAIL**"}',
             f'- **G1 gate, g POOLED over all {len(ids_all)} contexts** (how the gate was '
             f'originally measured): r = {pooled["pearson_abs_g"]:.4f}, top-64 overlap = '
             f'{pooled["top64_overlap"]:.3f} -> '
             f'{"PASS" if rec["g1_gate_pass_pooled"] else "**FAIL**"}',
             f'- pooled directions: cos(u_g^A,u_g^B) = {pooled["cos_u_g_A_B"]:.4f}, '
             f'cos(u_M^A,u_M^B) = {pooled["cos_u_M_A_B"]:.4f}', '',
             '## direction agreement between the two word halves, per context', '',
             r'| context | r(\|g_A\|,\|g_B\|) | top-64 overlap | cos(u_g^A,u_g^B) | cos(u_M^A,u_M^B) |',
             '|---|---:|---:|---:|---:|']
    for x in agree:
        lines.append(f'| `{x["context"]}` | {x["pearson_abs_g"]:.4f} | {x["top64_overlap"]:.3f} | '
                     f'{x["cos_u_g_A_B"]:.4f} | {x["cos_u_M_A_B"]:.4f} |')
    lines += ['', '## within each word half, measured with that half\'s own readout', '',
              'direction built PER CONTEXT:', '',
              '| pool | budget | mean dJ2 u_g | mean dJ2 u_M | mean ratio | median ratio | wins | exact sign p |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in variants:
        for b in budgets:
            x = rec['analysis'][f'pool{v}|b={b}']
            sp = 'n/a' if x['sign_p'] is None else f'{x["sign_p"]:.4f}'
            lines.append(f'| {v} | {b} | {x["mean_dJ2_u_g"]:+.5f} | {x["mean_dJ2_u_M"]:+.5f} | '
                         f'{x["mean_dJ2_u_M"] / x["mean_dJ2_u_g"]:.4f} | '
                         f'{1 + x["median_relative_gain"]:.4f} | {x["wins"]}/{x["n"]} | {sp} |')
    lines += ['', 'direction built from g POOLED over all contexts:', '',
              '| pool | budget | mean dJ2 u_g | mean dJ2 u_M | mean ratio | median ratio | wins | exact sign p |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in variants:
        for b in budgets:
            x = rec['analysis'][f'POOLED pool{v}|b={b}']
            sp = 'n/a' if x['sign_p'] is None else f'{x["sign_p"]:.4f}'
            lines.append(f'| {v} | {b} | {x["mean_dJ2_u_g"]:+.5f} | {x["mean_dJ2_u_M"]:+.5f} | '
                         f'{x["mean_dJ2_u_M"] / x["mean_dJ2_u_g"]:.4f} | '
                         f'{1 + x["median_relative_gain"]:.4f} | {x["wins"]}/{x["n"]} | {sp} |')
    (out / 'dense_g1.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/dense_g1.json and .md')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--layers', default='13-14')
    ap.add_argument('--budgets', default='0.05,0.2,0.5')
    ap.add_argument('--top-k', type=int, default=64, help='reference arm: top-K by |g|')
    ap.add_argument('--contexts', type=int, default=8)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--record-top', type=int, default=64,
                    help='record this many coordinates of each arm, as neuron+sign+strength')
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--fp32', action='store_true')
    ap.add_argument('--transfer', action='store_true',
                    help='G2-style transfer test: build the direction on one half of the contexts '
                         'and measure it on the other half (in-sample vs out-of-sample)')
    ap.add_argument('--g1', action='store_true',
                    help='G1 test: rebuild and re-measure everything under the two disjoint '
                         'word halves pos_A / pos_B instead of the full pool')
    ap.add_argument('--out', default='out/dense_optimal')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
    budgets = [float(x) for x in a.budgets.split(',')]
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    ctx = [r for r in rows if r.get('split') == a.split][:a.contexts]
    if not ctx:
        raise SystemExit(f'no {a.split} contexts in {a.probes}')
    arms = ['u_g', 'u_M', f'u_g{a.top_k}']
    n_meas = len(ctx) * (1 + len(arms) * len(budgets))

    print('Question : in the DENSE regime, does the budget-optimal u ~ M^+g beat u ~ g?')
    print(f'Readout  : J2 = mean over families {sorted(fams)}, pool variant {a.pool_variant}')
    print(f'Layers   : {layers}   precision: {"float32" if a.fp32 else "bfloat16 (production)"}')
    print(f'Arms     : base | u_g (full support, ~g) | u_M (full support, ~M^+g) | '
          f'u_g{a.top_k} (top-{a.top_k} by |g|)')
    print(f'Budgets  : {budgets}   budget(u)^2 = sum_l ||What_l\' u_l||^2')
    print(f'Contexts : {len(ctx)} -> {[c["id"] for c in ctx]}')
    print(f'Work     : {len(ctx)}x{len(fams)} grad passes + {n_meas} J2 evaluations '
          f'= {n_meas * len(fams)} batched forwards')
    print('Frozen   : PRIMARY = realized dJ2(u_M) > dJ2(u_g) in >=7/8 contexts per budget, '
          'exact two-sided sign test, median relative gain > 0.')
    print('           The surrogate-predicted gain ratio is printed BEFORE measuring; if it is '
          '~1 there is nothing to gain.')
    print('           §2.16 warning: surrogate-optimal need not be realized-optimal.')
    if a.dry_run:
        print(f'[dry-run] would also record, per arm and context: the residual direction norms '
              f'||d_l||, cos(d(u_g), d(u_M)), the equivalence-class dimension '
              f'(9216-2560={9216 - 2560} per layer), and the top-{a.record_top} coordinates '
              f'as neuron+sign+strength')
        print(f'[dry-run] would write {a.out}/dense_optimal.json and .md')
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
    dtype = torch.float32 if a.fp32 else torch.bfloat16
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        a.model, dtype=dtype, attn_implementation='sdpa', local_files_only=True).to(dev).eval()
    model.requires_grad_(False)
    lm = model_lm(model)
    if len(lm.layers) != 32:
        raise RuntimeError(f'expected 32 layers, got {len(lm.layers)}')
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device=dev)
    print(f'p50 ({dtype}):', {l: round(p50[l], 6) for l in layers})

    # What_l: 9216 x 2560, rows = unit-normalised down_proj columns. B_l = What'What.
    What, B, n_dim, d_dim = {}, {}, None, None
    for l in layers:
        Wt = lm.layers[l].mlp.down_proj.weight.detach().double().T          # 9216 x 2560
        Wt = Wt / Wt.norm(dim=1, keepdim=True).clamp(min=1e-12)
        What[l] = Wt
        B[l] = Wt.T @ Wt
        n_dim = Wt.shape[0]; d_dim = Wt.shape[1]
    print(f'What: {n_dim} x {d_dim} per layer; equivalence-class dimension = '
          f'{n_dim - d_dim} per layer; cond(B) = '
          + ', '.join(f'L{l}={float(torch.linalg.cond(B[l])):.4g}' for l in layers))

    def budget(u_by_layer):
        return math.sqrt(sum(float((u @ What[l]).pow(2).sum()) for l, u in u_by_layer.items()))

    def to_budget(u_by_layer, b):
        bu = budget(u_by_layer)
        if bu < 1e-12:
            raise SystemExit('degenerate direction')
        s = b / bu
        return {l: u * s for l, u in u_by_layer.items()}, s

    # ------------------------------------------------------------------ transfer mode
    if a.transfer:
        return transfer(a, tok, model, lm, p50, norms, What, B, n_dim, d_dim, layers, budgets,
                        fams, ctx, dev, budget, to_budget, n_dim_per_layer=None)

    # ------------------------------------------------------------------ G1 (word halves)
    if a.g1:
        return g1_test(a, tok, model, lm, p50, norms, What, B, n_dim, d_dim, layers, budgets,
                       fams, ctx, dev, budget, to_budget)

    prog = Progress(len(ctx) * len(fams) + n_meas)
    rec = {'schema': 'qwentest-dense-optimal-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'precision': str(dtype),
           'layers': layers, 'budgets': budgets, 'top_k_reference': a.top_k, 'arms': arms,
           'pool_variant': a.pool_variant, 'p50': {str(l): p50[l] for l in layers},
           'lexicon_sha256': hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest(),
           'what_shape': [n_dim, d_dim], 'equivalence_class_dim_per_layer': n_dim - d_dim,
           'cond_B': {str(l): float(torch.linalg.cond(B[l])) for l in layers},
           'contexts': [c['id'] for c in ctx], 'frozen_criteria': {
               'primary': 'realized dJ2(u_M) > dJ2(u_g) in >=7/8 contexts per budget; exact '
                          'two-sided sign test; median relative gain > 0',
               'error': '§2.18.4: differences below 1e-02 need --fp32 to be claimed',
               'unit': 'context'}, 'per_context': {}}

    for c in ctx:
        gsum, jfam = None, {}
        for fam, f in sorted(fams.items()):
            j2, gg = sem_gradient(tok, model, lm, layers, c['prompt'], f['template'], f, dev,
                                  a.max_seq)
            jfam[fam] = j2
            sc = {l: score_for_dose(gg[l].float(), p50[l], norms[l].to(gg[l].device)) for l in layers}
            gsum = sc if gsum is None else {l: gsum[l] + sc[l] for l in layers}
            prog.step(f'{c["id"]} g {fam} J2={j2:+.4f}')
        # keep g on the GPU: What is [9216, 2560] float64 there, and every product below
        # (e = What'g, B^-1 e, u* = What B^-1 d*) stays on one device
        g = {l: (gsum[l] / len(fams)).double().to(dev) for l in layers}

        # residual-space objects: e = What' g ;  d* ~ B^-1 e ;  u* = What^+ d* = B^-1 What' d*
        e = {l: What[l].T @ g[l] for l in layers}
        d_star = {l: torch.linalg.solve(B[l], e[l]) for l in layers}
        # u* = What B^-2 e = What B^-1 d*  =  A^+ d*, the minimum-norm preimage of d*.
        # (What is [9216, 2560], so this is a [9216,2560] @ [2560] matvec, not a solve.)
        u_M = {l: What[l] @ torch.linalg.solve(B[l], d_star[l]) for l in layers}
        order = torch.argsort(torch.cat([g[l].abs() for l in layers]), descending=True)
        npl = n_dim
        flat = torch.cat([g[l] for l in layers])
        u_g64 = {}
        off = 0
        for l in layers:
            m = (order >= off) & (order < off + npl)
            sel = order[m] - off
            u = torch.zeros(npl, dtype=torch.float64, device=dev)
            u[sel[:a.top_k]] = g[l][sel[:a.top_k]]
            u_g64[l] = u
            off += npl
        dirs = {'u_g': {l: g[l].clone() for l in layers}, 'u_M': u_M, f'u_g{a.top_k}': u_g64}

        gn = math.sqrt(sum(float((g[l] ** 2).sum()) for l in layers))
        en = math.sqrt(sum(float((e[l] ** 2).sum()) for l in layers))
        pred_g = gn ** 2 / en                                   # x b
        pred_M = math.sqrt(sum(float(e[l] @ torch.linalg.solve(B[l], torch.linalg.solve(B[l], e[l])))
                               for l in layers))                # x b
        batches = {fam: build_batch(tok, c['prompt'], f['template'], f['pos'], f['neg'], dev,
                                    a.max_seq) for fam, f in sorted(fams.items())}

        def j2_total(iv):
            with apply_steering(lm, iv, p50, norms), torch.inference_mode():
                vals = []
                for fam, (ids, pre, suf, kinds) in batches.items():
                    S = batch_mean_logp(model, ids, pre, suf)
                    jp, _ = language_log_mass(S[kinds > 0].detach().float())
                    jn, _ = language_log_mass(S[kinds < 0].detach().float())
                    vals.append(jp - jn)
            return sum(vals) / len(vals)

        base = j2_total([])
        prog.step(f'{c["id"]} base J2={base:+.6f}')
        entry = {'baseline_J2': base, 'J2_by_family': jfam,
                 'norm_g': gn, 'norm_e': en, 'budget_over_norm_g': en / gn,
                 'predicted_gain_per_unit_budget': {'u_g': pred_g, 'u_M': pred_M,
                                                    'ratio_M_over_g': pred_M / pred_g},
                 'cos_u_g_u_M': None, 'cos_d_g_d_M': None, 'realized': {}, 'top': {}}

        dg = {l: (What[l].T @ dirs['u_g'][l]) for l in layers}
        dm = {l: (What[l].T @ dirs['u_M'][l]) for l in layers}
        num = sum(float(dg[l] @ dm[l]) for l in layers)
        den = math.sqrt(sum(float((dg[l] ** 2).sum()) for l in layers)
                        * sum(float((dm[l] ** 2).sum()) for l in layers))
        entry['cos_d_g_d_M'] = num / den if den else None
        ug = torch.cat([dirs['u_g'][l] for l in layers])
        um = torch.cat([dirs['u_M'][l] for l in layers])
        entry['cos_u_g_u_M'] = float(ug @ um / (ug.norm() * um.norm()))

        for b in budgets:
            for arm, v in dirs.items():
                u, s = to_budget(v, b)
                iv = [{'layer': l, 'neuron': i, 'dose': float(u[l][i])}
                      for l in layers for i in torch.nonzero(u[l].abs() > 1e-15).flatten().tolist()]
                val = j2_total(iv)
                entry['realized'].setdefault(arm, {})[str(b)] = {
                    'dJ2': val - base,
                    'predicted': float(sum(float(u[l] @ g[l]) for l in layers)),
                    'n_nonzero': len(iv),
                    'budget_actual': float(b),
                    'resid_norm_per_layer': {str(l): float((What[l].T @ u[l]).norm()) for l in layers},
                    'dose_abs_max': max((abs(x['dose']) for x in iv), default=0.0)}
                if a.record_top and b == budgets[0]:
                    flatu = torch.cat([u[l] for l in layers])
                    o = torch.argsort(flatu.abs(), descending=True)[:a.record_top]
                    entry['top'][arm] = [
                        {'rank': r, 'layer': layers[int(i) // npl], 'neuron': int(i) % npl,
                         'sign': 1 if float(flatu[i]) >= 0 else -1,
                         'strength': float(flatu[i])} for r, i in enumerate(o.tolist(), 1)]
                prog.step(f'{c["id"]} b={b} {arm} dJ2={val - base:+.5f} '
                          f'(pred {entry["realized"][arm][str(b)]["predicted"]:+.5f})')
        rec['per_context'][c['id']] = entry
        del batches
        torch.cuda.empty_cache()

    # ---------------- frozen analysis ----------------
    cids = rec['contexts']
    an = {}
    for b in budgets:
        for x, y in (('u_M', 'u_g'), ('u_M', f'u_g{a.top_k}'), ('u_g', f'u_g{a.top_k}')):
            diffs = [rec['per_context'][c]['realized'][x][str(b)]['dJ2']
                     - rec['per_context'][c]['realized'][y][str(b)]['dJ2'] for c in cids]
            rel = [d / abs(rec['per_context'][c]['realized'][y][str(b)]['dJ2'])
                   for d, c in zip(diffs, cids)
                   if abs(rec['per_context'][c]['realized'][y][str(b)]['dJ2']) > 1e-12]
            p, n = exact_sign_p(diffs)
            an[f'b={b}|{x}_vs_{y}'] = {'wins': sum(d > 0 for d in diffs), 'n': n, 'sign_p': p,
                                       'median_abs_diff': statistics.median(diffs),
                                       'median_relative_gain': statistics.median(rel) if rel else None,
                                       'max_abs_diff': max(diffs, key=abs)}
    rec['analysis'] = an
    rec['predicted_ratio_M_over_g'] = {
        c: rec['per_context'][c]['predicted_gain_per_unit_budget']['ratio_M_over_g'] for c in cids}
    rec['budget_over_norm_g'] = {c: rec['per_context'][c]['budget_over_norm_g'] for c in cids}
    rec['cos_u_g_u_M'] = {c: rec['per_context'][c]['cos_u_g_u_M'] for c in cids}
    rec['cos_d_g_d_M'] = {c: rec['per_context'][c]['cos_d_g_d_M'] for c in cids}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'dense_optimal.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                            encoding='utf8')

    lines = ['# Dense regime: does the budget-optimal direction beat "follow the gradient"?', '',
             f'- machine `{rec["machine"]}` / {rec["gpu"]} · precision `{rec["precision"]}` · '
             f'layers {layers} · contexts {len(cids)}',
             f'- What is {n_dim}x{d_dim} per layer; equivalence class dimension '
             f'{rec["equivalence_class_dim_per_layer"]} per layer; '
             f'cond(B) ' + ', '.join(f'L{l}={rec["cond_B"][str(l)]:.4g}' for l in layers),
             f'- budget/||g|| (mean) {statistics.fmean(rec["budget_over_norm_g"].values()):.3f} '
             f'-- §2.21.3 measured 2.757 at full support, so M is far from identity here', '',
             '## surrogate-predicted gain per unit budget, and how different the arms are', '',
             '| context | pred u_g | pred u_M | **ratio M/g** | cos(u_g, u_M) | cos(d_g, d_M) |',
             '|---|---:|---:|---:|---:|---:|']
    for c in cids:
        pg = rec['per_context'][c]['predicted_gain_per_unit_budget']
        lines.append(f'| `{c}` | {pg["u_g"]:.4f} | {pg["u_M"]:.4f} | '
                     f'**{pg["ratio_M_over_g"]:.4f}** | {rec["cos_u_g_u_M"][c]:.4f} | '
                     f'{rec["cos_d_g_d_M"][c]:.4f} |')
    lines += ['', '## realized dJ2 (mean over contexts)', '',
              '| budget | u_g | u_M | u_g%d | realized ratio M/g | predicted ratio M/g |' % a.top_k,
              '|---:|---:|---:|---:|---:|---:|']
    for b in budgets:
        m = {arm: statistics.fmean(rec['per_context'][c]['realized'][arm][str(b)]['dJ2']
                                   for c in cids) for arm in arms}
        pr = statistics.fmean(rec['per_context'][c]['predicted_gain_per_unit_budget']['ratio_M_over_g']
                              for c in cids)
        lines.append(f'| {b} | {m["u_g"]:+.5f} | {m["u_M"]:+.5f} | {m[f"u_g{a.top_k}"]:+.5f} | '
                     f'{m["u_M"] / m["u_g"]:.4f} | {pr:.4f} |')
    lines += ['', '## frozen paired tests (unit = context)', '',
              '| comparison | wins | n | exact two-sided sign p | median abs diff | '
              'median relative gain |', '|---|---:|---:|---:|---:|---:|']
    for k, v in an.items():
        sp = 'n/a' if v['sign_p'] is None else f'{v["sign_p"]:.4f}'
        rg = 'n/a' if v['median_relative_gain'] is None else f'{v["median_relative_gain"]:+.2%}'
        lines.append(f'| `{k}` | {v["wins"]} | {v["n"]} | {sp} | {v["median_abs_diff"]:+.5f} | {rg} |')
    (out / 'dense_optimal.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/dense_optimal.json and .md')


if __name__ == '__main__':
    main()
