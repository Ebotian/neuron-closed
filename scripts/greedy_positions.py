#!/usr/bin/env python3
"""Positions as the deliverable: which (layer, neuron) list should we NAME, chosen by the
closed form itself rather than by |gradient|.

Why this experiment exists (the deliverable requirement, stated first). Every position list
this repo has produced so far ranks coordinates by mean |g| or by a bootstrap lower bound of
|g| (support_sweep.py, lcb_rank.py). Neither of those is the closed form. The closed form is

    u_S = M_SS^-1 g_S ,      budget(u)^2 = u' M u ,      predicted effect  Q(S) = g_S' M_SS^-1 g_S

so the closed form's own statement about positions is: pick the set S that maximises Q(S).
That selection rule has never been run. This script runs it and prices the result.

Selection. Greedy forward maximisation of Q(S) over the 18,432 coordinates of layers 13-14,
where M is block-diagonal by layer (M = Gram of unit-normalised W_down columns, so M_SS is the
exact cost metric of plan 2.16.1 and M_ii = 1). Greedy is not claimed to be optimal; it is the
cheap rule that reports Q at every K, and every K is then MEASURED, so the surrogate's quality
cannot hide.

Rank-1 bookkeeping (exact, no approximation). With P = M_SS^-1, appending coordinate c changes
the objective by

    dQ(c) = (g_c - M_cS P g_S)^2 / (1 - M_cS P M_Sc)

and P is updated by the bordering (Schur) formula. So the greedy step is O(|S| x 18,432), and
the whole nested family S_1 subset ... subset S_256 costs seconds.

What is measured, at matched physical budget, on the frozen readout J2 (same 8 heldout
contexts, same 3 budgets, same lexicon as support_sweep.py, so the numbers are directly
comparable with results/support_sweep_c8.json):
  arm GREEDY  per context: support chosen from THAT context's g, measured on that context
  arm pooled: support chosen from g pooled over the 8 contexts, measured on ALL 8 contexts
              (this is the list a paper would print: one list, no per-context tuning)
  arm topg_cf  support = top-K by |g| (ranked from THIS context's g), but the DOSE is the
              closed form M_SS^-1 g_S. This is what makes the comparison about the LIST alone:
              without it, "greedy vs |g|" would change both the list and the dose rule, and a win
              could not be attributed. §2.16 measured the dose-rule change alone at K=32 as +0.1%,
              so the bound is known, but the arm is cheap and removes the ambiguity.
  arm |g|     read from support_sweep_c8.json (already committed, same protocol; that arm is
              top-K by |g| with u ~ g, i.e. the historical list AND the historical dose rule)
  arm RANDOM  same K, drawn per context (control: is any list as good?)

FROZEN criteria (written before the run; do not tune after seeing numbers).
  PRIMARY    at K = 64 AND K = 256: per-context dJ2(greedy) > dJ2(topg_cf) in >= 7/8 contexts
             (exact two-sided sign test), median relative gain > 0. topg_cf is the |g| LIST with
             the SAME closed-form dose, so this isolates the selection rule.
  SECONDARY  the POOLED greedy list at K = 64, measured on all 8 contexts, must retain >= 73%
             of the full-support effect (73% is the level the |g| ranking only reaches at
             K = 256, so this is "1/4 of the coordinates for the same effect").
  STABILITY  median pairwise Jaccard of the 8 per-context greedy lists at K = 64 must be
             >= 0.30 — deliberately the same threshold as the registered G1 top-64 overlap
             gate, not a number invented here. A list that is not stable is not a result.
  ERROR      §2.18.4: a difference below 1e-02 is not claimed from bf16 alone.
  GUARD      report the whole K curve; no cherry-picked K.
  NOT RUN    word-half (G1-style pos_A/pos_B) stability is stage 2 and is NOT part of this
             run; it is registered in the plan and may only be quoted after it runs.

Outputs: results/greedy_positions.{json,md} plus one CSV per K for the pooled list
(layer, neuron, sign, dose at the smallest budget, rank), so the deliverable is an actual
printable table and not only a curve.

GPU: 8 contexts x 2 families grad passes (g), 4 greedy solves per context plus one pooled
(CPU-side algebra on an 18,432^2 float64 matrix, ~3 GB), then
8 x |K| x |budgets| x 2 arms J2 evaluations. Steering uses the frozen engine.apply_steering.

    [i/N] elapsed, eta ...   per J2 evaluation (package progress.Progress)
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, random, statistics, sys
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


def greedy_M(M, g, kmax, torch):
    """Greedy forward maximisation of Q(S) = g_S' M_SS^-1 g_S. Returns (S, Q trace)."""
    diag = torch.diagonal(M)
    N = M.shape[0]
    S, Qs, P = [], [], torch.zeros(0, 0, dtype=torch.float64, device=g.device)
    for _ in range(kmax):
        k = len(S)
        if k:
            v = M[:, S]                       # N x k
            y = P @ g[S]                      # k
            a = v @ y                         # N
            b = ((v @ P) * v).sum(1)          # N
        else:
            a = torch.zeros(N, dtype=torch.float64, device=g.device)
            b = torch.zeros(N, dtype=torch.float64, device=g.device)
        d = diag - b
        num = g - a
        gain = num * num / d.clamp(min=1e-12)
        gain[d <= 1e-10] = -float('inf')
        if S:
            gain[torch.tensor(S, device=gain.device)] = -float('inf')
        c = int(torch.argmax(gain))
        if d[c] <= 1e-10:
            raise SystemExit(f'greedy stalled at k={k}: no candidate with positive Schur '
                             f'complement (degenerate metric)')
        if k:
            vc = M[S, c]
            Pvc = P @ vc
            dstar = float(diag[c] - vc @ Pvc)
            newP = torch.zeros(k + 1, k + 1, dtype=torch.float64, device=g.device)
            newP[:k, :k] = P + torch.outer(Pvc, Pvc) / dstar
            newP[:k, k] = -Pvc / dstar
            newP[k, :k] = -Pvc / dstar
            newP[k, k] = 1.0 / dstar
            P = newP
        else:
            P = (1.0 / diag[c]).reshape(1, 1)
        S.append(c)
        Qs.append(float(g[S] @ (P @ g[S])))
    return S, Qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--reference', default=str(ROOT / 'results/support_sweep_c8.json'),
                    help='the committed |g| ranking curve to compare against (same protocol)')
    ap.add_argument('--topids', default=str(ROOT / 'results/support_sweep_topids_c8.json'),
                    help='committed per-context top-64 coordinate identity of the |g| ranking, '
                         'used for the LIST-level overlap with the greedy list')
    ap.add_argument('--layers', default='13-14')
    ap.add_argument('--contexts', type=int, default=8)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--ks', default='1,2,4,8,16,32,64,128,256')
    ap.add_argument('--budgets', default='0.05,0.2,0.5')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--seed', type=int, default=20261031)
    ap.add_argument('--out', default='out/greedy_positions')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
    ks = [int(x) for x in a.ks.split(',')]
    kmax = max(ks)
    budgets = [float(x) for x in a.budgets.split(',')]
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines()
            if l.strip()]
    ctx = [r for r in rows if r.get('split') == a.split][:a.contexts]
    if not ctx:
        raise SystemExit(f'no {a.split} contexts in {a.probes}')
    cids = [r['id'] for r in ctx]

    ref = None
    if Path(a.reference).is_file():
        ref = json.loads(Path(a.reference).read_text(encoding='utf8'))
        if ref.get('contexts') != cids:
            print(f'WARNING: reference contexts differ from this run: {ref.get("contexts")}')
        if ref.get('budgets') != budgets:
            print(f'WARNING: reference budgets {ref.get("budgets")} != {budgets}; the |g| '
                  f'column will be read at the budgets it actually has')

    n_eval = len(cids) * len(ks) * len(budgets) * 4
    print('Question : which neuron POSITIONS does the closed form itself name, and what do we lose?')
    print(f'Select   : greedy forward max of Q(S) = g_S\' M_SS^-1 g_S over '
          f'{len(layers)} x 9,216 coordinates (M = Gram of unit-normalised W_down columns)')
    print(f'Readout  : J2 = mean over families {sorted(fams)} (pool variant {a.pool_variant})')
    print(f'Contexts : {len(cids)} -> {cids}')
    print(f'K grid   : {ks}   budgets {budgets}')
    print(f'Arms     : GREEDY (per context) | POOLED greedy list measured on all contexts | '
          f'|g| from {Path(a.reference).name} | RANDOM')
    print(f'Frozen   : PRIMARY greedy > |g| at K=64 and K=256 in >=7/8 contexts (exact sign) with '
          f'median relative gain > 0; SECONDARY pooled K=64 retains >=73% of full support; '
          f'STABILITY median pairwise Jaccard at K=64 >= 0.30; ERROR <1e-02 needs fp32')
    print(f'Work     : {len(cids)}x{len(fams)} grad passes + 1 pooled greedy + '
          f'{n_eval} J2 evaluations')
    if a.dry_run:
        if ref:
            for b in map(str, budgets):
                if b in ref.get('mean_curve', {}):
                    print(f'  reference |g| curve at b={b}: ' + ', '.join(
                        f'K={k}:{ref["mean_curve"][b][k]:+.5f}' for k in ref['mean_curve'][b]
                        if int(k) in ks))
        print(f'[dry-run] would write {a.out}/greedy_positions.json/.md and one CSV per K')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import apply_steering, column_norms, estimate_p50, load_model, score_for_dose
    from progress import Progress
    from semantic import j2_value, sem_gradient

    tok, model, lm = load_model(a.model)
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device='cuda:0')
    dev = 'cuda:0'
    dims = {l: int(norms[l].numel()) for l in layers}
    N = sum(dims[l] for l in layers)
    off = {}
    acc = 0
    for l in layers:
        off[l] = acc
        acc += dims[l]

    def coord(i):
        for l in reversed(layers):
            if i >= off[l]:
                return l, i - off[l]
        raise IndexError(i)

    # ---- M: block-diagonal Gram of unit-normalised W_down columns (exact budget metric) ----
    M = torch.zeros(N, N, dtype=torch.float64, device=dev)
    for l in layers:
        Wt = lm.layers[l].mlp.down_proj.weight.detach().float().T.to(dev)   # 9216 x 2560
        Wt = (Wt / Wt.norm(dim=1, keepdim=True).clamp(min=1e-12)).double()
        M[off[l]:off[l] + dims[l], off[l]:off[l] + dims[l]] = Wt @ Wt.T
        del Wt
    torch.cuda.empty_cache()
    print(f'M built: {N} x {N} float64 on {dev}; diag range '
          f'{float(torch.diagonal(M).min()):.6f}..{float(torch.diagonal(M).max()):.6f}')

    prog = Progress(len(cids) * len(fams) + n_eval)

    # ---- per-context gradient, dose units, flattened into the coordinate order ----
    G = {}
    for c in ctx:
        gsum = None
        for fam, f in sorted(fams.items()):
            _j2, gg = sem_gradient(tok, model, lm, layers, c['prompt'], f['template'], f, dev,
                                   a.max_seq)
            sc = {l: score_for_dose(gg[l].float(), p50[l], norms[l].to(gg[l].device))
                  for l in layers}
            gsum = sc if gsum is None else {l: gsum[l] + sc[l] for l in layers}
            prog.step(f'{c["id"]} g {fam}')
        flat = torch.zeros(N, dtype=torch.float64, device=dev)
        for l in layers:
            flat[off[l]:off[l] + dims[l]] = gsum[l].double().to(dev) / len(fams)
        G[c['id']] = flat
    g_pool = torch.stack([G[c] for c in cids]).mean(0)
    print(f'g: per-context |g| median {statistics.median(float(G[c].norm()) for c in cids):.4f}; '
          f'pooled |g| = {float(g_pool.norm()):.4f}')

    def j2_total(prompt, interventions):
        vals = [j2_value(tok, model, lm, p50, norms, prompt, f['template'], f,
                         interventions or None, dev, a.max_seq) for f in fams.values()]
        return sum(vals) / len(vals)

    # ---- greedy lists: per context, and pooled (the printable one) ----
    lists, Qtrace = {}, {}
    for cid in cids:
        S, Qs = greedy_M(M, G[cid], kmax, torch)
        lists[cid] = S
        Qtrace[cid] = Qs
        prog.step(f'{cid} greedy K={kmax}')
    S_pool, Q_pool = greedy_M(M, g_pool, kmax, torch)
    lists['POOLED'] = S_pool
    Qtrace['POOLED'] = Q_pool
    prog.step('pooled greedy')

    def direction(S, g):
        idx = torch.tensor(S, dtype=torch.long, device=dev)
        Mss = M[idx][:, idx]
        u = torch.linalg.solve(Mss, g[idx])
        b0 = float(torch.sqrt(torch.clamp(g[idx] @ u, min=0)))
        return idx, u, b0

    def iv_of(S, g, b):
        idx, u, b0 = direction(S, g)
        if b0 < 1e-12:
            raise SystemExit('degenerate direction')
        u = u * (b / b0)
        out = []
        for j, i in enumerate(idx.tolist()):
            l, n = coord(i)
            out.append({'layer': l, 'neuron': n, 'dose': float(u[j])})
        return out

    # ---- realised readout ----
    rec = {'schema': 'qwentest-greedy-positions-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'precision': 'bfloat16',
           'layers': layers, 'ks': ks, 'budgets': budgets, 'contexts': cids,
           'coordinate_space': N, 'lexicon_sha256':
               hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest(),
           'p50': {str(l): p50[l] for l in layers},
           'frozen_criteria': {
               'primary': 'greedy > |g| at K=64 and K=256 in >=7/8 contexts, exact two-sided '
                          'sign test, median relative gain > 0',
               'secondary': 'pooled greedy list at K=64 retains >=73% of full-support effect',
               'stability': 'median pairwise Jaccard of the 8 per-context lists at K=64 >= 0.30 '
                            '(the registered G1 top-64 overlap gate)',
               'error': '§2.18.4: differences below 1e-02 need fp32',
               'not_run': 'word-half (G1) stability, stage 2'},
           'per_context': {c: {'baseline_J2': None, 'greedy': {}, 'topg_cf': {},
                               'pooled': {}, 'random': {},
                               'Q_surrogate': {str(k): Qtrace[c][k - 1] for k in ks}}
                           for c in cids},
           'pooled_list_Q_surrogate': {str(k): Q_pool[k - 1] for k in ks}}

    rng = random.Random(a.seed)
    base = {c: j2_total(ctx[i]['prompt'], []) for i, c in enumerate(cids)}
    for c in cids:
        rec['per_context'][c]['baseline_J2'] = base[c]
        prog.step(f'{c} BASE J2={base[c]:+.4f}')

    # |g| ranking per context, used for the topg_cf arm (list identical to the committed arm)
    rank_g = {c: torch.argsort(G[c].abs(), descending=True).tolist() for c in cids}

    for c in cids:
        prompt = ctx[cids.index(c)]['prompt']
        for K in ks:
            Sr = rng.sample(range(N), K)
            for arm, S in (('greedy', lists[c][:K]),
                           ('topg_cf', rank_g[c][:K]),
                           ('pooled', lists['POOLED'][:K]),
                           ('random', Sr)):
                g_use = g_pool if arm == 'pooled' else G[c]
                for b in budgets:
                    val = j2_total(prompt, iv_of(S, g_use, b))
                    rec['per_context'][c][arm].setdefault(str(K), {})[str(b)] = val - base[c]
                    prog.step(f'{c} K={K} b={b} {arm} dJ2={val - base[c]:+.5f}')

    # ---- stability: Jaccard of the per-context lists ----
    def jac(a1, b1):
        s1, s2 = set(a1), set(b1)
        return len(s1 & s2) / max(1, len(s1 | s2))

    stab = {}
    for K in ks:
        vals = [jac(lists[cids[i]][:K], lists[cids[j]][:K])
                for i in range(len(cids)) for j in range(i + 1, len(cids))]
        pooled = [jac(lists[c][:K], S_pool[:K]) for c in cids]
        stab[str(K)] = {'median_pairwise_jaccard': statistics.median(vals),
                        'min_pairwise_jaccard': min(vals),
                        'median_jaccard_vs_pooled': statistics.median(pooled)}
    rec['stability'] = stab

    rec['lists'] = {c: lists[c] for c in lists}

    # ---- LIST overlap with the committed |g| top-K (the registered prediction is about this) --
    if Path(a.topids).is_file():
        t = json.loads(Path(a.topids).read_text(encoding='utf8'))
        topg_lists = {}
        for cid, e in t.get('per_context', {}).items():
            lst = e.get('top_coordinates') or []
            topg_lists[cid] = [(int(x['layer']), int(x['neuron']))
                               for x in sorted(lst, key=lambda z: z['rank'])]

        def lc(i):
            return coord(i)

        ov = {}
        for K in ks:
            vals = []
            for c in cids:
                ref_list = topg_lists.get(c) or []
                if len(ref_list) < K:
                    continue
                A = {lc(i) for i in lists[c][:K]}
                B = set(ref_list[:K])
                vals.append(len(A & B) / max(1, len(A | B)))
            if vals:
                ov[str(K)] = {'median_jaccard_vs_topg': statistics.median(vals),
                              'min': min(vals), 'max': max(vals)}
        rec['list_overlap_vs_topg'] = ov
        pooled_ov = {}
        for K in ks:
            A = {lc(i) for i in S_pool[:K]}
            vals = []
            for c in cids:
                ref_list = topg_lists.get(c) or []
                if len(ref_list) < K:
                    continue
                vals.append(len(A & set(ref_list[:K])) / max(1, len(A | set(ref_list[:K]))))
            if vals:
                pooled_ov[str(K)] = statistics.median(vals)
        rec['pooled_list_overlap_vs_topg'] = pooled_ov
    else:
        print(f'NOTE: {a.topids} absent; list-level overlap not computed')

    # ---- compare with the committed |g| curve (same protocol) ----
    analysis = {}
    if ref:
        for K in ks:
            for b in budgets:
                key = str(K)
                gg = {c: rec['per_context'][c]['greedy'][key][str(b)] for c in cids}
                pp = {c: rec['per_context'][c]['pooled'][key][str(b)] for c in cids}
                rr = {c: rec['per_context'][c]['random'][key][str(b)] for c in cids}
                cc = {c: rec['per_context'][c]['topg_cf'][key][str(b)] for c in cids}
                gtop = {c: ref['per_context'][c]['curve'][str(b)][key]['dJ2']
                        for c in cids if str(b) in ref['per_context'][c]['curve']
                        and key in ref['per_context'][c]['curve'][str(b)]}
                if len(gtop) == len(cids):
                    for name, arm in (('greedy', gg), ('pooled', pp), ('random', rr),
                                  ('topg_cf', cc)):
                        diffs = [arm[c] - gtop[c] for c in cids]
                        p, n = exact_sign_p(diffs)
                        rel = [(arm[c] - gtop[c]) / abs(gtop[c]) for c in cids
                               if abs(gtop[c]) > 1e-12]
                        analysis[f'{name}_vs_topg|K={K}|b={b}'] = {
                            'wins': sum(d > 0 for d in diffs), 'n': n, 'sign_p': p,
                            'median_abs_diff': statistics.median(diffs),
                            'median_relative_gain': statistics.median(rel) if rel else None,
                            'mean_dJ2_arm': statistics.fmean(arm.values()),
                            'mean_dJ2_topg': statistics.fmean(gtop.values())}
    rec['analysis_vs_topg'] = analysis
    rec['retention'] = {
        'note': 'retention = pooled-greedy mean dJ2 / pooled-greedy at Kmax, per budget',
        'by_K': {str(K): {str(b): statistics.fmean(
            rec['per_context'][c]['pooled'][str(K)][str(b)] for c in cids) for b in budgets}
            for K in ks}}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'greedy_positions.json').write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf8')

    # ---- the printable deliverable: pooled lists as CSV ----
    for K in ks:
        S = S_pool[:K]
        idx, u, b0 = direction(S, g_pool)
        u = u * (budgets[0] / b0)
        lines = ['rank,layer,neuron,sign,dose_at_b%g,abs_dose,g_value,Q_share' % budgets[0]]
        Qprev = 0.0
        for j, i in enumerate(S):
            l, n = coord(i)
            Qj = Q_pool[j]
            share = (Qj - Qprev) / Q_pool[K - 1] if Q_pool[K - 1] else 0.0
            Qprev = Qj
            lines.append(f'{j + 1},{l},{n},{"+" if u[j] > 0 else "-"},'
                         f'{float(u[j]):.8f},{abs(float(u[j])):.8f},'
                         f'{float(g_pool[i]):.8f},{share:.6f}')
        (out / f'positions_K{K}.csv').write_text('\n'.join(lines) + '\n', encoding='utf8')

    md = ['# Which positions does the closed form itself name? (greedy max of Q(S))', '',
          f'- machine `{rec["machine"]}` / {rec["gpu"]} · {N} coordinates · contexts {len(cids)} · '
          f'budgets {budgets}',
          f'- selection: greedy forward max of `Q(S) = g_Sᵀ M_SS⁻¹ g_S`; budgets and readout as in '
          f'`{Path(a.reference).name}`', '',
          '## realised dJ2 by K (mean over 8 contexts)', '',
          '| K | greedy list + CF dose | `|g|` list + CF dose | pooled list | random list | '
          '`|g|` + `u∝g` (committed) |',
          '|---:|---:|---:|---:|---:|---:|']
    for K in ks:
        key = str(K)
        b = budgets[0]
        gr = statistics.fmean(rec['per_context'][c]['greedy'][key][str(b)] for c in cids)
        cc = statistics.fmean(rec['per_context'][c]['topg_cf'][key][str(b)] for c in cids)
        po = statistics.fmean(rec['per_context'][c]['pooled'][key][str(b)] for c in cids)
        ra = statistics.fmean(rec['per_context'][c]['random'][key][str(b)] for c in cids)
        if ref:
            gt = statistics.fmean(ref['per_context'][c]['curve'][str(b)][key]['dJ2']
                                  for c in cids
                                  if key in ref['per_context'][c]['curve'].get(str(b), {}))
            ratio = po / gt if gt else float('nan')
        else:
            gt, ratio = float('nan'), float('nan')
        md.append(f'| {K} | {gr:+.5f} | {cc:+.5f} | {po:+.5f} | {ra:+.5f} | {gt:+.5f} |')
    md += ['', f'## frozen tests at b={budgets[0]} (paired over contexts)', '',
           '| comparison | K | wins | n | exact sign p | median abs diff | median rel gain |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for k, v in analysis.items():
        if not k.endswith(f'b={budgets[0]}'):
            continue
        name, K = k.split('|')[0], k.split('|')[1].split('=')[1]
        sp = 'n/a' if v['sign_p'] is None else f'{v["sign_p"]:.4f}'
        rg = 'n/a' if v['median_relative_gain'] is None else f'{v["median_relative_gain"]:+.1%}'
        md.append(f'| `{name}` | {K} | {v["wins"]} | {v["n"]} | {sp} | '
                  f'{v["median_abs_diff"]:+.5f} | {rg} |')
    md += ['', '## stability of the per-context lists', '',
           '| K | median pairwise Jaccard | min | median Jaccard vs pooled |',
           '|---:|---:|---:|---:|']
    for K in ks:
        s = stab[str(K)]
        md.append(f'| {K} | {s["median_pairwise_jaccard"]:.3f} | {s["min_pairwise_jaccard"]:.3f} | '
                  f'{s["median_jaccard_vs_pooled"]:.3f} |')
    if 'list_overlap_vs_topg' in rec:
        md += ['', '## does the closed-form rule pick the same positions as |g|?', '',
               '| K | Jaccard(greedy list, |g| list) median | min | max | pooled list vs |g| |',
               '|---:|---:|---:|---:|---:|']
        for K in ks:
            v = rec['list_overlap_vs_topg'].get(str(K))
            if v:
                md.append(f'| {K} | {v["median_jaccard_vs_topg"]:.3f} | {v["min"]:.3f} | '
                          f'{v["max"]:.3f} | '
                          f'{rec["pooled_list_overlap_vs_topg"].get(str(K), float("nan")):.3f} |')
    md += ['', '## pooled share of the full-support effect', '',
           '| K | ' + ' | '.join(f'b={b}' for b in budgets) + ' |',
           '|---|' + '---:|' * len(budgets)]
    for K in ks:
        cells = []
        for b in budgets:
            num = rec['retention']['by_K'][str(K)][str(b)]
            den = rec['retention']['by_K'][str(kmax)][str(b)]
            cells.append(f'{num / den:.1%}' if den else 'n/a')
        md.append(f'| {K} | ' + ' | '.join(cells) + ' |')
    (out / 'greedy_positions.md').write_text('\n'.join(md) + '\n', encoding='utf8')
    print('\n' + '\n'.join(md))
    print(f'\nwrote {out}/greedy_positions.json/.md and positions_K*.csv')


if __name__ == '__main__':
    main()
