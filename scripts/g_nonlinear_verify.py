#!/usr/bin/env python3
"""Does a real G actually pay off on the NONLINEAR model? (plan 2.15 left this open)

Everything in plan §2.15 is linear-surrogate algebra. This is the measurement that decides
whether any of it matters: at a MATCHED PHYSICAL BUDGET, inject the direction each cost
matrix implies and read the REALIZED J2 off the frozen model.

Budget (exact, not approximate). Steering adds alpha_n = u_n * p50_l / ||W_down[:,n]|| to
MLP neuron n at every position, and down_proj is the MLP's last linear map, so the
residual-stream perturbation at layer l is exactly

    dresid_l = sum_n alpha_n * W_down[:,n]      =>   ||dresid_l||^2 / p50_l^2 = u_l' M_l u_l

with M_l the Gram matrix of the UNIT-normalised columns. Define

    budget(u) = sqrt(sum_l u_l' M_l u_l) = sqrt(u' M u),   M block-diagonal by layer

For a single coordinate this reduces to |dose| exactly, so it generalises the convention
every earlier result in this repo used. M is read off the frozen weights -- no backwards.

Arms, all on the SAME support S unless noted, so any difference is the cost matrix alone:
    u_I     ~ g                     (G = I, current practice everywhere)
    u_J     ~ Ghat^-1 g             (G = J'J on J2's own output space; plan 2.15.3)
    u_M     ~ M^-1 g                (G = the budget metric itself; optimal in the surrogate
                                     for THIS budget by construction -- the honest favourite)
    u_R     ~ random unit direction (control: is any direction as good?)
    u_known = L14N8949 alone at dose b  (anchor: should reproduce the plan-06 magnitude)
    pair_I / pair_J = the 2-coordinate choice of each rule (the discrete decision that
                     differed in 2/3 contexts in plan 2.15.3)

FROZEN criteria (written before the run; do not tune after seeing numbers).
  Unit of analysis = CONTEXT. Budgets are nested scalings of one direction, i.e. correlated,
  so they are reported as a profile and the primary statistic pools them per context first.
  PRIMARY   realized dJ2(u_J) > realized dJ2(u_I) in >= 7/8 contexts on the per-context mean
            over budgets; report the exact two-sided sign test p (8/8 -> 0.0078, 7/8 -> 0.070).
            Also require median relative gain > 0.
  SECONDARY same paired test for pair_J vs pair_I; and calibration realized/predicted per arm.
  CONTROL   u_R must NOT match u_J; if it does, the direction carries no information and the
            whole comparison is void -- report that instead of a win.
  GUARD     report non-monotonicity across budgets rather than quoting the best budget.

GPU: per context, (1 + n_arms * n_budgets) J2 evaluations, each = one batched forward per
template family. No backwards here: Ghat and g are read from a `g_j2_space.py` result.

    [i/N] elapsed, eta ...   per J2 evaluation (package progress.Progress)
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
    """Two-sided exact binomial sign test; zeros excluded. No iid claim beyond the unit used."""
    v = [d for d in diffs if abs(d) > 1e-12]
    n = len(v)
    if n == 0:
        return None, 0
    k = min(sum(x > 0 for x in v), sum(x < 0 for x in v))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--g-from', required=True, help='g_j2_space.py result JSON (Ghat and g per context)')
    ap.add_argument('--budgets', default='0.05,0.2,0.5',
                    help='comma list; budget = sqrt(u\'Mu), equal to |dose| for one coordinate')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--out', default='out/g_nonlinear')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    src = json.loads(Path(a.g_from).read_text(encoding='utf8'))
    coords = [(x['layer'], x['neuron']) for x in src['coordinates']]
    K = len(coords)
    layers = src['layers']
    per_ctx = {cid: r for cid, r in src['per_context'].items() if 'g' in r}
    dropped = sorted(set(src['per_context']) - set(per_ctx))
    if dropped:
        print(f'WARNING: {len(dropped)} context(s) in {Path(a.g_from).name} carry no g vector and '
              f'are dropped: {dropped}', file=sys.stderr)
    if not per_ctx:
        raise SystemExit(f'no context in {a.g_from} carries a g vector; re-run g_j2_space.py '
                         f'with the version that records it')
    budgets = [float(x) for x in a.budgets.split(',')]
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    if hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest() != src['lexicon_sha256']:
        raise SystemExit('lexicon differs from the one Ghat was built with; refusing to mix')
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    by_id = {r['id']: r for r in rows}
    missing = [c for c in per_ctx if c not in by_id]
    if missing:
        raise SystemExit(f'contexts not found in probes: {missing}')

    arms = ['u_I', 'u_J', 'u_M', 'u_R', 'u_known', 'pair_I', 'pair_J']
    n_eval = len(per_ctx) * (1 + len(arms) * len(budgets))
    print('Question : at a MATCHED physical budget, does a real G beat G=I on the NONLINEAR model?')
    print(f'Readout  : J2 = mean over families {sorted(fams)} (pool variant {a.pool_variant}), '
          f'lexicon sha256 {src["lexicon_sha256"][:16]}…')
    print(f'G source : {Path(a.g_from).name}  ({src["machine"]}, m={src["projections"]}, |S|={K})')
    print(f'Contexts : {len(per_ctx)} -> {sorted(per_ctx)}')
    print(f'Budgets  : {budgets}   (budget = sqrt(u\'Mu); equals |dose| for a single coordinate)')
    print(f'Arms     : {arms}')
    print(f'Frozen   : PRIMARY = dJ2(u_J) > dJ2(u_I) in >=7/{len(per_ctx)} contexts on the per-context '
          f'mean over budgets, exact two-sided sign test; CONTROL u_R must not match u_J')
    print(f'Work     : {n_eval} J2 evaluations = {n_eval * len(fams)} batched forwards, no backwards')
    if a.dry_run:
        for cid, r in sorted(per_ctx.items()):
            print(f'  {cid}: |g|={r["J2_gradient_abs_norm"]:.4f} cond(Ghat)={r["condition_number"]:.3g}')
        print(f'[dry-run] would write {a.out}/g_nonlinear.json and .md')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import column_norms, estimate_p50, load_model
    from progress import Progress
    from semantic import j2_value

    tok, model, lm = load_model(a.model)
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device='cuda:0')
    if max(abs(src['p50'][str(l)] - p50[l]) / p50[l] for l in layers) > 0.01:
        raise SystemExit('p50 differs from the Ghat run by >1%; budgets would not be comparable')

    # ---- M: Gram matrix of unit-normalised W_down columns, block-diagonal by layer ----
    W = {l: lm.layers[l].mlp.down_proj.weight.detach().float() for l in layers}
    M = torch.zeros(K, K, dtype=torch.float64)
    for l in layers:
        idx = [i for i, (ll, _) in enumerate(coords) if ll == l]
        if not idx:
            continue
        cols = W[l][:, [coords[i][1] for i in idx]].T                  # n_l x 2560
        cols = cols / cols.norm(dim=1, keepdim=True).clamp(min=1e-12)
        blk = (cols @ cols.T).double()
        for x, i in enumerate(idx):
            for y, j in enumerate(idx):
                M[i, j] = blk[x, y]
    off = ~torch.eye(K, dtype=torch.bool)
    same = torch.zeros(K, K, dtype=torch.bool)
    for i in range(K):
        for j in range(K):
            same[i, j] = coords[i][0] == coords[j][0]
    Mo = M[off & same].abs()
    print(f'M (physical budget metric): same-layer off-diag |M_ij| median {float(Mo.median()):.4f} '
          f'max {float(Mo.max()):.4f}  -> M is {"near-identity" if float(Mo.median()) < 0.05 else "NOT near-identity"}')

    def budget_of(u):
        return float(torch.sqrt(torch.clamp(u @ M @ u, min=0)))

    def to_budget(v, b):
        n = budget_of(v)
        if n < 1e-12:
            raise SystemExit('degenerate direction')
        return v * (b / n)

    def joint_Q(Gm, g, pair):
        i, j = pair
        aa, bb, cc = Gm[i, i], Gm[j, j], Gm[i, j]
        return float((bb * g[i] ** 2 - 2 * cc * g[i] * g[j] + aa * g[j] ** 2)
                     / max(float(aa * bb - cc ** 2), 1e-30))

    def best_pair(Gm, g):
        best, bq = None, None
        for i in range(K):
            for j in range(i + 1, K):
                q = joint_Q(Gm, g, (i, j))
                if bq is None or q > bq:
                    bq, best = q, (i, j)
        return best

    dev = 'cuda:0'

    def j2_total(prompt, interventions):
        vals = [j2_value(tok, model, lm, p50, norms, prompt, f['template'], f,
                         interventions or None, dev, a.max_seq) for f in fams.values()]
        return sum(vals) / len(vals)

    prog = Progress(n_eval)
    rec = {'schema': 'qwentest-g-nonlinear-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'layers': layers,
           'k': K, 'budgets': budgets, 'arms': arms, 'g_from': str(Path(a.g_from).name),
           'lexicon_sha256': src['lexicon_sha256'], 'pool_variant': a.pool_variant,
           'coordinates': [{'layer': l, 'neuron': n} for l, n in coords],
           'M_same_layer_offdiag_abs_median': float(Mo.median()),
           'M_same_layer_offdiag_abs_max': float(Mo.max()),
           'M_condition_number': float(torch.linalg.cond(M)),
           'frozen_criteria': {
               'primary': 'dJ2(u_J) > dJ2(u_I) in >=7/8 contexts on the per-context mean over '
                          'budgets; exact two-sided sign test; median relative gain > 0',
               'control': 'u_R must not match u_J',
               'unit': 'context (budgets are nested scalings, reported as a profile)'},
           'per_context': {}}

    for cid in sorted(per_ctx):
        r = per_ctx[cid]
        g = torch.tensor(r['g'], dtype=torch.float64)
        G = torch.tensor(r['G'], dtype=torch.float64)
        prompt = by_id[cid]['prompt']
        Gi = torch.linalg.pinv(G)
        rng = random.Random(a.seed + int(hashlib.sha256(cid.encode()).hexdigest()[:8], 16))
        wR = torch.tensor([rng.gauss(0, 1) for _ in range(K)], dtype=torch.float64)

        dirs = {'u_I': g.clone(), 'u_J': Gi @ g, 'u_M': torch.linalg.solve(M, g), 'u_R': wR}
        pi = sorted(torch.topk(g.abs(), 2).indices.tolist())
        pj = list(best_pair(G, g))
        known = coords.index((14, 8949)) if (14, 8949) in coords else None

        base = j2_total(prompt, [])
        prog.step(f'{cid} BASE J2={base:+.4f}')
        entry = {'baseline_J2': base, 'abs_g': float(g.norm()),
                 'predicted_per_budget': {}, 'realized': {}, 'directions': {}, 'pairs': {
                     'pair_I': [coords[i] for i in pi], 'pair_J': [coords[i] for i in pj],
                     'agree': pi == pj}}
        for b in budgets:
            uk = None
            if known is not None:
                uk = torch.zeros(K, dtype=torch.float64); uk[known] = 1.0
            for arm in arms:
                if arm == 'u_known':
                    u = uk * b                      # single coordinate: budget == |dose|
                    support = [known]
                elif arm == 'pair_I':
                    v = torch.zeros(K, dtype=torch.float64)
                    v[pi] = g[pi]; u = to_budget(v, b); support = pi
                elif arm == 'pair_J':
                    v = torch.zeros(K, dtype=torch.float64)
                    v[pj] = (Gi @ g)[pj]; u = to_budget(v, b); support = pj
                else:
                    u = to_budget(dirs[arm], b); support = list(range(K))
                iv = [{'layer': coords[i][0], 'neuron': coords[i][1], 'dose': float(u[i])}
                      for i in support if abs(float(u[i])) > 1e-15]
                val = j2_total(prompt, iv)
                entry['realized'].setdefault(arm, {})[str(b)] = val - base
                entry['predicted_per_budget'].setdefault(arm, {})[str(b)] = float(g @ u)
                entry['directions'][arm] = {'budget_actual': budget_of(u), 'n_nonzero': len(iv),
                                            'dose_abs_max': max((abs(float(u[i])) for i in support),
                                                                default=0.0)}
                prog.step(f'{cid} b={b} {arm} dJ2={val - base:+.5f} '
                          f'(pred {float(g @ u):+.5f})')
        rec['per_context'][cid] = entry

    # ---------------- frozen analysis ----------------
    cids = sorted(per_ctx)

    def per_ctx_mean(arm):
        return {c: statistics.fmean(rec['per_context'][c]['realized'][arm].values())
                for c in cids}

    an = {}
    for x, y in (('u_J', 'u_I'), ('u_J', 'u_M'), ('u_J', 'u_R'), ('u_I', 'u_R'),
                 ('u_M', 'u_I'), ('pair_J', 'pair_I')):
        mx, my = per_ctx_mean(x), per_ctx_mean(y)
        diffs = [mx[c] - my[c] for c in cids]
        p, n = exact_sign_p(diffs)
        rel = [(mx[c] - my[c]) / abs(my[c]) for c in cids if abs(my[c]) > 1e-12]
        an[f'{x}_vs_{y}'] = {'wins': sum(d > 0 for d in diffs), 'n': n, 'sign_p': p,
                             'median_abs_diff': statistics.median(diffs),
                             'median_relative_gain': statistics.median(rel) if rel else None,
                             'per_context_diff': {c: d for c, d in zip(cids, diffs)}}
    rec['analysis'] = an
    rec['profile_by_budget'] = {
        arm: {b: {c: rec['per_context'][c]['realized'][arm][b] for c in cids}
              for b in map(str, budgets)} for arm in arms}
    rec['calibration'] = {
        arm: {b: [rec['per_context'][c]['realized'][arm][b]
                  / rec['per_context'][c]['predicted_per_budget'][arm][b]
                  if abs(rec['per_context'][c]['predicted_per_budget'][arm][b]) > 1e-15 else None
                  for c in cids] for b in map(str, budgets)} for arm in arms}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'g_nonlinear.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf8')

    lines = ['# Realized vs surrogate: does a real G pay off on the nonlinear model?', '',
             f'- machine `{rec["machine"]}` / {rec["gpu"]} · |S| = {K} · contexts {len(cids)} · '
             f'budgets {budgets}',
             f'- budget = sqrt(u\'Mu), M = Gram of unit-normalised W_down columns (block-diagonal '
             f'by layer); same-layer off-diag |M_ij| median {rec["M_same_layer_offdiag_abs_median"]:.4f}, '
             f'max {rec["M_same_layer_offdiag_abs_max"]:.4f}, cond(M) {rec["M_condition_number"]:.3g}',
             f'- G and g from `{Path(a.g_from).name}`; readout J2 = mean over {sorted(fams)}', '',
             '## realized dJ2 per context (mean over budgets)', '',
             '| context | BASE J2 | u_I | u_J | u_M | u_R | u_known | pair_I | pair_J | pair rules agree |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in cids:
        e = rec['per_context'][c]
        m = {arm: statistics.fmean(e['realized'][arm].values()) for arm in arms}
        lines.append(f'| `{c}` | {e["baseline_J2"]:+.4f} | ' + ' | '.join(
            f'{m[arm]:+.5f}' for arm in arms) + f' | {"yes" if e["pairs"]["agree"] else "**no**"} |')
    lines += ['', '## frozen paired tests (unit = context)', '',
              '| comparison | wins | n | exact two-sided sign p | median abs diff | median relative gain |',
              '|---|---:|---:|---:|---:|---:|']
    for k, v in an.items():
        sp = 'n/a' if v['sign_p'] is None else f'{v["sign_p"]:.4f}'
        rg = 'n/a' if v['median_relative_gain'] is None else f'{v["median_relative_gain"]:+.1%}'
        lines.append(f'| `{k}` | {v["wins"]} | {v["n"]} | {sp} | '
                     f'{v["median_abs_diff"]:+.5f} | {rg} |')
    lines += ['', '## profile by budget (no cherry-picking: all levels shown)', '',
              '| arm | ' + ' | '.join(f'b={b}' for b in budgets) + ' |',
              '|---|' + '---:|' * len(budgets)]
    for arm in arms:
        cells = [statistics.fmean(rec['profile_by_budget'][arm][str(b)].values()) for b in budgets]
        lines.append(f'| `{arm}` | ' + ' | '.join(f'{v:+.5f}' for v in cells) + ' |')
    lines += ['', '## calibration: realized / surrogate-predicted (median over contexts)', '',
              '| arm | ' + ' | '.join(f'b={b}' for b in budgets) + ' |',
              '|---|' + '---:|' * len(budgets)]
    for arm in arms:
        cells = []
        for b in budgets:
            v = [x for x in rec['calibration'][arm][str(b)] if x is not None]
            cells.append(statistics.median(v) if v else None)
        lines.append(f'| `{arm}` | ' + ' | '.join('n/a' if c is None else f'{c:.3f}' for c in cells) + ' |')
    (out / 'g_nonlinear.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/g_nonlinear.json and .md')


if __name__ == '__main__':
    main()
