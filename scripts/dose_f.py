#!/usr/bin/env python3
"""Item 4 / plan 2.34: the THIRD answer to "which G" --- the Fisher of an output-KL
constraint --- measured on the same 32-coordinate support as plan 2.16.

Why this is the one missing dimension. Every arm in this repo so far buys the same
thing: a claim about how expensive a perturbation is. Plan 2.16 priced a perturbation by
the residual displacement it produces (M = Gram of unit-normalised W_down columns). CANS
(arXiv 2610.35530 v1, read in 13_paper) prices it instead by the KL it opens between the
steered and the base OUTPUT distribution, whose curvature is the Fisher information:

    cost(u) = 1/2 u' F u  (+ lambda/2 ||u||^2),    F_ij = E[ dlogp/du_i * dlogp/du_j ]

so per-coordinate (diagonal) F_ii = E[ (d log p / du_i)^2 ]. Item 4 asks the only question
that separates the two price tags: with G = F instead of G = M, does the direction that
the closed form u ~ G^-1 g implies do BETTER on the real model? The missing dimension is
not another readout or another layer --- it is the cost matrix itself.

SCALE INVARIANCE (stated because it decides what has to be measured). For any c > 0,

    (cF + c*lambda*I)^-1 g = (1/c) (F + lambda I)^-1 g ,

so the DIRECTION u_F is invariant to the global scale of F. "Mean over tokens" versus
"sum over tokens" therefore cannot change the answer, and lambda's role collapses to one
free relative scale. Hence: lambda = median(F) (the paper's default damping, `--lambda-scale
1`) as the primary arm, and a lambda profile {0.1, 1, 10} x median as a sensitivity check
reported with cosines, never with a cherry-picked level.

THE FISHER RECIPE (what is actually estimated, and its two deviations from CANS).
  * counted tokens: CANS counts only NEW sampled tokens (temperature 1, top-p 1, N_cont=4
    x 64 tokens over WikiText-2). We have no WikiText-2 on the box and, more importantly,
    our own cost axis is not a corpus --- it is `engine.REFERENCE_TEXT`, the frozen
    neutral prose+code+opinion text every capability number in this repo is measured on.
    So the constraint is estimated on THAT text (all of it, plus 4 x 64 sampled
    continuations at temperature 1 / top-p 1), and every token position of REFERENCE_TEXT
    is counted once. Stated because it is a deviation, not a detail: F is then the Fisher
    of OUR cost axis, which is the honest choice for a three-way comparison with M.
  * diagonal only: F_ii, no cross terms. CANS does the same and its own Limitations section
    says it "retains only the Fisher diagonal ..., omitting neuron interactions and their
    joint roles" --- the boundary is inherited deliberately, and it is the same boundary
    our superadditivity measurement (6+1 = 7 analytic, 16 measured) sits on.

ESTIMATOR. One forward pass per sample under `capture_mlp_inputs`, then ONE backward per
counted token t, giving dlogp(x_t)/du exactly (no sampling of the Fisher, unlike
g_feasibility.py's J'J estimator). The gradient w.r.t. the captured down_proj pre-activation
is summed over positions and scaled by p50/||W_down[:,n]|| --- byte-for-byte the same
dose convention as `engine.score_for_dose`, so F is expressed in the same units as g and M.

SPLIT-HALF REPRODUCIBILITY (the G1/G2 discipline applied to F). Counted tokens are split
odd/even; each half gets its own F, its own lambda = median(F_half) and its own direction.
If cos(u_F^A, u_F^B) is low, u_F is an estimator artifact and the three-way comparison is
void --- that is reported BEFORE any realised number is interpreted.

ARMS, all on the SAME 32-coordinate support S from `--g-from` (plan 2.16's S), so any
difference is the cost matrix alone:
    u_I  ~ g                       (G = I, current practice)
    u_M  ~ M^-1 g                  (G = the budget metric; plan 2.16's honest favourite)
    u_F  ~ (diag(F) + lambda I)^-1 g   (G = Fisher of the output-KL constraint; NEW)
    u_R  ~ random unit direction    (control: is any direction as good?)

FROZEN criteria (written before the run; do not tune after seeing numbers).
  Unit of analysis = CONTEXT. Budgets are nested scalings of one direction (correlated),
  so they are a profile; the primary statistic pools them per context first.
  PRIMARY    realised dJ2(u_F) > realised dJ2(u_M) in >= 7/8 contexts on the per-context
             mean over budgets; exact two-sided sign test; median relative gain > 0.
  SECONDARY  the same paired test for u_F vs u_I (does the Fisher-grounded G at least beat
             ignoring the geometry?), and u_M vs u_I (replication of plan 2.16: 5/8, p=0.73).
  CONTROL    u_R must NOT match u_F; and cos(u_F^A,u_F^B) must be reported before any win.
  GUARD      report the full budget profile, never the best budget.

GPU: 4 forwards + ~1 backward per counted token for F (~40-60 s), plus
len(contexts) x (1 + 4 arms x |budgets|) J2 evaluations.

    [i/N] elapsed, eta ...   per counted token (Fisher) and per J2 evaluation (package
                             progress.Progress)
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
    ap.add_argument('--g-from', default=str(ROOT / '07_results/g_j2_space_k32_m192_c8.json'),
                    help='g_j2_space.py result JSON: supplies the 32-coordinate support S '
                         'and the per-context gradient g (plan 2.16 support, unchanged)')
    ap.add_argument('--budgets', default='0.05,0.2,0.5',
                    help='comma list; budget = sqrt(u\'Mu), equal to |dose| for one coordinate')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--samples', type=int, default=4,
                    help='sampled continuations of REFERENCE_TEXT (CANS N_cont = 4)')
    ap.add_argument('--sample-tokens', type=int, default=64,
                    help='tokens per sampled continuation (CANS: 64)')
    ap.add_argument('--ref-tokens', type=int, default=128,
                    help='count at most this many REFERENCE_TEXT positions (CANS: 128)')
    ap.add_argument('--lambda-scale', default='0.1,1,10',
                    help='lambda = scale * median(diag F); 1 is the paper default, the others '
                         'are a sensitivity profile (the direction is scale-invariant in F, not '
                         'in lambda)')
    ap.add_argument('--seed', type=int, default=20261031)
    ap.add_argument('--out', default='out/dose_f')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    src = json.loads(Path(a.g_from).read_text(encoding='utf8'))
    coords = [(x['layer'], x['neuron']) for x in src['coordinates']]
    K = len(coords)
    layers = src['layers']
    per_ctx = {cid: r for cid, r in src['per_context'].items() if 'g' in r}
    budgets = [float(x) for x in a.budgets.split(',')]
    lam_scales = [float(x) for x in a.lambda_scale.split(',')]
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    if hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest() != src['lexicon_sha256']:
        raise SystemExit('lexicon differs from the one the support was built with; refusing to mix')
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    by_id = {r['id']: r for r in rows}
    missing = [c for c in per_ctx if c not in by_id]
    if missing:
        raise SystemExit(f'contexts not found in probes: {missing}')

    arms = ['u_I', 'u_M', 'u_F', 'u_R']
    n_eval = len(per_ctx) * (1 + len(arms) * len(budgets))
    print('Question : does G = diag Fisher of an output-KL constraint beat G = M on the model?')
    print(f'Readout  : J2 = mean over families {sorted(fams)} (pool variant {a.pool_variant}), '
          f'lexicon sha256 {src["lexicon_sha256"][:16]}…')
    print(f'Support  : the SAME |S| = {K} coordinates and the same g as '
          f'{Path(a.g_from).name} (plan 2.16), layers {layers}')
    print(f'Contexts : {len(per_ctx)} -> {sorted(per_ctx)}')
    print(f'Budgets  : {budgets}   (budget = sqrt(u\'Mu); equals |dose| for a single coordinate)')
    print(f'Fisher   : engine.REFERENCE_TEXT (all positions, capped at {a.ref_tokens}) + '
          f'{a.samples} x {a.sample_tokens} sampled continuations, T=1/top-p=1; '
          f'one backward per counted token')
    print(f'Damping  : lambda = median(diag F) [paper default]; sensitivity at '
          f'{lam_scales} x median')
    print(f'Arms     : {arms}')
    print(f'Frozen   : PRIMARY = dJ2(u_F) > dJ2(u_M) in >=7/{len(per_ctx)} contexts on the '
          f'per-context mean over budgets, exact two-sided sign test, median relative gain > 0')
    print(f'           CONTROL u_R must not match u_F; cos(u_F^A,u_F^B) reported first')
    print(f'Work     : ~{a.ref_tokens + a.samples * a.sample_tokens} backwards for F + '
          f'{n_eval} J2 evaluations = {n_eval * len(fams)} batched forwards')
    if a.dry_run:
        for cid in sorted(per_ctx):
            r = per_ctx[cid]
            print(f'  {cid}: |g|={r["J2_gradient_abs_norm"]:.4f} cond(Ghat)={r["condition_number"]:.3g}')
        print(f'[dry-run] would write {a.out}/dose_f.json and .md')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import (REFERENCE_TEXT, capture_mlp_inputs, column_norms, estimate_p50,
                        load_model)
    from progress import Progress
    from semantic import j2_value

    tok, model, lm = load_model(a.model)
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device='cuda:0')
    if max(abs(src['p50'][str(l)] - p50[l]) / p50[l] for l in layers) > 0.01:
        raise SystemExit('p50 differs from the support run by >1%; budgets would not be comparable')

    # ---- M: Gram of unit-normalised W_down columns, block-diagonal by layer (as plan 2.16) ----
    W = {l: lm.layers[l].mlp.down_proj.weight.detach().float() for l in layers}
    M = torch.zeros(K, K, dtype=torch.float64)
    for l in layers:
        idx = [i for i, (ll, _) in enumerate(coords) if ll == l]
        if not idx:
            continue
        cols = W[l][:, [coords[i][1] for i in idx]].T
        cols = cols / cols.norm(dim=1, keepdim=True).clamp(min=1e-12)
        blk = (cols @ cols.T).double()
        for x, i in enumerate(idx):
            for y, j in enumerate(idx):
                M[i, j] = blk[x, y]
    idx_by_layer = {l: [i for i, (ll, _) in enumerate(coords) if ll == l] for l in layers}

    def budget_of(u):
        return float(torch.sqrt(torch.clamp(u @ M @ u, min=0)))

    def to_budget(v, b):
        n = budget_of(v)
        if n < 1e-12:
            raise SystemExit('degenerate direction')
        return v * (b / n)

    dev = 'cuda:0'
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    def damped_direction(Fdiag, lam_scale, g):
        """(diag(F) + lam I)^-1 g, lam = lam_scale * median(diag F). Scale-invariant in F."""
        lam = float(Fdiag.median()) * lam_scale
        return torch.linalg.solve(torch.diag(Fdiag + lam), g), lam

    # ------------------------------------------------------------------ Fisher diagonal
    ref_ids = tok(REFERENCE_TEXT, return_tensors='pt').input_ids[0].tolist()[:a.ref_tokens]
    seqs = []
    for s in range(a.samples):
        gen = model.generate(input_ids=torch.tensor([ref_ids], device=dev),
                             attention_mask=torch.ones(1, len(ref_ids), device=dev, dtype=torch.long),
                             max_new_tokens=a.sample_tokens, do_sample=True, temperature=1.0,
                             top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        seqs.append(gen[0].tolist())
    counted = [(s, t) for s, seq in enumerate(seqs)
               for t in range(1, len(seq))
               if (s == 0 and t < len(ref_ids)) or t >= len(ref_ids)]
    prog = Progress(len(counted) + n_eval)
    print(f'Fisher corpus: |REFERENCE_TEXT| = {len(ref_ids)} tokens, '
          f'{a.samples} continuations, {len(counted)} counted token positions')

    F = torch.zeros(K, dtype=torch.float64)
    Fh = [torch.zeros(K, dtype=torch.float64), torch.zeros(K, dtype=torch.float64)]
    nh = [0, 0]
    by_seq = {}
    for s, t in counted:
        by_seq.setdefault(s, []).append(t)

    for s, targets in by_seq.items():
        seq = torch.tensor([seqs[s]], device=dev, dtype=torch.long)
        with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
            out = model(input_ids=seq, use_cache=False)
            lsm = out.logits[0].float().log_softmax(-1)
            for n_t, t in enumerate(targets):
                lp = lsm[t - 1, seqs[s][t]]
                grads = torch.autograd.grad(lp, [cap[l] for l in layers],
                                            retain_graph=(n_t < len(targets) - 1))
                v = torch.zeros(K, dtype=torch.float64)
                for l, g in zip(layers, grads):
                    sc = g.detach().double().sum(dim=tuple(range(g.ndim - 1)))
                    sc = sc * (p50[l] / norms[l].to(sc.device).double().clamp(min=1e-12))
                    for i in idx_by_layer[l]:
                        v[i] = sc[coords[i][1]]
                v = v.cpu()
                h = t % 2
                F += v * v
                Fh[h] += v * v
                nh[h] += 1
                prog.step(f'Fisher token {s}:{t}')
            del out, lsm
        torch.cuda.empty_cache()
    n_tok = len(counted)
    F /= n_tok
    Fh = [x / max(1, n) for x, n in zip(Fh, nh)]
    lam = float(F.median())
    print(f'Fisher: {n_tok} tokens; diag range {float(F.min()):.4g}…{float(F.max()):.4g}; '
          f'lambda = median(F) = {lam:.6g}; ratio max/min = {float(F.max() / F.min()):.4g}')

    def cos(x, y):
        return float(x @ y / (x.norm() * y.norm() + 1e-300))

    # ---- split-half reproducibility of F itself, and of the direction it implies ----
    splits = []
    for cid in sorted(per_ctx):
        g = torch.tensor(per_ctx[cid]['g'], dtype=torch.float64)
        uA, lamA = damped_direction(Fh[0], 1.0, g)
        uB, lamB = damped_direction(Fh[1], 1.0, g)
        splits.append({'context': cid, 'cos_u_F_halfA_halfB': cos(uA, uB),
                       'cos_u_F_full_halfA': cos(damped_direction(F, 1.0, g)[0], uA),
                       'lambda_A': lamA, 'lambda_B': lamB})
    def safe_r(x, y):
        """statistics.correlation raises on a constant vector; a degenerate half is a result."""
        try:
            return statistics.correlation(x, y)
        except statistics.StatisticsError:
            return None

    diag_half_r = safe_r([float(x) for x in Fh[0]], [float(x) for x in Fh[1]])
    diag_half_r_sqrt = safe_r([math.sqrt(float(x)) for x in Fh[0]],
                              [math.sqrt(float(x)) for x in Fh[1]])
    sp_half = statistics.median(x['cos_u_F_halfA_halfB'] for x in splits)
    fmt_r = lambda v: 'n/a' if v is None else f'{v:.4f}'
    print(f'split-half: r(diag F) = {fmt_r(diag_half_r)} (raw) / {fmt_r(diag_half_r_sqrt)} '
          f'(sqrt); median cos(u_F^A,u_F^B) = {sp_half:.4f}')

    # ---- directions and their cosines against the plan-2.16 arms ----
    def dirs_for(g):
        d = {'u_I': g, 'u_M': torch.linalg.solve(M, g)}
        d['u_F'], lam_used = damped_direction(F, 1.0, g)
        return d, lam_used

    lam_profile = {}
    for s in lam_scales:
        cs, cs_M, cs_I = [], [], []
        for cid in sorted(per_ctx):
            g = torch.tensor(per_ctx[cid]['g'], dtype=torch.float64)
            uF, _ = damped_direction(F, s, g)
            uF1, _ = damped_direction(F, 1.0, g)
            cs.append(cos(uF, uF1))
            cs_M.append(cos(uF, torch.linalg.solve(M, g)))
            cs_I.append(cos(uF, g))
        lam_profile[str(s)] = {'median_cos_vs_primary_lambda': statistics.median(cs),
                               'median_cos_vs_u_M': statistics.median(cs_M),
                               'median_cos_vs_u_I': statistics.median(cs_I)}

    cosines = {'cos_u_M_u_F': [], 'cos_u_I_u_F': [], 'cos_u_I_u_M': [], 'cos_u_M_u_I': []}
    for cid in sorted(per_ctx):
        g = torch.tensor(per_ctx[cid]['g'], dtype=torch.float64)
        d, _ = dirs_for(g)
        cosines['cos_u_M_u_F'].append(cos(d['u_M'], d['u_F']))
        cosines['cos_u_I_u_F'].append(cos(d['u_I'], d['u_F']))
        cosines['cos_u_I_u_M'].append(cos(d['u_I'], d['u_M']))
        cosines['cos_u_M_u_I'].append(cos(d['u_M'], d['u_I']))
    cos_summary = {k: {'median': statistics.median(v),
                       'per_context': dict(zip(sorted(per_ctx), v))} for k, v in cosines.items()}
    print(f'cos(u_M, u_F) median = {cos_summary["cos_u_M_u_F"]["median"]:.4f};  '
          f'cos(u_I, u_F) median = {cos_summary["cos_u_I_u_F"]["median"]:.4f}')

    # ------------------------------------------------------------------ realised readout
    def j2_total(prompt, interventions):
        vals = [j2_value(tok, model, lm, p50, norms, prompt, f['template'], f,
                         interventions or None, dev, a.max_seq) for f in fams.values()]
        return sum(vals) / len(vals)

    rec = {'schema': 'qwentest-dose-f-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'layers': layers,
           'k': K, 'budgets': budgets, 'arms': arms, 'g_from': str(Path(a.g_from).name),
           'lexicon_sha256': src['lexicon_sha256'], 'pool_variant': a.pool_variant,
           'coordinates': [{'layer': l, 'neuron': n} for l, n in coords],
           'fisher': {'source_text': 'engine.REFERENCE_TEXT',
                      'source_text_sha256': hashlib.sha256(REFERENCE_TEXT.encode()).hexdigest(),
                      'ref_tokens': len(ref_ids), 'samples': a.samples,
                      'sample_tokens': a.sample_tokens, 'counted_positions': n_tok,
                      'diag': [float(x) for x in F], 'lambda_median': lam,
                      'diag_ratio_max_over_min': float(F.max() / F.min()),
                      'split_half': {'pearson_diag_raw': diag_half_r,
                                     'pearson_diag_sqrt': diag_half_r_sqrt,
                                     'per_context': splits}},
           'cosines': cos_summary, 'lambda_profile': lam_profile,
           'frozen_criteria': {
               'primary': 'dJ2(u_F) > dJ2(u_M) in >=7/8 contexts on the per-context mean over '
                          'budgets; exact two-sided sign test; median relative gain > 0',
               'secondary': 'u_F vs u_I, and u_M vs u_I (replication of plan 2.16)',
               'control': 'u_R must not match u_F; cos(u_F^A,u_F^B) reported before any win',
               'unit': 'context (budgets are nested scalings, reported as a profile)'},
           'per_context': {}}

    for cid in sorted(per_ctx):
        r = per_ctx[cid]
        g = torch.tensor(r['g'], dtype=torch.float64)
        prompt = by_id[cid]['prompt']
        wR = torch.tensor([rng.gauss(0, 1) for _ in range(K)], dtype=torch.float64)
        d, lam_used = dirs_for(g)
        d['u_R'] = wR

        base = j2_total(prompt, [])
        prog.step(f'{cid} BASE J2={base:+.4f}')
        entry = {'baseline_J2': base, 'abs_g': float(g.norm()), 'lambda_used': lam_used,
                 'predicted_per_budget': {}, 'realized': {}, 'directions': {}}
        for b in budgets:
            for arm in arms:
                u = to_budget(d[arm], b)
                iv = [{'layer': coords[i][0], 'neuron': coords[i][1], 'dose': float(u[i])}
                      for i in range(K) if abs(float(u[i])) > 1e-15]
                val = j2_total(prompt, iv)
                entry['realized'].setdefault(arm, {})[str(b)] = val - base
                entry['predicted_per_budget'].setdefault(arm, {})[str(b)] = float(g @ u)
                entry['directions'][arm] = {'budget_actual': budget_of(u), 'n_nonzero': len(iv),
                                            'dose_abs_max': max(abs(float(u[i])) for i in range(K))}
                prog.step(f'{cid} b={b} {arm} dJ2={val - base:+.5f} (pred {float(g @ u):+.5f})')
        rec['per_context'][cid] = entry

    # ------------------------------------------------------------------ frozen analysis
    cids = sorted(per_ctx)

    def per_ctx_mean(arm):
        return {c: statistics.fmean(rec['per_context'][c]['realized'][arm].values()) for c in cids}

    an = {}
    for x, y in (('u_F', 'u_M'), ('u_F', 'u_I'), ('u_M', 'u_I'), ('u_F', 'u_R'), ('u_I', 'u_R')):
        mx, my = per_ctx_mean(x), per_ctx_mean(y)
        diffs = [mx[c] - my[c] for c in cids]
        p, n = exact_sign_p(diffs)
        rel = [(mx[c] - my[c]) / abs(my[c]) for c in cids if abs(my[c]) > 1e-12]
        an[f'{x}_vs_{y}'] = {'wins': sum(d > 0 for d in diffs), 'n': n, 'sign_p': p,
                             'median_abs_diff': statistics.median(diffs),
                             'median_relative_gain': statistics.median(rel) if rel else None,
                             'per_context_diff': dict(zip(cids, diffs))}
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
    (out / 'dose_f.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf8')

    lines = ['# Which G: the output-KL Fisher, on plan 2.16\'s 32 coordinates', '',
             f'- machine `{rec["machine"]}` / {rec["gpu"]} · |S| = {K} · contexts {len(cids)} · '
             f'budgets {budgets}',
             f'- support and g from `{Path(a.g_from).name}` (unchanged, so differences are the '
             f'cost matrix alone)',
             f'- Fisher: `engine.REFERENCE_TEXT` ({len(ref_ids)} tokens) + {a.samples} x '
             f'{a.sample_tokens} sampled continuations, {n_tok} counted positions, '
             f'one backward each; diagonal only (CANS\'s own limitation)',
             f'- lambda = median(diag F) = {lam:.6g}; diag ratio max/min = '
             f'{float(F.max() / F.min()):.4g}',
             f'- **split-half reproducibility** r(diag F) = {fmt_r(diag_half_r)} raw / '
             f'{fmt_r(diag_half_r_sqrt)} on sqrt; median cos(u_F^A, u_F^B) = {sp_half:.4f}',
             f'- **cos(u_M, u_F)** median = {cos_summary["cos_u_M_u_F"]["median"]:.4f}; '
             f'cos(u_I, u_F) median = {cos_summary["cos_u_I_u_F"]["median"]:.4f}', '',
             '## cosines and damping sensitivity', '',
             '| comparison | median cos |', '|---|---:|']
    for k, v in cos_summary.items():
        lines.append(f'| `{k}` | {v["median"]:.4f} |')
    lines += ['', '| lambda / median(F) | cos vs primary lambda | cos vs u_M | cos vs u_I |',
              '|---:|---:|---:|---:|']
    for k, v in lam_profile.items():
        lines.append(f'| {k} | {v["median_cos_vs_primary_lambda"]:.4f} | '
                     f'{v["median_cos_vs_u_M"]:.4f} | {v["median_cos_vs_u_I"]:.4f} |')
    lines += ['', '## realized dJ2 per context (mean over budgets)', '',
              '| context | BASE J2 | u_I | u_M | u_F | u_R |', '|---|---:|---:|---:|---:|---:|']
    for c in cids:
        e = rec['per_context'][c]
        m = {arm: statistics.fmean(e['realized'][arm].values()) for arm in arms}
        lines.append(f'| `{c}` | {e["baseline_J2"]:+.4f} | ' + ' | '.join(
            f'{m[arm]:+.5f}' for arm in arms) + ' |')
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
    (out / 'dose_f.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/dose_f.json and .md')


if __name__ == '__main__':
    main()
