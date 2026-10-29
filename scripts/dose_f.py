#!/usr/bin/env python3
"""Item 4 / plan 2.34: the THIRD answer to "which G" --- the Fisher of an output-KL
constraint --- measured on plan 2.16's 32-coordinate support; and plan 3 R8: the same
three-way comparison on the FULL 18,432-coordinate support (`--dense`).

Why this is the one missing dimension. Every arm in this repo so far buys the same
thing: a claim about how expensive a perturbation is. Plan 2.16 priced a perturbation by
the residual displacement it produces (M = Gram of unit-normalised W_down columns). CANS
(arXiv 2610.35530 v1, read in 13_paper) prices it instead by the KL it opens between the
steered and the base OUTPUT distribution, whose curvature is the Fisher information:

    cost(u) = 1/2 u' F u  (+ lambda/2 ||u||^2),    F_ij = E[ dlogp/du_i * dlogp/du_j ]

so per-coordinate (diagonal) F_ii = E[ (d log p / du_i)^2 ]. The missing dimension is not
another readout or another layer --- it is the cost matrix itself.

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
ONE estimator serves both supports: it always produces the full 9,216-dimensional per-layer
diagonal, and the 32-coordinate arm reads 32 entries out of it. That is why the two arms of
this file cannot drift apart.

SPLIT-HALF REPRODUCIBILITY (the G1/G2 discipline applied to F). Counted tokens are split
odd/even; each half gets its own F, its own lambda = median(F_half) and its own direction.
If cos(u_F^A, u_F^B) is low, u_F is an estimator artifact and the three-way comparison is
void --- that is reported BEFORE any realised number is interpreted.

ARMS, all on ONE support, so any difference is the cost matrix alone:
    u_I  ~ g                            (G = I, current practice; `u_g` in --dense)
    u_M  ~ M^-1 g                       (G = the budget metric; plan 2.16's favourite)
    u_F  ~ (diag(F) + lambda I)^-1 g     (G = Fisher of the output-KL constraint; NEW)
    u_R  ~ random direction              (control: is any direction as good?)

TWO MODES (the support is the variable that plan 2.32 showed is first-order).
  default (plan 2.34): S = the 32 coordinates of plan 2.16, g supplied by `--g-from`
      (`g_j2_space_k32_m192_c8.json`), M = the 32x32 Gram block, u_M = M^-1 g.
  `--dense` (plan 3 R8): S = ALL 2 x 9,216 coordinates of layers 13-14, g recomputed here
      per context (sem_gradient), and M is SINGULAR (rank <= 2,560 per layer), so u_M uses
      the plan 2.21 closed form u = What B^-1 d*, d* = B^-1 e, e = What' g, B = What'What;
      budget(u) = sqrt(sum_l ||What_l u_l||^2) = sqrt(u'Mu) exactly. J2 readout only ---
      `engine.apply_steering` cannot drive generation at that width (plan 2.27).
      The hypothesis for --dense is registered in plan 3 R8: cond(diag F + lambda I) must be
      clearly larger than the 1.995 measured on 32 coordinates, otherwise "the price tags
      differ" is not a measurable statement on this model at all.

FROZEN criteria (written before the run; do not tune after seeing numbers).
  Unit of analysis = CONTEXT. Budgets are nested scalings of one direction (correlated),
  so they are a profile; the primary statistic pools them per context first.
  PRIMARY    realised dJ2(u_F) > realised dJ2(u_M) in >= 7/8 contexts on the per-context
             mean over budgets; exact two-sided sign test; median relative gain > 0.
  SECONDARY  the same paired test for u_F vs u_I, and u_M vs u_I (in --dense the latter
             should reproduce plan 2.22's 36-43% gain; if it does not, this run's instrument
             is not comparable and nothing else here may be quoted).
  ERROR      §2.18.4: a median difference below 1e-02 is NOT claimed from bf16 alone; it
             needs an fp32 replication of the same arm pair. That replication is deliberately
             NOT implemented in this file -- if the primary difference lands below 1e-02, add
             it (dense_optimal.py's `--fp32` is the precedent) BEFORE writing any claim.
  CONTROL    u_R must NOT match u_F; and cos(u_F^A,u_F^B) must be reported before any win.
  GUARD      report the full budget profile, never the best budget.

GPU: 4 forwards + ~1 backward per counted token for F (~40 s), plus
len(contexts) x (1 + 4 arms x |budgets|) J2 evaluations. In --dense every arm perturbs
18,432 coordinates, so `apply_steering`'s per-coordinate Python hook is the slow part.

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
    """Two-sided exact binomial sign test; zeros excluded. No iid claim beyond the unit used."""
    v = [d for d in diffs if abs(d) > 1e-12]
    n = len(v)
    if n == 0:
        return None, 0
    k = min(sum(x > 0 for x in v), sum(x < 0 for x in v))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), n


def safe_r(x, y):
    """statistics.correlation raises on a constant vector; a degenerate half is a result."""
    try:
        return statistics.correlation(x, y)
    except statistics.StatisticsError:
        return None


def fmt_r(v):
    return 'n/a' if v is None else f'{v:.4f}'


def fisher_diag(a, tok, model, lm, layers, p50, norms, dev, progress, torch, REFERENCE_TEXT):
    """Diagonal empirical Fisher of the output-KL constraint, in DOSE units, per layer.

    Returns (F, halves, meta): F is {layer: tensor(9216) float64}; halves are two such dicts
    from the odd/even split of the counted tokens. Nothing here is support-specific: the
    caller restricts to its own coordinates, so the 32-coordinate and the dense arms read the
    SAME numbers.
    """
    from engine import capture_mlp_inputs

    torch.manual_seed(a.seed)
    ref_ids = tok(REFERENCE_TEXT, return_tensors='pt').input_ids[0].tolist()[:a.ref_tokens]
    seqs = []
    for _ in range(a.samples):
        gen = model.generate(
            input_ids=torch.tensor([ref_ids], device=dev),
            attention_mask=torch.ones(1, len(ref_ids), device=dev, dtype=torch.long),
            max_new_tokens=a.sample_tokens, do_sample=True, temperature=1.0, top_p=1.0,
            top_k=0, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        seqs.append(gen[0].tolist())
    counted = [(s, t) for s, seq in enumerate(seqs)
               for t in range(1, len(seq))
               if (s == 0 and t < len(ref_ids)) or t >= len(ref_ids)]
    by_seq = {}
    for s, t in counted:
        by_seq.setdefault(s, []).append(t)

    dims = {l: int(norms[l].numel()) for l in layers}
    F = {l: torch.zeros(dims[l], dtype=torch.float64) for l in layers}
    Fh = [{l: torch.zeros(dims[l], dtype=torch.float64) for l in layers} for _ in range(2)]
    nh = [0, 0]

    for s, targets in by_seq.items():
        seq = torch.tensor([seqs[s]], device=dev, dtype=torch.long)
        with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
            out = model(input_ids=seq, use_cache=False)
            lsm = out.logits[0].float().log_softmax(-1)
            for n_t, t in enumerate(targets):
                lp = lsm[t - 1, seqs[s][t]]
                grads = torch.autograd.grad(lp, [cap[l] for l in layers],
                                            retain_graph=(n_t < len(targets) - 1))
                h = t % 2
                for l, gr in zip(layers, grads):
                    sc = gr.detach().double().sum(dim=tuple(range(gr.ndim - 1)))
                    # dose units: alpha = dose * p50 / ||W_down[:, n]|| (engine.apply_steering)
                    sc = sc * (p50[l] / norms[l].to(sc.device).double().clamp(min=1e-12))
                    sq = (sc * sc).cpu()      # accumulate on CPU: 18,432 doubles, no GPU need
                    F[l] += sq
                    Fh[h][l] += sq
                nh[h] += 1
                progress.step(f'Fisher token {s}:{t}')
            del out, lsm
        torch.cuda.empty_cache()

    n_tok = len(counted)
    F = {l: v / n_tok for l, v in F.items()}
    Fh = [{l: v / max(1, n) for l, v in d.items()} for d, n in zip(Fh, nh)]
    gen_ids = [seq[len(ref_ids):] for seq in seqs]
    meta = {'ref_tokens': len(ref_ids), 'samples': a.samples, 'sample_tokens': a.sample_tokens,
            'counted_positions': n_tok, 'per_layer_dims': dims,
            'text_sha256': hashlib.sha256(REFERENCE_TEXT.encode()).hexdigest(),
            'sampled_ids_sha256': hashlib.sha256(
                json.dumps(gen_ids).encode()).hexdigest(),
            'sampled_ids': gen_ids}
    return F, Fh, meta


def cos(x, y):
    return float(x @ y / (x.norm() * y.norm() + 1e-300))


def report_tail(a, rec, arms, budgets, cids, an, cos_summary, header, out, extra_lines, torch):
    """Shared markdown rendering for both supports."""
    lines = header + ['', '## cosines and damping sensitivity', '',
                      '| comparison | median cos |', '|---|---:|']
    for k, v in cos_summary.items():
        lines.append(f'| `{k}` | {v["median"]:.4f} |')
    lines += ['', '| lambda / median(F) | cos vs primary lambda | cos vs u_M | cos vs u_I |',
              '|---:|---:|---:|---:|']
    for k, v in rec['lambda_profile'].items():
        lines.append(f'| {k} | {v["median_cos_vs_primary_lambda"]:.4f} | '
                     f'{v["median_cos_vs_u_M"]:.4f} | {v["median_cos_vs_u_I"]:.4f} |')
    short = {'u_I': 'u_I', 'u_g': 'u_g', 'u_M': 'u_M', 'u_F': 'u_F', 'u_R': 'u_R'}
    lines += ['', '## realized dJ2 per context (mean over budgets)', '',
              '| context | BASE J2 | ' + ' | '.join(short[x] for x in arms) + ' |',
              '|---|---:|' + '---:|' * len(arms)]
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
        lines.append(f'| `{arm}` | ' + ' | '.join('n/a' if c is None else f'{c:.3f}'
                                                  for c in cells) + ' |')
    lines += extra_lines
    name = 'dose_f_dense' if a.dense else 'dose_f'
    (out / f'{name}.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf8')
    (out / f'{name}.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/{name}.json and .md')


def paired(x, y, cids, per_ctx_mean, exact_sign_p, statistics):
    mx, my = per_ctx_mean[x], per_ctx_mean[y]
    diffs = [mx[c] - my[c] for c in cids]
    p, n = exact_sign_p(diffs)
    rel = [(mx[c] - my[c]) / abs(my[c]) for c in cids if abs(my[c]) > 1e-12]
    return {'wins': sum(d > 0 for d in diffs), 'n': n, 'sign_p': p,
            'median_abs_diff': statistics.median(diffs),
            'median_relative_gain': statistics.median(rel) if rel else None,
            'per_context_diff': dict(zip(cids, diffs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--g-from', default=str(ROOT / 'results/g_j2_space_k32_m192_c8.json'),
                    help='32-coordinate mode only: g_j2_space.py result JSON supplying the '
                         'support S of plan 2.16 and the per-context gradient g')
    ap.add_argument('--dense', action='store_true',
                    help='plan 3 R8: full-support arm (all 9,216 coordinates per layer), g '
                         'recomputed here, u_M via the plan 2.21 pseudo-inverse closed form')
    ap.add_argument('--layers', default='13-14', help='--dense only')
    ap.add_argument('--contexts', type=int, default=8, help='--dense only')
    ap.add_argument('--split', default='heldout', help='--dense only')
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
    ap.add_argument('--out', default='')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    budgets = [float(x) for x in a.budgets.split(',')]
    lam_scales = [float(x) for x in a.lambda_scale.split(',')]
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    if not a.out:
        a.out = 'out/dose_f_dense' if a.dense else 'out/dose_f'

    src = None
    if not a.dense:
        src = json.loads(Path(a.g_from).read_text(encoding='utf8'))
        coords = [(x['layer'], x['neuron']) for x in src['coordinates']]
        K = len(coords)
        layers = src['layers']
        per_ctx = {cid: r for cid, r in src['per_context'].items() if 'g' in r}
        if hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest() != src['lexicon_sha256']:
            raise SystemExit('lexicon differs from the one the support was built with; '
                             'refusing to mix')
    else:
        lo, hi = (int(x) for x in a.layers.split('-'))
        layers = list(range(lo, hi + 1))
        coords = None                       # built after the model is loaded (needs 9,216)
        K = None
        per_ctx = None

    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines()
            if l.strip()]
    by_id = {r['id']: r for r in rows}
    if a.dense:
        ctx = [r for r in rows if r.get('split') == a.split][:a.contexts]
        if not ctx:
            raise SystemExit(f'no {a.split} contexts in {a.probes}')
        cids_declared = [r['id'] for r in ctx]
        # comparability guard: these must be the SAME 8 contexts the 32-coordinate run used
        ctx_check = None
        if Path(a.g_from).is_file():
            sup = json.loads(Path(a.g_from).read_text(encoding='utf8'))['per_context']
            sup_ids = {c for c, r in sup.items() if 'g' in r}
            ctx_check = set(sup_ids) == set(cids_declared)
            if not ctx_check:
                print(f'WARNING: context set differs from {Path(a.g_from).name}: '
                      f'{sorted(sup_ids ^ set(cids_declared))}')
            else:
                print(f'context set = the one in {Path(a.g_from).name} (8 ids match)')
        else:
            print(f'NOTE: {Path(a.g_from).name} absent; context-set check skipped')
    else:
        missing = [c for c in per_ctx if c not in by_id]
        if missing:
            raise SystemExit(f'contexts not found in probes: {missing}')
        cids_declared = sorted(per_ctx)

    arms = ['u_I', 'u_M', 'u_F', 'u_R'] if not a.dense else ['u_g', 'u_M', 'u_F', 'u_R']
    n_eval = len(cids_declared) * (1 + len(arms) * len(budgets))

    print('Question : does G = diag Fisher of an output-KL constraint beat G = M on the model?')
    print(f'Readout  : J2 = mean over families {sorted(fams)} (pool variant {a.pool_variant}), '
          f'precision bfloat16 (production)')
    if a.dense:
        print(f'Support  : DENSE, all coordinates of layers {layers} '
              f'(plan 3 R8); g recomputed per context; u_M via What B^-1 d* (M is singular)')
    else:
        print(f'Support  : the SAME |S| = {K} coordinates and the same g as '
              f'{Path(a.g_from).name} (plan 2.16), layers {layers}')
    print(f'Contexts : {len(cids_declared)} -> {cids_declared}')
    print(f'Budgets  : {budgets}   (budget = sqrt(u\'Mu); equals |dose| for a single coordinate)')
    print(f'Fisher   : engine.REFERENCE_TEXT (all positions, capped at {a.ref_tokens}) + '
          f'{a.samples} x {a.sample_tokens} sampled continuations, T=1/top-p=1; '
          f'one backward per counted token')
    print(f'Damping  : lambda = median(diag F) [paper default]; sensitivity at '
          f'{lam_scales} x median')
    print(f'Arms     : {arms}')
    print(f'Frozen   : PRIMARY = dJ2(u_F) > dJ2(u_M) in >=7/{len(cids_declared)} contexts on the '
          f'per-context mean over budgets, exact two-sided sign test, median relative gain > 0')
    print(f'           ERROR: a median difference below 1e-02 needs an fp32 replication (not '
          f'implemented here; do not claim)')
    print(f'           CONTROL u_R must not match u_F; cos(u_F^A,u_F^B) reported first')
    print(f'Work     : ~{a.ref_tokens + a.samples * a.sample_tokens} backwards for F + '
          f'{n_eval} J2 evaluations = {n_eval * len(fams)} batched forwards')
    if a.dry_run:
        if not a.dense:
            for cid in cids_declared:
                r = per_ctx[cid]
                print(f'  {cid}: |g|={r["J2_gradient_abs_norm"]:.4f} '
                      f'cond(Ghat)={r["condition_number"]:.3g}')
        else:
            for cid in cids_declared:
                print(f'  {cid}: prompt {len(by_id[cid]["prompt"])} chars')
        print(f'[dry-run] would write {a.out}/dose_f{"_dense" if a.dense else ""}.json and .md')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import (REFERENCE_TEXT, apply_steering, column_norms, estimate_p50,
                        load_model, score_for_dose)
    from progress import Progress
    from semantic import j2_value

    tok, model, lm = load_model(a.model)
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device='cuda:0')
    dev = 'cuda:0'

    per_layer_dims = {l: int(norms[l].numel()) for l in layers}
    if a.dense:
        coords = [(l, n) for l in layers for n in range(per_layer_dims[l])]
        K = len(coords)
        per_ctx = None
    else:
        if max(abs(src['p50'][str(l)] - p50[l]) / p50[l] for l in layers) > 0.01:
            raise SystemExit('p50 differs from the support run by >1%; budgets would not be '
                             'comparable')
        bad = [(l, n) for l, n in coords if not (0 <= n < per_layer_dims[l])]
        if bad:
            raise SystemExit(f'coordinates outside this model\'s intermediate width: {bad[:5]}')
    idx_by_layer = {l: [i for i, (ll, _) in enumerate(coords) if ll == l] for l in layers}

    # ---- W / M: unit-normalised W_down columns; M block-diagonal by layer (plan 2.16) ----
    # Direction algebra runs on CPU: the gradients/ g are small, and keeping one device avoids
    # silent cuda/cpu mismatches. The MODEL stays on the GPU; only the readout touches it.
    W = {l: lm.layers[l].mlp.down_proj.weight.detach().double().T.cpu() for l in layers}
    W = {l: W[l] / W[l].norm(dim=1, keepdim=True).clamp(min=1e-12) for l in layers}  # 9216 x 2560
    Bm = {l: W[l].T @ W[l] for l in layers}                                          # 2560 x 2560
    # In the 32-coordinate mode each direction lives on the support only, so the budget uses
    # the corresponding ROWS of What; that is u'Mu exactly, with M the 32x32 Gram block.
    # NOTE: select rows by NEURON id, not by the coordinate's index in `coords`. Using the
    # coordinate index silently picks neurons 0..|S|-1 instead of the support, which changes
    # every budget by 3-13% (caught by the regression run against the committed plan-2.34
    # result, 2026-10-31; in --dense mode the two coincide, which is why only that arm was
    # unaffected).
    Wsel = {l: (W[l] if a.dense else
                W[l][[coords[i][1] for i in idx_by_layer[l]]].contiguous()) for l in layers}
    pos = {i: j for l in layers for j, i in enumerate(idx_by_layer[l])}   # coord -> slot in layer
    Mmat = None
    if not a.dense:
        Mmat = torch.zeros(K, K, dtype=torch.float64)
        for x, (lx, nx) in enumerate(coords):
            for y, (ly, ny) in enumerate(coords):
                if lx == ly:
                    Mmat[x, y] = W[lx][nx] @ W[ly][ny]

    def budget_of(u):
        # displacement in residual space = sum_n u_n * What[n, :] = What' u  (a 2560-vector)
        return float(torch.sqrt(sum((Wsel[l].T @ u[l]).pow(2).sum() for l in layers)))

    def to_budget(u, b):
        n = budget_of(u)
        if n < 1e-12:
            raise SystemExit('degenerate direction')
        s = b / n
        return {l: v * s for l, v in u.items()}

    prog = Progress(len(cids_declared) * (len(fams) if a.dense else 0)
                    + a.ref_tokens + a.samples * a.sample_tokens + n_eval)

    # ---- per-context gradient: from the support run (32) or recomputed here (dense) ----
    G = {}
    if a.dense:
        from semantic import sem_gradient
        for c in ctx:
            gsum = None
            for fam, f in sorted(fams.items()):
                _j2, gg = sem_gradient(tok, model, lm, layers, c['prompt'], f['template'], f,
                                       dev, a.max_seq)
                sc = {l: score_for_dose(gg[l].float(), p50[l],
                                        norms[l].to(gg[l].device)) for l in layers}
                gsum = sc if gsum is None else {l: gsum[l] + sc[l] for l in layers}
                prog.step(f'{c["id"]} g {fam}')
            G[c['id']] = {l: (gsum[l] / len(fams)).double().cpu() for l in layers}
    else:
        for cid, r in per_ctx.items():
            flat = torch.tensor(r['g'], dtype=torch.float64)
            G[cid] = {l: torch.stack([flat[i] for i in idx_by_layer[l]])
                      if idx_by_layer[l] else torch.zeros(0, dtype=torch.float64)
                      for l in layers}
        for l in layers:
            if idx_by_layer[l]:
                assert G[cids_declared[0]][l].numel() == len(idx_by_layer[l])

    # ---- Fisher (one estimator, both supports) ----
    Ffull, Fhalf, fmeta = fisher_diag(a, tok, model, lm, layers, p50, norms, dev, prog, torch,
                                      REFERENCE_TEXT)
    if a.dense:
        F, Fh = Ffull, Fhalf
        Fflat = torch.cat([F[l] for l in layers])
        Fhflat = [torch.cat([Fh[h][l] for l in layers]) for h in range(2)]
    else:
        Fflat = torch.tensor([float(Ffull[l][n]) for l, n in coords], dtype=torch.float64)
        Fhflat = [torch.tensor([float(Fhalf[h][l][n]) for l, n in coords], dtype=torch.float64)
                  for h in range(2)]
        F = {l: Fflat[[i for i, (ll, _) in enumerate(coords) if ll == l]] for l in layers}
        Fh = [{l: Fhflat[h][[i for i, (ll, _) in enumerate(coords) if ll == l]]
               for l in layers} for h in range(2)]
    lam = float(Fflat.median())
    cond_F = float((Fflat.max() + lam) / (Fflat.min() + lam))
    print(f'Fisher: {fmeta["counted_positions"]} tokens over {Fflat.numel()} coordinates; '
          f'diag range {float(Fflat.min()):.4g}…{float(Fflat.max()):.4g} '
          f'(spread {float(Fflat.max() / Fflat.min()):.3f}x); lambda = median(F) = {lam:.6g}; '
          f'cond(diag F + lambda I) = {cond_F:.3f}')

    def damped(Fd, scale, g):
        """(diag(F) + lam*I)^-1 g per layer, lam = scale * median(diag F)."""
        m = float(torch.cat([Fd[l] for l in layers]).median()) * scale
        return {l: g[l] / (Fd[l] + m) for l in layers}, m

    def u_M_of(g):
        if a.dense:                                     # M singular: plan 2.21 closed form
            out = {}
            for l in layers:
                e = W[l].T @ g[l]
                d_star = torch.linalg.solve(Bm[l], e)
                out[l] = W[l] @ torch.linalg.solve(Bm[l], d_star)
            return out
        # 32-coordinate mode: M is the 32x32 Gram block (SPD, cond 1.43) and the flat g must be
        # assembled in COORDS order, not in layer-blocked order -- mixing the two silently
        # permutes the solution (caught by the regression run against the committed result).
        gflat = torch.tensor([float(g[coords[i][0]][pos[i]]) for i in range(K)],
                             dtype=torch.float64)
        sol = torch.linalg.solve(Mmat, gflat)
        return {l: sol[[i for i in range(K) if coords[i][0] == l]] for l in layers}

    def dirs_for(g):
        d = {arms[0]: g, 'u_M': u_M_of(g)}
        d['u_F'], lam_used = damped(F, 1.0, g)
        return d, lam_used

    def flat(d):
        return torch.cat([d[l] for l in layers])

    def cossum(pair):
        x, y = pair
        return cos(flat(x), flat(y))

    # ---- split-half reproducibility of F itself, and of the direction it implies ----
    splits, cos_summary = [], {'cos_u_M_u_F': [], 'cos_u_I_u_F': [], 'cos_u_I_u_M': []}
    for cid in cids_declared:
        g = G[cid]
        uA, lamA = damped(Fh[0], 1.0, g)
        uB, lamB = damped(Fh[1], 1.0, g)
        uFull, _ = damped(F, 1.0, g)
        splits.append({'context': cid, 'cos_u_F_halfA_halfB': cossum([uA, uB]),
                       'cos_u_F_full_halfA': cossum([uFull, uA]),
                       'lambda_A': lamA, 'lambda_B': lamB})
        d, _ = dirs_for(g)
        cos_summary['cos_u_M_u_F'].append(cossum([d['u_M'], d['u_F']]))
        cos_summary['cos_u_I_u_F'].append(cossum([d[arms[0]], d['u_F']]))
        cos_summary['cos_u_I_u_M'].append(cossum([d[arms[0]], d['u_M']]))
    diag_half_r = safe_r([float(x) for x in Fhflat[0]], [float(x) for x in Fhflat[1]])
    diag_half_r_sqrt = safe_r([math.sqrt(float(x)) for x in Fhflat[0]],
                              [math.sqrt(float(x)) for x in Fhflat[1]])
    sp_half = statistics.median(x['cos_u_F_halfA_halfB'] for x in splits)
    print(f'split-half: r(diag F) = {fmt_r(diag_half_r)} (raw) / {fmt_r(diag_half_r_sqrt)} '
          f'(sqrt); median cos(u_F^A,u_F^B) = {sp_half:.4f}')
    cos_summary = {k: {'median': statistics.median(v), 'per_context': dict(zip(cids_declared, v))}
                   for k, v in cos_summary.items()}
    print(f'cos(u_M, u_F) median = {cos_summary["cos_u_M_u_F"]["median"]:.4f}; '
          f'cos(u_{"g" if a.dense else "I"}, u_F) median = '
          f'{cos_summary["cos_u_I_u_F"]["median"]:.4f}; '
          f'cos(u_{"g" if a.dense else "I"}, u_M) median = '
          f'{cos_summary["cos_u_I_u_M"]["median"]:.4f}')

    lam_profile = {}
    for s in lam_scales:
        vs_primary, vs_M, vs_I = [], [], []
        for cid in cids_declared:
            g = G[cid]
            uF, _ = damped(F, s, g)
            uF1, _ = damped(F, 1.0, g)
            vs_primary.append(cossum([uF, uF1]))
            vs_M.append(cossum([uF, u_M_of(g)]))
            vs_I.append(cossum([uF, g]))
        lam_profile[str(s)] = {'median_cos_vs_primary_lambda': statistics.median(vs_primary),
                               'median_cos_vs_u_M': statistics.median(vs_M),
                               'median_cos_vs_u_I': statistics.median(vs_I)}

    # ------------------------------------------------------------------ realised readout
    def j2_total(prompt, interventions):
        vals = [j2_value(tok, model, lm, p50, norms, prompt, f['template'], f,
                         interventions or None, dev, a.max_seq) for f in fams.values()]
        return sum(vals) / len(vals)

    rng = random.Random(a.seed)
    rec = {'schema': 'qwentest-dose-f-dense-v1' if a.dense else 'qwentest-dose-f-v1',
           'machine': platform.node(), 'gpu': torch.cuda.get_device_name(0), 'model': a.model,
           'precision': 'bfloat16', 'layers': layers, 'k': K,
           'budgets': budgets, 'arms': arms, 'pool_variant': a.pool_variant,
           'lexicon_sha256': hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest(),
           'coordinates': ([{'layer': l, 'neuron': n} for l, n in coords] if not a.dense
                           else {'mode': 'dense', 'per_layer': per_layer_dims}),
           'fisher': {'source_text': 'engine.REFERENCE_TEXT', **fmeta,
                      'diag': [float(x) for x in Fflat], 'lambda_median': lam,
                      'diag_spread_max_over_min': float(Fflat.max() / Fflat.min()),
                      'cond_diagF_plus_lambda': cond_F,
                      'split_half': {'pearson_diag_raw': diag_half_r,
                                     'pearson_diag_sqrt': diag_half_r_sqrt,
                                     'per_context': splits}},
           'cosines': cos_summary, 'lambda_profile': lam_profile,
           'frozen_criteria': {
               'primary': 'dJ2(u_F) > dJ2(u_M) in >=7/8 contexts on the per-context mean over '
                          'budgets; exact two-sided sign test; median relative gain > 0',
               'secondary': 'u_F vs u_I (u_F vs u_g in --dense) and u_M vs that same arm',
               'error': '§2.18.4: median difference below 1e-02 needs a --fp32 replication',
               'control': 'u_R must not match u_F; cos(u_F^A,u_F^B) reported before any win',
               'unit': 'context (budgets are nested scalings, reported as a profile)'},
           'per_context': {}}
    if not a.dense:
        rec['g_from'] = str(Path(a.g_from).name)
    if a.dense:
        rec['contexts_match_support_run'] = ctx_check
        rec['registered_prediction'] = ('plan 3 R8: cond(diag F + lambda I) must be clearly '
                                        'larger than the 1.995 measured on 32 coordinates')

    for cid in cids_declared:
        g = G[cid]
        prompt = by_id[cid]['prompt']
        d, lam_used = dirs_for(g)
        rnd_len = {l: (per_layer_dims[l] if a.dense else len(idx_by_layer[l])) for l in layers}
        d['u_R'] = {l: torch.tensor([rng.gauss(0, 1) for _ in range(rnd_len[l])],
                                    dtype=torch.float64) for l in layers}

        base = j2_total(prompt, [])
        prog.step(f'{cid} BASE J2={base:+.4f}')
        entry = {'baseline_J2': base, 'abs_g': float(torch.cat([g[l] for l in layers]).norm()),
                 'lambda_used': lam_used, 'predicted_per_budget': {}, 'realized': {},
                 'directions': {}}
        for b in budgets:
            for arm in arms:
                u = to_budget(d[arm], b)
                iv = []
                if a.dense:
                    for l in layers:
                        for n in range(per_layer_dims[l]):
                            dose = float(u[l][n])
                            if abs(dose) > 1e-15:
                                iv.append({'layer': l, 'neuron': n, 'dose': dose})
                else:
                    for l in layers:
                        for j, i in enumerate(idx_by_layer[l]):
                            dose = float(u[l][j])
                            if abs(dose) > 1e-15:
                                iv.append({'layer': coords[i][0], 'neuron': coords[i][1],
                                           'dose': dose})
                val = j2_total(prompt, iv)
                pred = float(sum(float(u[l] @ g[l]) for l in layers))
                entry['realized'].setdefault(arm, {})[str(b)] = val - base
                entry['predicted_per_budget'].setdefault(arm, {})[str(b)] = pred
                entry['directions'][arm] = {
                    'budget_actual': budget_of(u), 'n_nonzero': len(iv),
                    'dose_abs_max': max(abs(float(x)) for l in layers for x in u[l])}
                prog.step(f'{cid} b={b} {arm} dJ2={val - base:+.5f} (pred {pred:+.5f})')
        rec['per_context'][cid] = entry

    # ------------------------------------------------------------------ frozen analysis
    cids = cids_declared
    per_ctx_mean = {arm: {c: statistics.fmean(rec['per_context'][c]['realized'][arm].values())
                          for c in cids} for arm in arms}
    an = {}
    base_arm = arms[0]
    for x, y in (('u_F', 'u_M'), ('u_F', base_arm), ('u_M', base_arm), ('u_F', 'u_R'),
                 (base_arm, 'u_R')):
        an[f'{x}_vs_{y}'] = paired(x, y, cids, per_ctx_mean, exact_sign_p, statistics)
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
    gname = 'u_g' if a.dense else 'u_I'
    header = ['# Which G' + (' on the FULL support: plan 3 R8' if a.dense
                             else ': the output-KL Fisher on plan 2.16\'s 32 coordinates'), '',
              f'- machine `{rec["machine"]}` / {rec["gpu"]} · |S| = {K} · contexts {len(cids)} · '
              f'budgets {budgets} · precision {rec["precision"]}',
              (f'- DENSE support: all coordinates of layers {layers}; g recomputed per context; '
               f'u_M via the plan 2.21 pseudo-inverse closed form' if a.dense else
               f'- support and g from `{Path(a.g_from).name}` (unchanged, so differences are the '
               f'cost matrix alone)'),
              f'- Fisher: `engine.REFERENCE_TEXT` ({fmeta["ref_tokens"]} tokens) + '
              f'{a.samples} x {a.sample_tokens} sampled continuations, '
              f'{fmeta["counted_positions"]} counted positions, one backward each; '
              f'diagonal only (CANS\'s own limitation)',
              f'- lambda = median(diag F) = {lam:.6g}; diag spread max/min = '
              f'{float(Fflat.max() / Fflat.min()):.4g}; **cond(diag F + lambda I) = {cond_F:.3f}**',
              f'- **split-half reproducibility** r(diag F) = {fmt_r(diag_half_r)} raw / '
              f'{fmt_r(diag_half_r_sqrt)} on sqrt; median cos(u_F^A, u_F^B) = {sp_half:.4f}',
              f'- **cos(u_M, u_F)** median = {cos_summary["cos_u_M_u_F"]["median"]:.4f}; '
              f'cos({gname}, u_F) median = {cos_summary["cos_u_I_u_F"]["median"]:.4f}; '
              f'cos({gname}, u_M) median = {cos_summary["cos_u_I_u_M"]["median"]:.4f}']
    extra = []
    if a.dense:
        extra = ['', '## registered prediction (plan 3 R8, written before the run)', '',
                 f'- prediction: `cond(diag F + lambda I)` clearly larger than the **1.995** '
                 f'measured on 32 coordinates',
                 f'- measured here: **{cond_F:.3f}** on {K} coordinates -> '
                 f'{"CONFIRMED" if cond_F > 1.995 * 1.5 else "**NOT** confirmed"}',
                 '', '`u_M` vs `u_g` must reproduce plan 2.22\'s 36–43% gain; if it does not, the '
                 'instrument is not comparable and nothing else in this run may be quoted:',
                 f'- measured: median relative gain '
                 f'{an["u_M_vs_u_g"]["median_relative_gain"]:+.1%}, '
                 f'{an["u_M_vs_u_g"]["wins"]}/{an["u_M_vs_u_g"]["n"]} wins, '
                 f'p = {an["u_M_vs_u_g"]["sign_p"]:.4f}']
    report_tail(a, rec, arms, budgets, cids, an, cos_summary, header, out, extra, torch)


if __name__ == '__main__':
    main()
