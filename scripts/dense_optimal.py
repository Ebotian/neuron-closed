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
