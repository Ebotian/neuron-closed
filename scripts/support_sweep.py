#!/usr/bin/env python3
"""How many coordinates should the answer contain? Realized dJ2 vs support size K.

Plan §2.18 restored one positive finding: at a fixed physical budget, injecting 32
coordinates beats injecting the single best one (mean dJ2 +0.056177 bf16 / +0.057607 fp32
vs +0.036488 / +0.037018, gap +0.0197 established to 4.4%, dense > single in 8/8 contexts
under BOTH precisions). It also killed the intermediate step: 1 -> 2 coordinates (+0.00525)
is below the single-measurement bf16 error (median 2.0e-03, worst 6.5e-03), so "denser is
monotonically better" was never established.

This measures the whole curve instead of two points, so the question becomes answerable:

    at a fixed physical budget, does realized dJ2 keep growing with support size K,
    where does it stop, and does the linear surrogate predict that?

It also answers the design question the output type depends on: how many coordinates are
worth naming. The marginal gain per coordinate is compared against the measured error scale
from §2.18, so "stop at K" is a numerical statement, not a taste judgement.

Budget, exactly as in §2.16.1: steering adds alpha_n = u_n * p50_l / ||W_down[:,n]|| at
every position and down_proj is the MLP's last linear map, so the residual perturbation at
layer l is d_l = sum_n u_n * What[:,n] with What the UNIT-normalised columns, and

    budget(u) = sqrt(sum_l ||d_l||^2)

which equals |dose| for a single coordinate. No Gram matrix is formed: d_l is a matvec, so
this stays cheap at K = 18432.

Supports are NESTED (top-K by |g|, the same ranking `engine.ranked_candidates` uses), so
the curve is a single ordering read at increasing depth, not K independent selections.

Error discipline (§2.18.4): differences below 1e-02 are not claimed from bf16 alone;
pass --fp32 to re-run the whole sweep in float32 with p50, the column norms and the
direction g held at their bf16 values, so the only change is readout numerics.

GPU: per context, one grad-enabled pass per family (for g), then |K| x |budgets| J2
evaluations, each 2 batched forwards. Steering uses the frozen `engine.apply_steering`.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--layers', default='13-14')
    ap.add_argument('--ks', default='1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,18432')
    ap.add_argument('--budgets', default='0.05,0.2,0.5')
    ap.add_argument('--contexts', type=int, default=8)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--fp32', action='store_true', help='readout forward in float32')
    ap.add_argument('--record-top', type=int, default=64,
                    help='record this many top-|g| coordinate identities per context, so the result '
                         'can be expressed as neuron + sign + strength (0 disables)')
    ap.add_argument('--out', default='out/support_sweep')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
    ks = [int(x) for x in a.ks.split(',')]
    budgets = [float(x) for x in a.budgets.split(',')]
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    ctx = [r for r in rows if r.get('split') == a.split][:a.contexts]
    if not ctx:
        raise SystemExit(f'no {a.split} contexts in {a.probes}')

    n_meas = len(ctx) * len(ks) * len(budgets)
    print('Question : at a fixed physical budget, how does realized dJ2 depend on support size K?')
    print(f'Readout  : J2 = mean over families {sorted(fams)}, pool variant {a.pool_variant}')
    print(f'Layers   : {layers}   precision: {"float32" if a.fp32 else "bfloat16 (production)"}')
    print(f'Supports : {ks} (nested top-K by |g|, the ranking engine.ranked_candidates uses)')
    print(f'Budgets  : {budgets}   budget(u) = sqrt(sum_l ||What_l u_l||^2), = |dose| for one coordinate')
    print(f'Contexts : {len(ctx)} -> {[c["id"] for c in ctx]}')
    print(f'Work     : {len(ctx)} grad passes (x{len(fams)} families) + {n_meas} J2 evaluations '
          f'= {n_meas * len(fams)} batched forwards')
    print('Frozen   : report the whole curve, argmax K, and marginal gain per coordinate against the')
    print('           §2.18 error scale (single measurement 2.0e-03 median / 6.5e-03 worst;')
    print('           mean over 8 contexts ~9e-04). Differences below 1e-02 need --fp32 to be claimed.')
    if a.dry_run:
        print(f'[dry-run] would write {a.out}/support_sweep.json and .md')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import (apply_steering, column_norms, estimate_p50, model_lm, score_for_dose)
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

    # unit-normalised down_proj columns, per layer: budget(u) needs no Gram matrix
    What = {l: (lm.layers[l].mlp.down_proj.weight.detach().float()
                .T) for l in layers}                       # 9216 x 2560
    What = {l: W / W.norm(dim=1, keepdim=True).clamp(min=1e-12) for l, W in What.items()}
    dim = len(layers) * len(norms[layers[0]])
    if max(ks) > dim:
        raise SystemExit(f'K={max(ks)} exceeds the coordinate space {dim}')

    def budget(u_by_layer):
        # What[l] is 9216 x 2560 with UNIT-normalised rows (= the down_proj columns),
        # so the residual perturbation is d_l = u_l @ What[l]  (a 2560-vector).
        return math.sqrt(sum(
            float((u @ What[l].to(device=u.device, dtype=u.dtype)).pow(2).sum())
            for l, u in u_by_layer.items()))

    prog = Progress(len(ctx) * len(fams) + n_meas)
    rec = {'schema': 'qwentest-support-sweep-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'precision': str(dtype),
           'layers': layers, 'ks': ks, 'budgets': budgets, 'pool_variant': a.pool_variant,
           'lexicon_sha256': hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest(),
           'p50': {str(l): p50[l] for l in layers}, 'contexts': [c['id'] for c in ctx],
           'coordinate_space': dim, 'per_context': {}}

    for c in ctx:
        # full-dimensional g in dose units, averaged over families exactly as scan_sem does
        gsum, jfam = None, {}
        for fam, f in sorted(fams.items()):
            j2, g = sem_gradient(tok, model, lm, layers, c['prompt'], f['template'], f,
                                 dev, a.max_seq)
            jfam[fam] = j2
            sc = {l: score_for_dose(g[l].float(), p50[l], norms[l].to(g[l].device)) for l in layers}
            gsum = sc if gsum is None else {l: gsum[l] + sc[l] for l in layers}
            prog.step(f'{c["id"]} g {fam} J2={j2:+.4f}')
        g = {l: (gsum[l] / len(fams)).double().cpu() for l in layers}
        flat = torch.cat([g[l] for l in layers])
        order = torch.argsort(flat.abs(), descending=True)
        per_layer_n = {l: len(g[l]) for l in layers}

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
        entry = {'baseline_J2': base, 'J2_by_family': jfam,
                 'abs_g_total': float(flat.norm()),
                 'abs_g_top1': float(flat.abs()[order[0]]),
                 'curve': {}}
        # coordinate identities, so "neuron + sign + strength" is expressible and the
        # cumulative share of the gradient's squared mass is auditable per prefix
        if a.record_top:
            npl = per_layer_n[layers[0]]
            g2 = float((flat ** 2).sum()) or 1.0
            cum = 0.0
            top = []
            for r, i in enumerate(order[:a.record_top].tolist(), 1):
                l = layers[i // npl]
                n = i % npl
                v = float(flat[i])
                cum += v * v
                top.append({'rank': r, 'layer': int(l), 'neuron': int(n), 'g': v,
                            'sign': 1 if v >= 0 else -1,
                            'cum_share_of_g_squared': cum / g2})
            entry['top_coordinates'] = top
            entry['known_evil_ranks'] = {
                f'L{l}N{n}': next((t['rank'] for t in top
                                   if t['layer'] == l and t['neuron'] == n), None)
                for l, n in ((14, 8949), (13, 7956))}
        for b in budgets:
            for K in ks:
                idx = order[:K]
                u_by_layer, sel_by_layer = {}, {}
                off = 0
                for l in layers:
                    n = per_layer_n[l]
                    m = (idx >= off) & (idx < off + n)
                    sel = idx[m] - off
                    u = torch.zeros(n, dtype=torch.float64)
                    u[sel] = flat[idx[m]]
                    u_by_layer[l] = u
                    sel_by_layer[l] = sel
                    off += n
                bu = budget({l: u.to(dev) for l, u in u_by_layer.items()})
                if bu < 1e-12:
                    raise SystemExit(f'degenerate direction at K={K}')
                scale = b / bu
                iv = [{'layer': l, 'neuron': int(i), 'dose': float(u_by_layer[l][i]) * scale}
                      for l in layers for i in sel_by_layer[l]
                      if abs(float(u_by_layer[l][i]) * scale) > 1e-15]
                val = j2_total(iv)
                pred = float(sum(float(u_by_layer[l] @ g[l]) for l in layers) * scale)
                entry['curve'].setdefault(str(b), {})[str(K)] = {
                    'dJ2': val - base, 'predicted': pred, 'n_nonzero': len(iv),
                    'budget_actual': float(bu * scale),
                    'dose_abs_max': max((abs(x['dose']) for x in iv), default=0.0)}
                prog.step(f'{c["id"]} b={b} K={K} dJ2={val - base:+.5f} pred={pred:+.5f} '
                          f'({len(iv)} coords)')
        rec['per_context'][c['id']] = entry
        del batches
        torch.cuda.empty_cache()

    # ---------------- analysis ----------------
    cids = rec['contexts']
    mean_curve = {str(b): {str(K): statistics.fmean(
        rec['per_context'][c]['curve'][str(b)][str(K)]['dJ2'] for c in cids)
        for K in ks} for b in budgets}
    mean_pred = {str(b): {str(K): statistics.fmean(
        rec['per_context'][c]['curve'][str(b)][str(K)]['predicted'] for c in cids)
        for K in ks} for b in budgets}
    wins = {str(b): {str(K): sum(1 for c in cids if
                   rec['per_context'][c]['curve'][str(b)][str(K)]['dJ2']
                   > rec['per_context'][c]['curve'][str(b)]['1']['dJ2'])
               for K in ks} for b in budgets}
    rec['mean_curve'] = mean_curve
    rec['mean_predicted'] = mean_pred
    rec['beats_K1_in_n_contexts'] = wins
    argmax = {str(b): max(ks, key=lambda K: mean_curve[str(b)][str(K)]) for b in budgets}
    rec['argmax_K'] = argmax

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'support_sweep.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                            encoding='utf8')

    lines = ['# Realized dJ2 vs support size K, at fixed physical budget', '',
             f'- machine `{rec["machine"]}` / {rec["gpu"]} · precision `{rec["precision"]}` · '
             f'layers {layers} · coordinate space {dim} · contexts {len(cids)}',
             f'- budget(u) = sqrt(sum_l ||What_l u_l||^2); supports are nested top-K by |g|',
             f'- §2.18 error scale: single measurement 2.0e-03 median / 6.5e-03 worst; '
             f'mean over {len(cids)} contexts ~9e-04', '',
             '## mean realized dJ2 over contexts (rows = K, cols = budget)', '',
             '| K | ' + ' | '.join(f'b={b}' for b in budgets) + ' | beats K=1 in n contexts (b='
             + str(budgets[0]) + ') |',
             '|---:|' + '---:|' * len(budgets) + '---:|']
    for K in ks:
        lines.append(f'| {K} | ' + ' | '.join(f'{mean_curve[str(b)][str(K)]:+.5f}' for b in budgets)
                     + f' | {wins[str(budgets[0])][str(K)]}/{len(cids)} |')
    lines += ['', '## marginal gain per doubling of K (b=' + str(budgets[0]) + ')', '',
              '| K | mean dJ2 | gain over previous K | gain per coordinate | surrogate predicted |',
              '|---:|---:|---:|---:|---:|']
    prev = None
    for K in ks:
        v = mean_curve[str(budgets[0])][str(K)]
        g_ = '' if prev is None else f'{v - prev:+.5f}'
        pc = '' if prev is None else f'{(v - prev) / max(K - prev_K, 1):+.2e}'
        lines.append(f'| {K} | {v:+.5f} | {g_} | {pc} | {mean_pred[str(budgets[0])][str(K)]:+.5f} |')
        prev, prev_K = v, K
    lines += ['', 'argmax K per budget: ' + ', '.join(f'b={b} -> {argmax[str(b)]}' for b in budgets),
              '', f'K=1 mean dJ2 (b={budgets[0]}): {mean_curve[str(budgets[0])]["1"]:+.5f}; '
              f'K={max(ks)} mean: {mean_curve[str(budgets[0])][str(max(ks))]:+.5f}']
    (out / 'support_sweep.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/support_sweep.json and .md')


if __name__ == '__main__':
    main()
