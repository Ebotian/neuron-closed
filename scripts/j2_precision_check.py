#!/usr/bin/env python3
"""Measure the bf16 error on the quantity we actually report, at the PRODUCTION batch shape.

Why. Plan §2.17 measured J2's numerical sensitivity by changing the BATCH SHAPE and found
drift up to 1.209e-02 (median error on dJ2 1.764e-03). That is a *proxy* for rounding
magnitude: every production run uses one fixed batch shape, so the question that decides
whether §2.16.5's density steps and plan 06's two lowest dose points survive is narrower --

    at the production shape, how far is the bf16 dJ2 from a higher-precision dJ2?

This script answers it directly: same prompts, same pool, same physical intervention, same
batch shape; only the forward dtype changes (bfloat16 = production, float32 = reference).

Held fixed across the two precisions, deliberately:
  * p50 and the W_down column norms are computed ONCE in bf16 and reused, so `alpha` and
    therefore the injected residual perturbation are bit-identical;
  * the dense direction u is computed ONCE (from the committed Ghat/g) and reused;
  * the batch shape is the production one (adj 38 seqs, verb 28 seqs).
So any difference is readout numerics, not a different experiment.

FROZEN reading (written before the run):
  err = max over contexts/arms of |dJ2_bf16 - dJ2_fp32|, where dJ2 = arm - none.
  err << 5.25e-03 (the §2.16.5 1->2 coordinate step, the smallest downgraded quantity)
        => bf16 is NOT the limiting factor at fixed shape; §2.17's drift was a shape
           artifact and the downgraded findings can be restored.
  err ~  1e-2  => the error bar is real at fixed shape too; the §2.17.4 downgrades stand.
  Report err against every downgraded quantity, not only the smallest.

GPU: 8 contexts x 3 arms x 2 precisions = 48 J2 evaluations = 96 batched forwards,
plus two model loads (bf16 ~8 GB, fp32 ~16 GB; the bf16 model is freed before the fp32 load).

    [i/N] elapsed, eta ...   per J2 evaluation (package progress.Progress)
"""
from __future__ import annotations
import argparse, hashlib, json, platform, statistics, sys
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--g-from', default=str(ROOT / 'results/g_j2_space_k32_m192_c8.json'),
                    help='committed Ghat/g, used only to build the dense direction u ~ g')
    ap.add_argument('--layers', default='13-14')
    ap.add_argument('--budget', type=float, default=0.05,
                    help='physical budget sqrt(u\'Mu); equals |dose| for a single coordinate')
    ap.add_argument('--steer-layer', type=int, default=14)
    ap.add_argument('--steer-neuron', type=int, default=8949)
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--out', default='out/j2_precision')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    gsrc = json.loads(Path(a.g_from).read_text(encoding='utf8'))
    coords = [(x['layer'], x['neuron']) for x in gsrc['coordinates']]
    g_by_ctx = {c: r['g'] for c, r in gsrc['per_context'].items() if 'g' in r}
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    by_id = {r['id']: r for r in rows}
    cids = [c for c in g_by_ctx if c in by_id]
    if not cids:
        raise SystemExit('no context has both a g vector and a probe row')
    if hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest() != gsrc['lexicon_sha256']:
        raise SystemExit('lexicon differs from the one Ghat was built with')

    arms = ['none', 'single', 'dense']
    n_eval = len(cids) * len(arms) * 2
    print('Question : at the PRODUCTION batch shape, how far is bf16 dJ2 from float32 dJ2?')
    print(f'Readout  : J2 = mean over families {sorted(fams)}, pool variant {a.pool_variant}')
    print(f'Arms     : none | single = L{a.steer_layer}N{a.steer_neuron} at dose {a.budget} '
          f'| dense = {len(coords)} coords, u ~ g, budget {a.budget}')
    print(f'Held fixed across precisions: p50 and W_down column norms (computed once in bf16), '
          f'the dense direction u, and the batch shape')
    print(f'Contexts : {len(cids)} -> {cids}')
    print(f'Work     : {n_eval} J2 evaluations = {n_eval * len(fams)} batched forwards + 2 model loads')
    print('Frozen   : err = max |dJ2_bf16 - dJ2_fp32|; compare against the smallest downgraded '
          'quantity, 5.25e-03 (plan 2.16.5, 1->2 coordinates)')
    if a.dry_run:
        for c in cids:
            print(f'  {c}: |g| = {statistics.pstdev(g_by_ctx[c]):.4g} (sd of the g entries)')
        print(f'[dry-run] would write {a.out}/j2_precision.json and .md')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import apply_steering, column_norms, estimate_p50, model_lm
    from progress import Progress
    from semantic import batch_mean_logp, build_batch, language_log_mass

    dev = 'cuda:0'
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)

    def load(dtype):
        m = Qwen3_5ForConditionalGeneration.from_pretrained(
            a.model, dtype=dtype, attn_implementation='sdpa', local_files_only=True).to(dev).eval()
        m.requires_grad_(False)
        lm = model_lm(m)
        if len(lm.layers) != 32:
            raise RuntimeError(f'expected 32 layers, got {len(lm.layers)}')
        return m, lm

    # ---- calibration and the dense direction are computed ONCE, in bf16 ----
    model, lm = load(torch.bfloat16)
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device=dev)
    print('p50 (bf16, used for BOTH precisions):', {l: round(p50[l], 6) for l in layers})
    W = {l: lm.layers[l].mlp.down_proj.weight.detach().float() for l in layers}
    K = len(coords)
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
    dense = {}
    for c in cids:
        g = torch.tensor(g_by_ctx[c], dtype=torch.float64)
        u = g * (a.budget / float(torch.sqrt(torch.clamp(g @ M @ g, min=1e-30))))
        dense[c] = [{'layer': coords[i][0], 'neuron': coords[i][1], 'dose': float(u[i])}
                    for i in range(K) if abs(float(u[i])) > 1e-15]
    single = [{'layer': a.steer_layer, 'neuron': a.steer_neuron, 'dose': a.budget}]
    ivs = {'none': [], 'single': single, 'dense': None}      # None => per-context dense
    del W, model, lm
    torch.cuda.empty_cache()

    prog = Progress(n_eval)
    res = {}
    for prec, dtype in (('bf16', torch.bfloat16), ('fp32', torch.float32)):
        model, lm = load(dtype)
        for c in cids:
            batches = {}
            for fam, f in sorted(fams.items()):
                batches[fam] = build_batch(tok, by_id[c]['prompt'], f['template'], f['pos'],
                                          f['neg'], dev, a.max_seq)
            for arm in arms:
                iv = dense[c] if arm == 'dense' else ivs[arm]
                with apply_steering(lm, iv, p50, norms), torch.inference_mode():
                    vals = []
                    for fam, (ids, pre, suf, kinds) in batches.items():
                        S = batch_mean_logp(model, ids, pre, suf)
                        jp, _ = language_log_mass(S[kinds > 0].detach().float())
                        jn, _ = language_log_mass(S[kinds < 0].detach().float())
                        vals.append(jp - jn)
                res[(prec, c, arm)] = sum(vals) / len(vals)
                prog.step(f'{prec} {c} {arm} J2={res[(prec, c, arm)]:+.6f}')
        del model, lm
        torch.cuda.empty_cache()
        print(f'{prec} pass done; peak GPU mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
        torch.cuda.reset_peak_memory_stats()

    rec = {'schema': 'qwentest-j2-precision-v1', 'machine': platform.node(), 'model': a.model,
           'layers': layers, 'budget': a.budget, 'arms': arms, 'contexts': cids,
           'pool_variant': a.pool_variant, 'g_from': Path(a.g_from).name,
           'lexicon_sha256': gsrc['lexicon_sha256'],
           'p50_bf16_used_for_both': {str(l): p50[l] for l in layers},
           'n_dense_coordinates': {c: len(dense[c]) for c in cids},
           'values': {f'{p}|{c}|{arm}': v for (p, c, arm), v in res.items()},
           'per_context': {}}
    errs_d, errs_j = [], []
    for c in cids:
        e = {}
        for arm in arms:
            b, f = res[('bf16', c, arm)], res[('fp32', c, arm)]
            e[arm] = {'J2_bf16': b, 'J2_fp32': f, 'abs_diff_J2': abs(b - f)}
            errs_j.append(abs(b - f))
            if arm != 'none':
                db = b - res[('bf16', c, 'none')]
                df = f - res[('fp32', c, 'none')]
                e[arm]['dJ2_bf16'] = db
                e[arm]['dJ2_fp32'] = df
                e[arm]['abs_diff_dJ2'] = abs(db - df)
                e[arm]['rel_diff_dJ2'] = abs(db - df) / abs(df) if df else None
                errs_d.append(abs(db - df))
        rec['per_context'][c] = e
    rec['err_max_dJ2'] = max(errs_d)
    rec['err_median_dJ2'] = statistics.median(errs_d)
    rec['err_max_J2'] = max(errs_j)
    ref = {'plan216_density_1_to_2': 0.00525, 'plan216_density_2_to_32': 0.01444,
           'plan216_density_1_to_32': 0.01969, 'plan06_dose_0.005': 0.0059,
           'plan06_dose_0.0095': 0.0095, 'plan06_dose_0.02': 0.0191,
           'plan217_shape_drift_worst': 1.188e-2}
    rec['downgraded_quantities'] = {k: {'value': v, 'err_over_value': rec['err_max_dJ2'] / v}
                                    for k, v in ref.items()}
    verdict = ('bf16 is NOT the limiting factor at fixed shape; the 2.17 drift was a shape '
               'artifact and the downgraded findings can be restored'
               if rec['err_max_dJ2'] < 0.2 * ref['plan216_density_1_to_2'] else
               ('the error bar is real at fixed shape too; the 2.17.4 downgrades stand'
                if rec['err_max_dJ2'] > 0.5 * ref['plan216_density_1_to_2'] else
                'intermediate: some downgraded quantities survive, the smallest does not'))
    rec['verdict'] = verdict

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'j2_precision.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf8')
    lines = ['# bf16 vs float32 at the production batch shape', '',
             f'- machine `{rec["machine"]}` · budget {a.budget} · dense arm = {K} coordinates · '
             f'contexts {len(cids)}',
             f'- p50 (bf16, used for both precisions) ' + ', '.join(f'L{l}={p50[l]:.4f}' for l in layers),
             f'- err = max |dJ2_bf16 − dJ2_fp32| = **{rec["err_max_dJ2"]:.3e}** '
             f'(median {rec["err_median_dJ2"]:.3e}); on J2 itself max {rec["err_max_J2"]:.3e}', '',
             '| context | arm | dJ2 bf16 | dJ2 fp32 | abs diff | rel diff |', '|---|---|---:|---:|---:|---:|']
    for c in cids:
        for arm in arms:
            if arm == 'none':
                continue
            e = rec['per_context'][c][arm]
            lines.append(f'| `{c}` | {arm} | {e["dJ2_bf16"]:+.6f} | {e["dJ2_fp32"]:+.6f} | '
                         f'{e["abs_diff_dJ2"]:.3e} | {e["rel_diff_dJ2"]:.2%} |')
    lines += ['', '## err against every quantity §2.17.4 downgraded', '',
              '| quantity | value | err / value | stands? |', '|---|---:|---:|---|']
    for k, v in rec['downgraded_quantities'].items():
        r = v['err_over_value']
        lines.append(f'| `{k}` | {v["value"]:.5f} | {r:.2%} | '
                     f'{"yes (>5x)" if r < 0.2 else ("marginal" if r < 0.5 else "**no**")} |')
    lines += ['', f'verdict: {verdict}']
    (out / 'j2_precision.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/j2_precision.json and .md')


if __name__ == '__main__':
    main()
