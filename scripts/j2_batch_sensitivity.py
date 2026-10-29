#!/usr/bin/env python3
"""Is the J2 readout sensitive to BATCH COMPOSITION? (registered in plan R5 / 08_ref §8.3)

Why this question. Every number in this repo was produced with one fixed batch layout, so
the batch-composition degree of freedom has never been probed. The external Qwen3-8B atlas
measured it for free generation and found 28.34% of runs change text when only the batch
changes, with 58.1% byte-exact across three layouts (08_reference §8.1). "J2 is stable
under re-batching" is currently an ASSUMPTION, not a measured fact -- and it is the error
bar on every dJ2 we have ever reported.

Mathematically it should be stable: `build_batch` produces equal-length sequences with no
padding, and attention is causal, so each row's logits are independent of the other rows.
Numerically it need not be: batch shape changes kernel selection and therefore the
floating-point accumulation order, in bf16.

Variants, all read out with the SAME definition (log-mean-exp over the slot pool):
  repeat    identical batched call twice      -> pure run-to-run determinism
  single    one sequence per forward          -> batch vs no batch
  dup       every sequence duplicated (2n)    -> batch size
  half      two sub-batches                   -> batch layout
  shuffle   rows permuted, then un-permuted   -> row order (closest to their "batch layout")

Each variant is run twice: unsteered, and with the reference intervention
(L14 N8949 at the given dose) so we can see whether an active steering hook changes the
sensitivity.

Read-out of the result, frozen before running:
  drift = max over variants of |J2_variant - J2_full|, per context.
  Compare drift against the effect sizes we actually report:
    plan 06 single-neuron dJ2 at dose 0.05 = +0.0484
    plan 2.16 dense u_I at budget 0.05     = +0.05618
  drift << 0.001  -> negligible, prior numbers stand as they are
  drift ~ 0.01    -> a real error bar; every dJ2 needs it quoted
  drift ~ effect  -> the readout is not measuring what we think

GPU: per context, (n_adj + 6) + (n_verb + 6) forwards per steering state, two states.

    [i/N] elapsed, eta ...   per forward group (package progress.Progress)
"""
from __future__ import annotations
import argparse, hashlib, json, platform, random, statistics, sys
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
    ap.add_argument('--contexts', type=int, default=8)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--steer-layer', type=int, default=14)
    ap.add_argument('--steer-neuron', type=int, default=8949)
    ap.add_argument('--steer-dose', type=float, default=0.05)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--out', default='out/j2_batch_sensitivity')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lex, fams = load_pools(a.lexicon, a.pool_variant)
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    ctx = [r for r in rows if r.get('split') == a.split][:a.contexts]
    if not ctx:
        raise SystemExit(f'no {a.split} contexts in {a.probes}')
    variants = ['repeat', 'single', 'dup', 'half', 'shuffle']
    n_seq = {f: len(fams[f]['pos']) + len(fams[f]['neg']) for f in fams}
    per_state = sum(n_seq.values()) + 6 * len(fams)
    total = len(ctx) * 2 * per_state

    print('Question : does the J2 readout depend on batch composition?')
    print(f'Readout  : J2 = mean over families {sorted(fams)} of (log mean e^S+ - log mean e^S-), '
          f'pool variant {a.pool_variant}')
    print(f'Pool size: ' + ', '.join(f'{f}={n_seq[f]} seqs' for f in sorted(fams))
          + '  (equal-length, no padding, causal attention => rows SHOULD be independent)')
    print(f'Variants : {variants}   x 2 steering states (none, L{a.steer_layer}N{a.steer_neuron} '
          f'dose {a.steer_dose})')
    print(f'Contexts : {len(ctx)} -> {[c["id"] for c in ctx]}')
    print(f'Work     : {total} forwards ({len(ctx)} ctx x 2 states x {per_state})')
    print('Frozen   : drift = max over variants of |J2_variant - J2_full|; compare against '
          'dJ2 = +0.0484 (plan 06, single neuron, dose 0.05) and +0.05618 (plan 2.16, dense, b=0.05)')
    if a.dry_run:
        for f in sorted(fams):
            print(f'  {f}: template {fams[f]["template"]!r} pos={len(fams[f]["pos"])} neg={len(fams[f]["neg"])}')
        print(f'[dry-run] would write {a.out}/j2_batch_sensitivity.json and .md')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import apply_steering, column_norms, estimate_p50, load_model
    from progress import Progress
    from semantic import batch_mean_logp, build_batch, language_log_mass

    tok, model, lm = load_model(a.model)
    norms = column_norms(lm, [a.steer_layer])
    p50 = estimate_p50(tok, model, lm, [a.steer_layer], device='cuda:0')
    dev = 'cuda:0'
    steer = [{'layer': a.steer_layer, 'neuron': a.steer_neuron, 'dose': a.steer_dose}]

    def S_of(ids, pre, suf):
        """Per-sequence mean logP of its slot suffix, under the current steering context."""
        return batch_mean_logp(model, ids, pre, suf)

    def j2_from_S(S, kinds):
        j_pos, _ = language_log_mass(S[kinds > 0].detach())
        j_neg, _ = language_log_mass(S[kinds < 0].detach())
        return j_pos - j_neg

    prog = Progress(total)
    rec = {'schema': 'qwentest-j2-batch-sensitivity-v1', 'machine': platform.node(),
           'model': a.model, 'pool_variant': a.pool_variant, 'variants': variants,
           'steer': steer, 'contexts': [c['id'] for c in ctx], 'seed': a.seed,
           'lexicon_sha256': hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest(),
           'p50': {str(a.steer_layer): p50[a.steer_layer]},
           'reference_effects': {'plan06_single_neuron_dose_0.05': 0.0484,
                                 'plan216_dense_budget_0.05': 0.05618},
           'per_context': {}}

    for c in ctx:
        batches = {}
        for fam, f in sorted(fams.items()):
            ids, pre, suf, kinds = build_batch(tok, c['prompt'], f['template'], f['pos'], f['neg'],
                                              dev, a.max_seq)
            batches[fam] = (ids, pre, suf, kinds)
        entry = {}
        for state, iv in (('none', []), ('steered', steer)):
            with apply_steering(lm, iv, p50, norms), torch.inference_mode():
                def total_j2(collect):
                    vals = []
                    for fam, (ids, pre, suf, kinds) in batches.items():
                        vals.append(j2_from_S(collect(fam, ids, pre, suf, kinds), kinds))
                    return sum(vals) / len(vals)

                def c_full(fam, ids, pre, suf, kinds):
                    return S_of(ids, pre, suf)

                base = total_j2(c_full)
                prog.step(f'{c["id"]} {state} full J2={base:+.6f}')
                out = {'full': base}

                out['repeat'] = total_j2(c_full)
                prog.step(f'{c["id"]} {state} repeat')

                def c_single(fam, ids, pre, suf, kinds):
                    return torch.cat([S_of(ids[i:i + 1], pre, suf) for i in range(ids.shape[0])])

                out['single'] = total_j2(c_single)
                prog.step(f'{c["id"]} {state} single ({sum(n_seq.values())} forwards)')

                def c_dup(fam, ids, pre, suf, kinds):
                    S = S_of(ids.repeat(2, 1), pre, suf)
                    n = ids.shape[0]
                    out.setdefault('_dup_halves_agree', []).append(
                        float((S[:n] - S[n:]).abs().max()))
                    return S[:n]

                out['dup'] = total_j2(c_dup)
                prog.step(f'{c["id"]} {state} dup')

                def c_half(fam, ids, pre, suf, kinds):
                    n = ids.shape[0]; h = n // 2
                    return torch.cat([S_of(ids[:h], pre, suf), S_of(ids[h:], pre, suf)])

                out['half'] = total_j2(c_half)
                prog.step(f'{c["id"]} {state} half')

                perms = {}
                for fam, (ids, pre, suf, kinds) in batches.items():
                    # stable seed: Python's hash() on str is randomised per process
                    h = int(hashlib.sha256(f'{c["id"]}|{state}|{fam}'.encode()).hexdigest()[:8], 16)
                    rng = random.Random(a.seed + h)
                    perms[fam] = rng.sample(range(ids.shape[0]), ids.shape[0])

                def c_shuf(fam, ids, pre, suf, kinds):
                    p = torch.tensor(perms[fam], device=ids.device)
                    S = S_of(ids[p], pre, suf)
                    inv = torch.empty_like(p); inv[p] = torch.arange(len(p), device=p.device)
                    return S[inv]

                out['shuffle'] = total_j2(c_shuf)
                prog.step(f'{c["id"]} {state} shuffle')
            entry[state] = out
        rec['per_context'][c['id']] = entry

    # ---------------- analysis ----------------
    drift = {}
    for state in ('none', 'steered'):
        per_var = {}
        for v in variants:
            ds = [abs(rec['per_context'][c][state][v] - rec['per_context'][c][state]['full'])
                  for c in rec['contexts']]
            per_var[v] = {'max': max(ds), 'median': statistics.median(ds), 'all': ds}
        drift[state] = per_var
    rec['drift'] = drift
    ref = min(rec['reference_effects'].values())
    worst = max(drift[st][v]['max'] for st in drift for v in variants)
    rec['worst_drift'] = worst
    rec['worst_drift_over_smallest_reference_effect'] = worst / ref
    # the duplicated batch's two halves must agree if rows are truly independent
    dup_halves = [x for c in rec['contexts'] for st in ('none', 'steered')
                  for x in rec['per_context'][c][st].get('_dup_halves_agree', [])]
    rec['dup_halves_max_abs_diff'] = max(dup_halves) if dup_halves else None

    # The quantity we actually REPORT is dJ2 = steered - none, not J2 itself. Its error
    # under a batch-shape change is the number that bounds every published dJ2.
    dj = {}
    for c in rec['contexts']:
        e = rec['per_context'][c]
        dj[c] = {'full': e['steered']['full'] - e['none']['full'],
                 **{v: e['steered'][v] - e['none'][v] for v in variants}}
    rec['dJ2'] = dj
    err = {}
    for v in variants:
        ds = [abs(dj[c][v] - dj[c]['full']) for c in rec['contexts']]
        err[v] = {'max': max(ds), 'median': statistics.median(ds), 'all': ds}
    rec['dJ2_error'] = err
    worst_dj = max(err[v]['max'] for v in variants)
    med_dj = statistics.median([err[v]['median'] for v in variants if v != 'repeat'])
    rec['worst_dJ2_error'] = worst_dj
    rec['median_dJ2_error'] = med_dj
    rec['smallest_abs_dJ2_full'] = min(abs(dj[c]['full']) for c in rec['contexts'])
    rec['dJ2_error_over_reference'] = {k: worst_dj / v for k, v in rec['reference_effects'].items()}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'j2_batch_sensitivity.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                                   encoding='utf8')
    lines = ['# J2 batch-composition sensitivity', '',
             f'- machine `{rec["machine"]}` · model `{a.model}` · pool variant `{a.pool_variant}` · '
             f'{len(ctx)} contexts',
             f'- reference: L{a.steer_layer}N{a.steer_neuron} dose {a.steer_dose}; '
             f'p50 = {p50[a.steer_layer]:.6f}',
             f'- effect sizes we report elsewhere: plan 06 single-neuron dJ2 = +0.0484, '
             f'plan 2.16 dense b=0.05 = +0.05618', '',
             '| context | state | J2 full | repeat | single | dup | half | shuffle |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for c in rec['contexts']:
        for state in ('none', 'steered'):
            e = rec['per_context'][c][state]
            lines.append(f'| `{c}` | {state} | {e["full"]:+.6f} | '
                         + ' | '.join(f'{e[v] - e["full"]:+.2e}' for v in variants) + ' |')
    lines += ['', '## drift = |J2_variant - J2_full|', '',
              '| state | variant | max | median |', '|---|---|---:|---:|']
    for state in ('none', 'steered'):
        for v in variants:
            lines.append(f'| {state} | `{v}` | {drift[state][v]["max"]:.3e} | '
                         f'{drift[state][v]["median"]:.3e} |')
    lines += ['', '## error on the REPORTED quantity dJ2 = J2(steered) - J2(none)', '',
              '| variant | max | median | max as %% of the smallest |dJ2| here (%.5f) |' %
              rec['smallest_abs_dJ2_full'], '|---|---:|---:|---:|']
    for v in variants:
        lines.append(f'| `{v}` | {err[v]["max"]:.3e} | {err[v]["median"]:.3e} | '
                     f'{err[v]["max"] / rec["smallest_abs_dJ2_full"]:.2%} |')
    dh = rec['dup_halves_max_abs_diff']
    dh_s = 'n/a' if dh is None else f'{dh:.3e}'
    lines += ['', f'worst drift on J2 itself            : {worst:.3e}',
              f'worst error on the reported dJ2     : {worst_dj:.3e}   '
              f'(median over variants {med_dj:.3e})',
              f'duplicated batch, max |S(half1) - S(half2)|: {dh_s}',
              f'reference effects                    : '
              + ', '.join(f'{k}={v}' for k, v in rec['reference_effects'].items()),
              f'worst dJ2 error as a share of those  : '
              + ', '.join(f'{k}={v:.2%}' for k, v in rec['dJ2_error_over_reference'].items()), '']
    # Bands are the ones frozen in this file's docstring BEFORE the run:
    #   drift << 0.001 -> negligible | drift ~ 0.01 -> real error bar | drift ~ effect -> dominant
    # The middle band is therefore "well below the effects we report", implemented as < 0.5*ref.
    # (An earlier version of this code used 0.1*ref, which did not implement the frozen text and
    # mislabelled a 1.2e-2 drift against a 4.84e-2 effect as DOMINANT. The docstring is the
    # criterion; this only fixes the code to match it.)
    if worst < 1e-3:
        verdict = (f'NEGLIGIBLE (frozen band "drift << 0.001"): measured {worst:.3e}; '
                   f'prior numbers stand as they are')
    elif worst < 0.5 * ref:
        verdict = (f'REAL ERROR BAR (frozen band "drift ~ 0.01"): measured {worst:.3e} = '
                   f'{worst / ref:.1%} of the smallest effect we report ({ref}); '
                   f'every dJ2 must quote it')
    else:
        verdict = (f'DOMINANT (frozen band "drift ~ effect"): measured {worst:.3e} vs smallest '
                   f'reported effect {ref}; the readout is not measuring what we think')
    lines.append(f'verdict: {verdict}')
    rec['verdict'] = verdict
    (out / 'j2_batch_sensitivity.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                                   encoding='utf8')
    (out / 'j2_batch_sensitivity.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/j2_batch_sensitivity.json and .md')


if __name__ == '__main__':
    main()
