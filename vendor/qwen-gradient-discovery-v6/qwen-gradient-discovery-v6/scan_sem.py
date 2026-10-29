#!/usr/bin/env python3
"""Scan with the language-mass readout J2 (plan §2.2). Same artifact schema as scan.py.

Writes exactly what verify.py / stability.py / report.py expect:
  manifest.json (schema gradient-scan-v2), calibration.json, <trait>_top.json,
  <trait>_all_gradients.pt, <trait>_known_rank.json, <trait>_probe_scores.jsonl

The readout is:  J2 = mean over template families of  log(mean e^{S+}) - log(mean e^{S-})
with S the teacher-forced per-token mean log-probability of a slot language.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import torch

from audit import code_fingerprint, file_hash, fingerprint_with_stat_cache, guarded_manifest, runtime_versions
from engine import (aggregate_probe_gradients, column_norms, dimensions, estimate_p50,
                    known_ranks, load_model, parse_layers, ranked_candidates, read_jsonl, write_json)
from progress import Progress
from semantic import fd_check, sem_gradient


def load_pools(path, variant):
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
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes', default='probes/contrasts.jsonl')
    ap.add_argument('--specificity-probes', default='probes/specificity.jsonl')
    ap.add_argument('--choices-tasks', default='probes/choices.jsonl')
    ap.add_argument('--trait', default='hostility')
    ap.add_argument('--layers', default='13-14')
    ap.add_argument('--max-probes', type=int, default=6)
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--top-k', type=int, default=256)
    ap.add_argument('--lexicon', required=True, help='pools.json from build_lexicon.py (NOT committed)')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'],
                    help='full = whole hostile pool; A/B = disjoint halves (readout-variance test)')
    ap.add_argument('--fd-check', type=float, default=None, metavar='DOSE',
                    help='run a same-prompt central-difference check at this dose after scanning')
    ap.add_argument('--out', default='out/j2')
    ap.add_argument('--reuse-fingerprint', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    layers = parse_layers(args.layers)
    probes = read_jsonl(args.probes)
    rows = [p for p in probes if p['trait'] == args.trait and p['split'] == 'discovery'][:args.max_probes]
    if not rows:
        raise ValueError(f'no discovery probes for {args.trait}')
    lex, fams = load_pools(args.lexicon, args.pool_variant)
    print(f'Plan: {len(rows)} probes x {len(fams)} template families ({", ".join(fams)}); '
          f'{sum(len(f["pos"]) + len(f["neg"]) for f in fams.values())} slot sequences per probe')
    print(f'readout: language-mass J2, pool variant {args.pool_variant}  '
          f'(hostile slots per family: ' + ', '.join(f'{k}:{len(v["pos"])}' for k, v in fams.items()) + ')')
    if args.dry_run:
        for fam, f in fams.items():
            print(f'  {fam}: {f["template"]!r}  pos={f["pos"]}  neg={f["neg"]}')
        return

    out = Path(args.out)
    print('SHA256 hashing model shards & tokenizer for reproducibility...', flush=True)
    identity, fp_mode = fingerprint_with_stat_cache(args.model, reuse=args.reuse_fingerprint)
    plan = {'schema': 'gradient-scan-v2', 'package_release': 'v6',
            'objective': 'language-mass-v2 (J2: log mean e^S+ - log mean e^S-)',
            'pool_variant': args.pool_variant,
            'lexicon_sha256': file_hash(args.lexicon),
            'templates': {k: v['template'] for k, v in fams.items()},
            'pools': {k: {'pos': v['pos'], 'neg': v['neg']} for k, v in fams.items()},
            'model_fingerprint': identity, 'fingerprint_verification': fp_mode,
            'runtime': runtime_versions(), 'code_fingerprint': code_fingerprint(),
            'probe_file_sha256': file_hash(args.probes),
            'auxiliary_dataset_sha256': {'specificity': file_hash(args.specificity_probes),
                                         'choices': file_hash(args.choices_tasks)},
            'traits': [args.trait], 'subsets': {args.trait: rows}, 'layers': layers,
            'max_seq': args.max_seq, 'top_k': args.top_k,
            'exploratory_gate': {'required': False, 'override': False},
            'method': 'teacher-forced slot languages; one grad-enabled batched pass per family per probe'}
    guarded_manifest(out, plan)
    print('Loading model...', flush=True)
    tok, model, lm = load_model(args.model)
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device='cuda:0')
    dims = dimensions(lm, layers)
    write_json(out / 'calibration.json', {'p50': p50, 'dims': dims})

    per_probe = []
    prog = Progress(len(rows))
    with (out / f'{args.trait}_probe_scores.jsonl').open('w', encoding='utf8') as log:
        for probe in rows:
            t0 = time.perf_counter()
            gsum = None
            jfam = {}
            for fam, f in fams.items():
                j2, g = sem_gradient(tok, model, lm, layers, probe['prompt'], f['template'], f,
                                     'cuda:0', args.max_seq)
                jfam[fam] = j2
                gsum = g if gsum is None else {l: gsum[l] + g[l] for l in layers}
            grad = {l: gsum[l] / len(fams) for l in layers}       # mean over families
            per_probe.append(grad)
            rec = {'probe_id': probe['id'], 'J2': sum(jfam.values()) / len(jfam),
                   'J2_by_family': jfam, 'elapsed_s': round(time.perf_counter() - t0, 3)}
            log.write(json.dumps(rec, ensure_ascii=False) + '\n'); log.flush()
            prog.step(f'{args.trait} {probe["id"]} J2={rec["J2"]:+.4f} ' +
                      ' '.join(f'{k}={v:+.3f}' for k, v in jfam.items()))
            for l in layers:
                if not torch.isfinite(grad[l]).all():
                    raise FloatingPointError(f'non-finite gradient {args.trait} {probe["id"]} L{l}')

    aggr = aggregate_probe_gradients(per_probe, layers, p50, norms)
    winners, ranked = ranked_candidates(aggr, args.top_k)
    write_json(out / f'{args.trait}_top.json', winners)
    write_json(out / f'{args.trait}_known_rank.json', known_ranks(ranked))
    torch.save({l: {k: aggr[l][k] for k in ('mean', 'per_probe', 'consistency')} for l in layers},
               out / f'{args.trait}_all_gradients.pt')
    for i, w in enumerate(winners[:10], 1):
        print(f'  {i}: L{w["layer"]} N{w["neuron"]} sign={w["sign"]:+d} |g|={w["abs_gradient"]:.5g}')

    if args.fd_check:
        top4 = [{**w, 'gradient_per_dose': w['gradient_per_dose']} for w in winners[:4]]
        rows_fd = fd_check(tok, model, lm, p50, norms, layers, rows[0]['prompt'],
                           list(fams.values())[0]['template'], list(fams.values())[0],
                           top4, args.fd_check, 'cuda:0', args.max_seq)
        write_json(out / 'fd_check.json', {'dose': args.fd_check, 'probe': rows[0]['id'],
                                           'family': list(fams)[0], 'rows': rows_fd})
        print('fd_check (analytic vs central difference, same prompt):')
        for r in rows_fd:
            print(f'  L{r["layer"]}N{r["neuron"]}{r["sign"]:+d} analytic={r["analytic"]:+.4g} '
                  f'fd={r["fd"]:+.4g}')
    print('Done:', out)


if __name__ == '__main__':
    main()
