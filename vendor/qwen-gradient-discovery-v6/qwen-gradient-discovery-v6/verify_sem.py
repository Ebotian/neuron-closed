#!/usr/bin/env python3
"""Held-out effect of the J2 (language-mass) candidates, with the J2 objective itself.

Same design as verify.py (physical coordinates, top-K vs same-layer random with the
random neuron's OWN predicted sign), but the measured quantity is J2, not the string
contrast. Each physical coordinate is measured once per probe at dose 0 and at the
signed dose; --opposite adds the mirrored arm (2x cost).
"""
from __future__ import annotations

import argparse, json, statistics, time
from pathlib import Path

import torch

from audit import (check_scan_file, file_hash, guarded_manifest, require_same_code, require_same_model,
                   require_same_runtime, runtime_versions)
from design import build_arms, physical_coordinates
from engine import column_norms, load_model, read_jsonl, write_json
from progress import Progress
from report import distribution, exact_sign_test
from semantic import j2_value


def j2_mean(tok, model, lm, p50, norms, prompt, fams, interventions=None, device='cuda:0', max_seq=384):
    vals = {k: j2_value(tok, model, lm, p50, norms, prompt, f['template'], f, interventions,
                        device, max_seq) for k, f in fams.items()}
    return statistics.mean(vals.values()), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan-dir', default='out/j2')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes', default='probes/contrasts.jsonl')
    ap.add_argument('--trait', default='hostility')
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--random-k', type=int, default=8)
    ap.add_argument('--dose', type=float, default=0.02)
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--seed', type=int, default=1729)
    ap.add_argument('--opposite', action='store_true', help='also measure the mirrored arm (2x cost)')
    ap.add_argument('--reuse-fingerprint', action='store_true')
    ap.add_argument('--out', default='out/verify_j2')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if a.top_k <= 0 or a.dose <= 0:
        raise ValueError('--top-k and --dose must be positive')

    scan = Path(a.scan_dir)
    meta = json.loads((scan / 'manifest.json').read_text(encoding='utf8'))
    if meta.get('schema') != 'gradient-scan-v2':
        raise ValueError('need a gradient-scan-v2 scan dir')
    check_scan_file(a.probes, meta)
    top = json.loads((scan / f'{a.trait}_top.json').read_text(encoding='utf8'))[:a.top_k]
    grads = torch.load(scan / f'{a.trait}_all_gradients.pt', map_location='cpu', weights_only=True)
    layers = meta['layers']
    dims = {int(l): len(grads[l]['mean']) for l in layers}
    arms = build_arms(top, grads, dims, a.random_k, a.seed)
    coords = physical_coordinates(arms)
    rows = [r for r in read_jsonl(a.probes) if r['trait'] == a.trait and r['split'] == 'heldout']
    if not rows:
        raise ValueError('no held-out probes')
    fams = {k: {'pos': v['pos'], 'neg': v['neg'], 'template': meta['templates'][k]}
            for k, v in meta['pools'].items()}
    model_fp = meta['model_fingerprint'] if a.dry_run else require_same_model(
        a.model, meta, reuse_fingerprint=a.reuse_fingerprint)
    plan = {'schema': 'verification-j2', 'trait': a.trait, 'dose': a.dose, 'seed': a.seed,
            'scan_identity': file_hash(scan / 'manifest.json'), 'top_sha256': file_hash(
                scan / f'{a.trait}_top.json'),
            'probe_file_sha256': file_hash(a.probes), 'objective': 'language-mass-v2 (J2)',
            'model_fingerprint': model_fp, 'runtime': meta['runtime'],
            'code_fingerprint': meta['code_fingerprint'],
            'heldout_probes': rows, 'arms': arms, 'opposite': a.opposite,
            'physical_coordinates': coords, 'max_seq': a.max_seq}
    print(f'Plan: {len(top)} top + {a.random_k} same-layer random; {len(coords)} physical coords x '
          f'{len(rows)} held-out probes; arms per probe = {1 if not a.opposite else 2}; '
          f'families = {list(fams)}')
    if a.dry_run:
        return
    require_same_runtime(meta)
    require_same_code(meta)
    out = Path(a.out)
    guarded_manifest(out, plan)
    tok, model, lm = load_model(a.model)
    cal = json.loads((scan / 'calibration.json').read_text())
    p50 = {int(l): float(v) for l, v in cal['p50'].items()}
    norms = column_norms(lm, layers)
    path = out / 'verify.jsonl'
    done = {(r['probe_id'], r['coordinate'], r['arm']) for r in read_jsonl(path)} if path.exists() else set()
    prog = Progress(max(1, len(rows) * len(coords)))
    with path.open('a', encoding='utf8') as f:
        for probe in rows:
            base, base_fam = j2_mean(tok, model, lm, p50, norms, probe['prompt'], fams,
                                     max_seq=a.max_seq)
            for a_ in arms:
                key = (probe['id'], a_['coordinate'], 'signed')
                if key in done:
                    continue
                iv = [{'layer': a_['layer'], 'neuron': a_['neuron'], 'dose': a.dose * a_['sign']}]
                val, fam_vals = j2_mean(tok, model, lm, p50, norms, probe['prompt'], fams, iv,
                                        max_seq=a.max_seq)
                rec = {'probe_id': probe['id'], 'coordinate': a_['coordinate'], 'arm': 'signed',
                       'group': a_['group'], 'sign': a_['sign'], 'delta': val - base,
                       'delta_by_family': {k: fam_vals[k] - base_fam[k] for k in fams},
                       'baseline': base, 'steered': val}
                f.write(json.dumps(rec, ensure_ascii=False) + '\n'); f.flush()
                done.add(key)
                prog.step(f'{probe["id"]} {a_["coordinate"]} {a_["sign"]:+d} dJ2={rec["delta"]:+.5f}')
    # summarise
    recs = read_jsonl(path)
    by = {}
    for r in recs:
        by.setdefault(r['group'], []).append(r)
    summary = {'dose': a.dose, 'groups': {}}
    for g, rs in by.items():
        per_arm = {}
        for r in rs:
            per_arm.setdefault(r['coordinate'], []).append(r['delta'])
        means = {c: statistics.mean(v) for c, v in per_arm.items()}
        summary['groups'][g] = {'n_coordinates': len(means), 'mean_delta': statistics.mean(means.values()),
                                **distribution(list(means.values()))}
    pairs = []
    top_by_id = {a_['arm_id']: a_ for a_ in arms if a_['group'] == 'gradient_top'}
    for a_ in arms:
        if a_['group'] != 'random_own_sign':
            continue
        par = top_by_id.get(a_.get('matched_to'))
        if par is None:
            continue
        t = [r['delta'] for r in recs if r['coordinate'] == par['coordinate'] and r['group'] == 'gradient_top']
        r_ = [r['delta'] for r in recs if r['coordinate'] == a_['coordinate'] and r['group'] == 'random_own_sign']
        if t and r_:
            pairs.append(statistics.mean(t) - statistics.mean(r_))
    summary['paired_top_minus_random'] = {'n': len(pairs), 'mean': statistics.mean(pairs) if pairs else None,
                                          'sign_test': exact_sign_test(pairs), **distribution(pairs)}
    write_json(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False)[:1200])


if __name__ == '__main__':
    main()
