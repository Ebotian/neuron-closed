#!/usr/bin/env python3
"""Is "hostility never first" a property of the top-K, or only of the single draw?

For each of the top-K coordinates of a J2 scan, measure the five J1 trait margins on the
shared contexts at one dose, and report which trait each coordinate lifts most, plus the
rank of hostility. One draw cannot speak for a class (plan 06 §8, conclusion 3); K draws
can at least show whether the profile is uniform.

    [i/N] elapsed, eta ...   per measurement (package progress.Progress)
"""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path


def resolve_pkg(given):
    here = Path(__file__).resolve().parent
    cands = [Path(given)] if given else []
    cands += [here.parent / '03_unpacked/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
              Path('/root/autodl-tmp/qwen-gradient-discovery-v6')]
    for c in cands:
        if (c / 'engine.py').is_file():
            return c
    raise SystemExit('package not found; pass --pkg <dir with engine.py>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--scan-dir', required=True)
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--dose', type=float, default=0.02)
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--out', default='out/crosstrait_k')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from progress import Progress

    scan = Path(a.scan_dir)
    meta = json.loads((scan / 'manifest.json').read_text(encoding='utf8'))
    cal = json.loads((scan / 'calibration.json').read_text(encoding='utf8'))
    top = json.loads((scan / 'hostility_top.json').read_text(encoding='utf8'))[:a.top_k]
    spec = [json.loads(l) for l in (pkg / 'probes/specificity.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
    traits = sorted(spec[0]['continuations'])
    n_base = len(spec) * len(traits)
    n = n_base + len(top) * len(spec) * (len(traits) + 1)
    print(f'Plan: {len(top)} coordinates x {len(spec)} contexts x {len(traits)} traits at dose {a.dose}')
    print(f'      measurements: {n_base} shared baselines + {len(top)}x{len(spec)}x{len(traits) + 1} '
          f'(traits + placebo) = {n}')
    print(f'      estimate ~{n * 0.06 + 40:.0f} s (0.06 s/pair + 40 s hash/load)')
    if a.dry_run:
        for c in top:
            print(f'      L{c["layer"]}N{c["neuron"]}{c["sign"]:+d} |g|={c["abs_gradient"]:.3f}')
        return

    import torch
    from engine import column_norms, load_model, pair_objective

    tok, model, lm = load_model(a.model)
    p50 = {int(l): float(v) for l, v in cal['p50'].items()}
    norms = column_norms(lm, meta['layers'])
    device = 'cuda:0'
    prog = Progress(n)
    base = {}
    for ctx in spec:
        for t in traits:
            base[(ctx['id'], t)] = pair_objective(
                tok, model, lm, {'prompt': ctx['prompt'], 'positive': ctx['continuations'][t],
                                 'negative': ctx['neutral_1']}, device, p50, norms, [], a.max_seq)[0]
            prog.step(f'baseline {ctx["id"]} {t}')
    rows = []
    for c in top:
        iv = [{'layer': int(c['layer']), 'neuron': int(c['neuron']), 'dose': a.dose * int(c['sign'])}]
        prof = {}
        for ctx in spec:
            for t in traits:
                v = pair_objective(tok, model, lm, {'prompt': ctx['prompt'],
                                                    'positive': ctx['continuations'][t],
                                                    'negative': ctx['neutral_1']},
                                   device, p50, norms, iv, a.max_seq)[0]
                prof.setdefault(t, []).append(v - base[(ctx['id'], t)])
                prog.step(f'L{c["layer"]}N{c["neuron"]}{c["sign"]:+d} {ctx["id"]} {t} d={v - base[(ctx["id"], t)]:+.4f}')
            pb = pair_objective(tok, model, lm, {'prompt': ctx['prompt'], 'positive': ctx['neutral_2'],
                                                'negative': ctx['neutral_1']},
                                device, p50, norms, iv, a.max_seq)[0]
            prog.step(f'L{c["layer"]}N{c["neuron"]}{c["sign"]:+d} {ctx["id"]} placebo')
        m = {t: statistics.mean(v) for t, v in prof.items()}
        order = sorted(m.items(), key=lambda kv: -kv[1])
        rows.append({'coord': f'L{int(c["layer"])}N{int(c["neuron"])}{int(c["sign"]):+d}',
                     'abs_gradient': c['abs_gradient'], 'means': m,
                     'order': [t for t, _ in order],
                     'hostility_rank': [t for t, _ in order].index('hostility') + 1,
                     'hostility_over_best': m['hostility'] / max(1e-9, order[0][1])})
    rows.sort(key=lambda r: r['hostility_rank'])
    outp = Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / 'crosstrait_k.json').write_text(json.dumps({'dose': a.dose, 'contexts': len(spec),
                                                        'traits': traits, 'rows': rows}, indent=2,
                                                       ensure_ascii=False), encoding='utf8')
    lines = ['# Cross-trait profile of the top-K (J1 string margins, shared contexts)', '',
             f'dose {a.dose} · {len(spec)} contexts · control-free (absolute profile per coordinate)', '',
             '| coord | hostility rank | hostility / best | order (best→worst) |', '|---|---:|---:|---|']
    for r in rows:
        lines.append(f'| `{r["coord"]}` | {r["hostility_rank"]}/5 | {r["hostility_over_best"]:.2f} | '
                     + ' > '.join(r['order']) + ' |')
    ranks = [r['hostility_rank'] for r in rows]
    lines += ['', f'hostility rank across the top-{len(rows)}: {ranks} '
                  f'(median {statistics.median(ranks)}, first-place count {ranks.count(1)})',
              f'hostility/best ratio: median {statistics.median(r["hostility_over_best"] for r in rows):.2f}']
    (outp / 'crosstrait_k.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
