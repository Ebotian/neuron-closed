#!/usr/bin/env python3
"""Is the J2 hostility readout more than a "praise register" readout?

Two readouts inside the SAME family (same template, same tokenizer, single-token slots):

  R_H = log mass(hostile adjectives)      - log mass(compliment adjectives)
  R_P = log mass(compliment adjectives)   - log mass(neutral adjectives)

R_H is the readout we have been calling "hostility". If a top-K coordinate moves R_H much
more than R_P, the readout is not just "evaluative register"; if both move alike, it is.
Needs the two pool files (not vendored): lexicons/jibes_adj_verb_pools.json (hostile,
compliment) and a HurtLex-derived pool for the neutral adjectives (r2_pools.json).

    [i/N] elapsed, eta ...   per measurement
"""
from __future__ import annotations
import argparse, json, statistics, sys
from pathlib import Path


def resolve_pkg(given):
    here = Path(__file__).resolve().parent
    cands = [Path(given)] if given else []
    cands += [here.parent / 'vendor/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
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
    ap.add_argument('--probes', required=True)
    ap.add_argument('--pools', required=True, help='jibes_adj_verb_pools.json (hostile + compliment)')
    ap.add_argument('--neutral-pools', required=True, help='r2_pools.json (neutral adjectives in .families.adj.neg)')
    ap.add_argument('--top-k', type=int, default=8)
    ap.add_argument('--dose', type=float, default=0.02)
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--out', default='out/j2_two_readouts')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from progress import Progress

    scan = Path(a.scan_dir)
    meta = json.loads((scan / 'manifest.json').read_text(encoding='utf8'))
    cal = json.loads((scan / 'calibration.json').read_text(encoding='utf8'))
    top = json.loads((scan / 'hostility_top.json').read_text(encoding='utf8'))[:a.top_k]
    jib = json.loads(Path(a.pools).read_text(encoding='utf8'))['families']['adj']
    neu = json.loads(Path(a.neutral_pools).read_text(encoding='utf8'))['families']['adj']['neg']
    held = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    held = [r for r in held if r['trait'] == 'hostility' and r['split'] == 'heldout']
    tpl = 'You are so {w}.'
    RH = {'pos': jib['pos'], 'neg': jib['neg'], 'template': tpl}
    RP = {'pos': jib['neg'], 'neg': neu, 'template': tpl}
    n = len(held) * 2 + len(top) * len(held) * 2 * 2
    print(f'Plan: R_H = hostile({len(RH["pos"])}) - compliment({len(RH["neg"])}); '
          f'R_P = compliment({len(RP["pos"])}) - neutral({len(RP["neg"])})')
    print(f'      {len(top)} coordinates x {len(held)} held-out probes at dose {a.dose}; '
          f'measurements {n} (+{len(held)*2} baselines)')
    print(f'      estimate ~{(n + 2*len(held)) * 0.1 + 40:.0f} s (0.1 s/batched J2 + 40 s hash/load)')
    if a.dry_run:
        return

    import torch
    from engine import column_norms, load_model
    from semantic import j2_value

    tok, model, lm = load_model(a.model)
    p50 = {int(l): float(v) for l, v in cal['p50'].items()}
    norms = column_norms(lm, meta['layers'])
    device = 'cuda:0'
    prog = Progress(n + 2 * len(held))
    base = {}
    for r in held:
        base[r['id']] = (j2_value(tok, model, lm, p50, norms, r['prompt'], tpl, RH, [], device, a.max_seq),
                         j2_value(tok, model, lm, p50, norms, r['prompt'], tpl, RP, [], device, a.max_seq))
        prog.step(f'baseline {r["id"]}  R_H={base[r["id"]][0]:+.4f} R_P={base[r["id"]][1]:+.4f}')
    rows = []
    for c in top:
        iv = [{'layer': int(c['layer']), 'neuron': int(c['neuron']), 'dose': a.dose * int(c['sign'])}]
        dh, dp = [], []
        for r in held:
            h = j2_value(tok, model, lm, p50, norms, r['prompt'], tpl, RH, iv, device, a.max_seq) - base[r['id']][0]
            p = j2_value(tok, model, lm, p50, norms, r['prompt'], tpl, RP, iv, device, a.max_seq) - base[r['id']][1]
            dh.append(h); dp.append(p)
            prog.step(f'L{c["layer"]}N{c["neuron"]}{c["sign"]:+d} {r["id"]} dR_H={h:+.4f} dR_P={p:+.4f}')
        rows.append({'coord': f'L{int(c["layer"])}N{int(c["neuron"])}{int(c["sign"]):+d}',
                     'abs_gradient': c['abs_gradient'],
                     'dR_H': statistics.mean(dh), 'dR_P': statistics.mean(dp),
                     'dR_H_sd': statistics.stdev(dh), 'dR_P_sd': statistics.stdev(dp),
                     'diff': statistics.mean(dh) - statistics.mean(dp),
                     'H_over_P': (statistics.mean(dh) / statistics.mean(dp)
                                  if abs(statistics.mean(dp)) > 1e-9 else None),
                     'per_probe_diff': [h - p for h, p in zip(dh, dp)]})
    rows.sort(key=lambda r: -r['diff'])
    outp = Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / 'two_readouts.json').write_text(json.dumps({'dose': a.dose, 'heldout': [r['id'] for r in held],
                                                        'readouts': {'R_H': 'hostile - compliment',
                                                                     'R_P': 'compliment - neutral'},
                                                        'pools': {'hostile': RH['pos'], 'compliment': RH['neg'],
                                                                  'neutral': RP['neg']},
                                                        'rows': rows}, indent=2, ensure_ascii=False),
                                            encoding='utf8')
    lines = ['# J2 two-readout test (hostile-vs-compliment vs compliment-vs-neutral)', '',
             f'dose {a.dose} · {len(held)} held-out probes · same template `You are so {{w}}.`', '',
             '| coord | ΔR_H (hostile−compliment) | ΔR_P (compliment−neutral) | ΔR_H−ΔR_P | H/P |',
             '|---|---:|---:|---:|---:|']
    for r in rows:
        hp = f'{r["H_over_P"]:.2f}' if r['H_over_P'] is not None else 'n/a'
        lines.append(f'| `{r["coord"]}` | {r["dR_H"]:+.5f} | {r["dR_P"]:+.5f} | {r["diff"]:+.5f} | {hp} |')
    pos = sum(1 for r in rows if r['diff'] > 0)
    lines += ['', f'ΔR_H > ΔR_P for {pos}/{len(rows)} coordinates; '
                  f'median ΔR_H {statistics.median(r["dR_H"] for r in rows):+.5f}, '
                  f'median ΔR_P {statistics.median(r["dR_P"] for r in rows):+.5f}',
              '', 'Reading: if the two readouts move alike (ΔR_H ≈ ΔR_P), these coordinates are broad',
              'evaluative-register levers. If ΔR_H dominates, the hostility readout carries something',
              'beyond "praise register". This is a within-family comparison; no cross-instrument confound.']
    (outp / 'two_readouts.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
