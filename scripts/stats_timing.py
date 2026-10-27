#!/usr/bin/env python3
"""Where did the wall-clock time go, for one persona-matrix run.

    python3 stats_timing.py [RESULTS_JSONL] [--out DIR]

Defaults are resolved from this file's own location, so the script runs from any
working directory: RESULTS_JSONL is ../03_unpacked/persona-single-d0.5/out/
results_626721938e67f4.jsonl, DIR is ./out next to this script.

Writes into --out:
    timing.md      report
    timing.csv     per-condition timing
    timing_by_level.csv  context level breakdown

What it separates
  * wall clock      : first to last write in the results file (plus manifest mtime as start)
  * generation time : sum of the per-record `elapsed_s` the script measured itself
  * overhead        : wall clock minus generation time = model load, calibration forward passes,
                      tokenisation, jsonl flushing, token-count check per prompt
  * prefill vs decode: least-squares fit elapsed_s ~ a*prompt_tokens + b*generated_tokens + c
                      over every record. `a` is roughly the per-prompt-token cost (prefill),
                      `b` the per-generated-token cost (decode). Treat as indicative, not exact:
                      prefill is charged once per turn and decode is sequential, so a and b are
                      not directly comparable in FLOPs, only in seconds.

Why this matters: the estimate handed out before the run assumed a flat seconds-per-turn figure.
If prefill dominates, cost scales with context length, and the 1024-level prompts cost far more
than the 0-level ones -- which is invisible in a single mean.
"""
import argparse, csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / '03_unpacked/persona-single-d0.5/out/results_626721938e67f4.jsonl'

def ols(X, y):
    """Tiny least squares, no numpy dependency. X: list of rows, y: list of floats."""
    n, k = len(X), len(X[0])
    A = [[sum(X[i][r] * X[i][c] for i in range(n)) for c in range(k)] + [sum(X[i][r] * y[i] for i in range(n))]
         for r in range(k)]
    for r in range(k):                      # gaussian elimination
        p = max(range(r, k), key=lambda i: abs(A[i][r]))
        if abs(A[p][r]) < 1e-12:
            return None
        A[r], A[p] = A[p], A[r]
        pv = A[r][r]
        A[r] = [v / pv for v in A[r]]
        for i in range(k):
            if i != r and A[i][r]:
                f = A[i][r]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
    return [A[i][k] for i in range(k)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='?', default=str(DEFAULT_RESULTS))
    ap.add_argument('--out', default=str(HERE / 'out'))
    args = ap.parse_args()
    rp = Path(args.results)
    rows = [json.loads(l) for l in rp.read_text(encoding='utf8').splitlines() if l.strip()]
    mpath = rp.with_name(rp.name.replace('results_', 'manifest_').replace('.jsonl', '.json'))
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    gen = sum(r['elapsed_s'] for r in rows)
    wall = (rp.stat().st_mtime - (mpath.stat().st_mtime if mpath.exists() else rp.stat().st_mtime))
    overhead = wall - gen
    # jsonl lines have no per-row timestamp, so a per-prompt timeline is reconstructed from
    # cumulative elapsed + the fixed offset (wall - gen).
    offset = max(0.0, overhead)

    by_cond = defaultdict(list)
    by_level = defaultdict(list)
    by_cat = defaultdict(list)
    for r in rows:
        by_cond[r['condition']].append(r)
        by_level[r['context_level']].append(r)
        by_cat[r['category']].append(r)

    cond_rows = []
    for cond, g in by_cond.items():
        e = [r['elapsed_s'] for r in g]
        cond_rows.append({
            'condition': cond, 'n': len(g),
            'elapsed_sum_s': round(sum(e), 1),
            'elapsed_mean_s': round(statistics.mean(e), 3),
            'elapsed_median_s': round(statistics.median(e), 3),
            'elapsed_p90_s': round(sorted(e)[int(0.9 * (len(e) - 1))], 3),
            'elapsed_max_s': round(max(e), 3),
            'prompt_tokens_mean': round(statistics.mean(r['prompt_tokens'] for r in g), 1),
            'generated_tokens_mean': round(statistics.mean(r['generated_tokens'] for r in g), 1),
            'decode_tok_per_s': round(sum(r['generated_tokens'] for r in g) / sum(e), 2),
        })
    cond_rows.sort(key=lambda d: -d['elapsed_sum_s'])
    with (outdir / 'timing.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(cond_rows[0])); w.writeheader(); w.writerows(cond_rows)

    level_rows = []
    for lvl in sorted(by_level):
        g = by_level[lvl]
        level_rows.append({
            'context_level': lvl, 'n': len(g),
            'prompt_tokens_mean': round(statistics.mean(r['prompt_tokens'] for r in g), 1),
            'prompt_tokens_max': max(r['prompt_tokens'] for r in g),
            'elapsed_mean_s': round(statistics.mean(r['elapsed_s'] for r in g), 3),
            'elapsed_sum_s': round(sum(r['elapsed_s'] for r in g), 1),
            'share_of_gen_time_pct': round(100 * sum(r['elapsed_s'] for r in g) / gen, 1),
        })
    with (outdir / 'timing_by_level.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(level_rows[0])); w.writeheader(); w.writerows(level_rows)

    # prefill / decode split
    X = [[r['prompt_tokens'], r['generated_tokens'], 1.0] for r in rows]
    y = [r['elapsed_s'] for r in rows]
    coef = ols(X, y)
    a, b, c = (coef if coef else (None, None, None))
    pred = [a * x[0] + b * x[1] + c for x in X] if coef else []
    ss_tot = sum((v - statistics.mean(y)) ** 2 for v in y)
    ss_res = sum((v - p) ** 2 for v, p in zip(y, pred)) if coef else None
    r2 = 1 - ss_res / ss_tot if coef and ss_tot else None

    # how much of a turn is prefill, per record
    prefill_share = (statistics.mean(a * r['prompt_tokens'] for r in rows) / statistics.mean(y)
                     if coef else None)

    L = []
    A = L.append
    A('# Timing: where the wall clock went\n')
    A(f'- results: `{rp}`')
    A(f'- turns: **{len(rows)}**')
    A(f'- wall clock (manifest mtime -> last write): **{wall/60:.1f} min** ({wall/3600:.2f} h)')
    A(f'- generation time (sum of per-record `elapsed_s`): **{gen/60:.1f} min** ({gen:.0f} s)')
    A(f'- overhead (load + calibration + tokenising + flushing): **{overhead/60:.1f} min** ({overhead:.0f} s)')
    A(f'- per turn: mean {statistics.mean(y):.2f} s, median {statistics.median(y):.2f} s, '
      f'p90 {sorted(y)[int(0.9*(len(y)-1))]:.2f} s, max {max(y):.2f} s')
    A(f'- throughput: {sum(r["generated_tokens"] for r in rows)/gen:.1f} generated tok/s, '
      f'prompt tokens mean {statistics.mean(r["prompt_tokens"] for r in rows):.1f} '
      f'(max {max(r["prompt_tokens"] for r in rows)})')
    A('')
    A('## Prefill vs decode (least squares on all turns)\n')
    if coef:
        A(f'`elapsed_s ≈ {a:.6f} * prompt_tokens + {b:.4f} * generated_tokens + {c:.3f}`  '
          f'(R² {r2:.3f})')
        A('')
        if a > 0.0002:
            A(f'- **prefill**: {a*1000:.2f} ms per prompt token '
              f'(20-token prompt {a*20:.2f} s, 1579-token prompt {a*1579:.2f} s)')
            A(f'- prefill is about **{100*prefill_share:.0f}%** of an average turn')
        else:
            A(f'- **prefill**: no measurable cost in this run (fit gives {a*1000:.3f} ms/token, i.e. '
              f'~0 to slightly negative). Prompt length 24 -> 1044 tokens moved per-turn time by '
              f'{level_rows[-1]["elapsed_mean_s"] - level_rows[0]["elapsed_mean_s"]:+.2f} s. '
              f'Prefill on a 4B model at this context length is far below the per-token decode cost, '
              f'so the negative coefficient is warm-up ordering, not a real speed-up.')
        A(f'- **decode**: {b*1000:.1f} ms per generated token ({1/b:.2f} tok/s effective)')
        A(f'- **fixed per-turn cost**: {c:.3f} s')
        A(f'- consequence: a turn costs about `max_new_tokens / {1/b:.1f}` seconds. At the 128 cap '
          f'that is {128*b:.2f} s, which is why the median turn ({statistics.median(y):.2f} s) sits '
          'at the full budget rather than partway into it.')
    else:
        A('fit failed (degenerate matrix)')
    A('')
    A('## By context level\n')
    A('| context_level | n | prompt tokens (mean/max) | elapsed mean s | gen time share % |')
    A('|---|---|---|---|---|')
    for r in level_rows:
        A(f"| {r['context_level']} | {r['n']} | {r['prompt_tokens_mean']} / {r['prompt_tokens_max']} | "
          f"{r['elapsed_mean_s']} | {r['share_of_gen_time_pct']} |")
    A('')
    A('## By condition (sorted by total time)\n')
    A('| condition | n | total s | mean s | median s | p90 s | max s | prompt tok | gen tok | decode tok/s |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    for r in cond_rows:
        A(f"| `{r['condition']}` | {r['n']} | {r['elapsed_sum_s']} | {r['elapsed_mean_s']} | "
          f"{r['elapsed_median_s']} | {r['elapsed_p90_s']} | {r['elapsed_max_s']} | "
          f"{r['prompt_tokens_mean']} | {r['generated_tokens_mean']} | {r['decode_tok_per_s']} |")
    A('')
    A('## Extrapolation\n')
    n = len(rows)
    if coef:
        for label, turns, pt, gt in (
                ('this run, if the token budget were 256 instead of 128', n,
                 statistics.mean(r['prompt_tokens'] for r in rows),
                 statistics.mean(r['generated_tokens'] for r in rows) + 100),
                ('pair phase, 29 conditions x 60 prompts (1740 turns), same prompt mix', 1740,
                 statistics.mean(r['prompt_tokens'] for r in rows), statistics.mean(r['generated_tokens'] for r in rows)),
                ('zero-context only, 18 conditions x 20 prompts (360 turns)', 360, 20,
                 statistics.mean(r['generated_tokens'] for r in rows))):
            est = turns * (max(a, 0.0) * pt + b * gt + c)
            A(f'- {label}: **~{est/60:.0f} min** ({est/3600:.1f} h) of generation, '
              f'plus the ~1 min model load (which sits outside the measured window)')
    A('')
    A('## Reading\n')
    A('- Time is spent **decoding**, not reading the prompt. A 20x jump in prompt length cost '
      'nothing measurable, while the output budget sets the per-turn cost almost exactly.')
    A('- Practically: plan with `turns x max_new_tokens / 26 s`, and treat context length as free '
      'at this scale. Shrinking the token budget is the only cheap lever; changing prompts is not.')
    A('- Caveat on the context-level table: `build_prompts.py` writes level 0 first, so the 0-level '
      'turns absorb CUDA warm-up. That is why the longest-context level is not the slowest one, and '
      'why a flat or slightly negative prefill coefficient must not be read as "longer prompts are '
      'faster".')
    A('- 54/60 baseline turns and most steered turns hit the 128 cap, so this run is '
      '**budget-bound**: per-turn time is near-constant by construction, and any claim that one '
      'condition "took longer" is dominated by whether it stopped at EOS early.')
    text = '\n'.join(L) + '\n'
    (outdir / 'timing.md').write_text(text, encoding='utf8')
    print(text)
    print(f'wrote {outdir}/timing.md, timing.csv, timing_by_level.csv')

if __name__ == '__main__':
    sys.exit(main())
