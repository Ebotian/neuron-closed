#!/usr/bin/env python3
"""Descriptive statistics for one persona-matrix run. No significance claims, no ratings.

    python3 stats_single.py [RESULTS_JSONL] [--out DIR]

Defaults are resolved from this file's own location, so the script runs from any
working directory: RESULTS_JSONL is ../03_unpacked/persona-single-d0.5/out/
results_626721938e67f4.jsonl, DIR is ./out next to this script.

Writes into --out (default: ./out):
    summary.md        short human-readable report
    summary.json      the same numbers as data
    conditions.csv    one row per condition
    by_category.csv   changed-vs-baseline counts, condition x category
    by_context.csv    changed-vs-baseline counts, condition x context_level

What is measured, and what is deliberately not:
  * changed = response differs from the baseline of the same prompt (string inequality).
    It is a sensitivity measure, NOT a quality or trait measure.
  * degenerate = output contains a repeated 8-word/char window >= 3 times, or template
    leakage (user/assistant/<think>), or the answer is empty. Crude on purpose: it flags
    "this output is broken", which is what the earlier runs actually showed.
  * no semantic scoring. Trait/task/integrity scores belong to rubric.md and a blind rater.
"""
import argparse, csv, difflib, json, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / '03_unpacked/persona-single-d0.5/out/results_626721938e67f4.jsonl'

TEMPLATE = re.compile(r'(^|\n)\s*(user|assistant|system|human)\s*$|<think>|</think>', re.M | re.I)

def norm(s):
    return ' '.join(s.split())

def repeat_score(s, n=8):
    """Largest count of any repeated n-gram (by characters for CJK, words for latin)."""
    if not s:
        return 0
    toks = list(s) if len(re.findall(r'[\u4e00-\u9fff]', s)) > 0.3 * len(s) else s.split()
    if len(toks) < n:
        return 0
    grams = Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))
    return max(grams.values())

def is_degenerate(resp, n=8):
    return (repeat_score(resp, n) >= 3) or bool(TEMPLATE.search(resp)) or not resp.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='?', default=str(DEFAULT_RESULTS))
    ap.add_argument('--out', default=str(HERE / 'out'))
    args = ap.parse_args()
    rp = Path(args.results)
    rows = [json.loads(l) for l in rp.read_text(encoding='utf8').splitlines() if l.strip()]
    mpath = rp.with_name(rp.name.replace('results_', 'manifest_').replace('.jsonl', '.json'))
    man = json.loads(mpath.read_text(encoding='utf8')) if mpath.exists() else {}
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)

    base = {r['prompt_id']: r for r in rows if r['condition'] == 'baseline'}
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r['condition']].append(r)
    prompt_order = [r['prompt_id'] for r in rows if r['condition'] == 'baseline']
    cats = {r['prompt_id']: (r['category'], r['context_level']) for r in rows}

    # ---------- per condition ----------
    def common_prefix(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    cond_rows = []
    for cond, g in by_cond.items():
        cand = [r for r in g if r['prompt_id'] in base]
        changed = [r for r in cand if r['response'] != base[r['prompt_id']]['response']]
        sims = [difflib.SequenceMatcher(None, r['response'],
                                        base[r['prompt_id']]['response']).ratio() for r in cand]
        prefs = [common_prefix(r['response'], base[r['prompt_id']]['response']) for r in cand]
        deg = sum(is_degenerate(r['response']) for r in g)
        cond_rows.append({
            'condition': cond,
            'n': len(g),
            'changed': len(changed),
            'changed_pct': round(100 * len(changed) / max(1, len(cand)), 1),
            'truncated': sum(r['truncated'] for r in g),
            'degenerate': deg,
            'degenerate_pct': round(100 * deg / max(1, len(g)), 1),
            'tokens_mean': round(statistics.mean(r['generated_tokens'] for r in g), 1),
            'tokens_median': statistics.median(r['generated_tokens'] for r in g),
            'chars_mean': round(statistics.mean(len(r['response']) for r in g), 1),
            'sim_mean': round(statistics.mean(sims), 4) if sims else None,
            'prefix_mean': round(statistics.mean(prefs), 1) if prefs else None,
            'elapsed_mean_s': round(statistics.mean(r['elapsed_s'] for r in g), 2),
        })
    order = (['baseline'] + [c for c in by_cond if c == 'zero_hook'] +
             sorted([c for c in by_cond if c.startswith('single:')]) +
             sorted([c for c in by_cond if c.startswith('randdir:')]) +
             sorted([c for c in by_cond if c.startswith('negative:')]) +
             sorted([c for c in by_cond if c.startswith('pair:')]))
    cond_rows.sort(key=lambda d: order.index(d['condition']) if d['condition'] in order else 99)
    with (outdir / 'conditions.csv').open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=list(cond_rows[0]))
        w.writeheader(); w.writerows(cond_rows)

    # ---------- changed x category / context ----------
    def cross(keyidx):
        keys = list(dict.fromkeys(cats[p][keyidx] for p in prompt_order))
        tab = []
        for c in cond_rows:
            cond = c['condition']; g = by_cond[cond]
            row = {'condition': cond}
            for k in keys:
                cand = [r for r in g if r['prompt_id'] in base and cats[r['prompt_id']][keyidx] == k]
                row[k] = sum(r['response'] != base[r['prompt_id']]['response'] for r in cand)
            row['n_per_key'] = len([p for p in prompt_order if cats[p][keyidx] == keys[0]])
            tab.append(row)
        return keys, tab
    cat_keys, cat_tab = cross(0)
    ctx_keys, ctx_tab = cross(1)
    for name, keys, tab in (('by_category.csv', cat_keys, cat_tab), ('by_context.csv', ctx_keys, ctx_tab)):
        with (outdir / name).open('w', newline='', encoding='utf8') as f:
            w = csv.DictWriter(f, fieldnames=list(tab[0]))
            w.writeheader(); w.writerows(tab)

    # ---------- headline numbers ----------
    zh = by_cond.get('zero_hook', [])
    zh_match = sum(1 for r in zh if r['prompt_id'] in base and r['response'] == base[r['prompt_id']]['response'])
    singles = [c for c in cond_rows if c['condition'].startswith('single:')]
    rdirs = [c for c in cond_rows if c['condition'].startswith('randdir:')]
    base_tok = statistics.mean(r['generated_tokens'] for r in by_cond['baseline'])
    base_deg = sum(is_degenerate(r['response']) for r in by_cond['baseline'])

    lines = []
    A = lines.append
    A(f'# persona-matrix run {man.get("run_id", rp.stem)} — descriptive statistics\n')
    A(f'- results: `{rp}`')
    A(f'- rows {len(rows)} = {len(prompt_order)} prompts x {len(by_cond)} conditions')
    A(f'- dose **{man.get("dose")}** | phase {man.get("phase")} | package {man.get("package_version")} '
      f'| max_new_tokens {man.get("max_new_tokens")} | greedy {man.get("greedy")}')
    A(f'- p50: ' + ', '.join(f'L{k} {v:.4f}' for k, v in sorted(man.get('p50', {}).items(), key=lambda x: int(x[0]))))
    A('')
    A('> `changed` = differs from that prompt\'s own baseline (sensitivity, not quality). '
      '`degenerate` = repeated 8-gram x3 / template leakage / empty (crude brokenness flag).')
    A('')
    A('## Conditions\n')
    A('| condition | n | changed | % | sim | shared prefix | trunc | degen % | tok mean | tok med | chars | s/turn |')
    A('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for c in cond_rows:
        A(f"| `{c['condition']}` | {c['n']} | {c['changed']} | {c['changed_pct']} | {c['sim_mean']} | "
          f"{c['prefix_mean']} | {c['truncated']} | {c['degenerate_pct']} | {c['tokens_mean']} | "
          f"{c['tokens_median']} | {c['chars_mean']} | {c['elapsed_mean_s']} |")
    A('')
    A('## Headline\n')
    A(f'- **zero_hook == baseline**: {zh_match}/{len(zh)} exact (a no-op hook must not change the text; '
      f'this is the run\'s own validity check)')
    A(f'- **baseline output**: mean {base_tok:.1f} tok, degenerate {base_deg}/{len(by_cond["baseline"])}')
    if singles:
        top = sorted(singles, key=lambda c: -c['changed'])[0]
        low = sorted(singles, key=lambda c: c['changed'])[0]
        A(f'- **single nodes** (dose {man.get("dose")}): changed ranges {low["changed"]}–{top["changed"]}/60 '
          f'(min `{low["condition"]}`, max `{top["condition"]}`)')
        A(f'- single nodes, sorted by changed: ' + ', '.join(f'`{c["condition"].split(":",1)[1]}` {c["changed"]}'
                                                            for c in sorted(singles, key=lambda c: -c['changed'])))
    if singles and rdirs:
        s_sim = statistics.mean(c['sim_mean'] for c in singles)
        r_sim = statistics.mean(c['sim_mean'] for c in rdirs)
        A(f'- **similarity to baseline** (1.0 = identical): single nodes mean {s_sim:.3f}, '
          f'random directions mean {r_sim:.3f} -> ' +
          ('target nodes drift *less* than random directions' if s_sim > r_sim else
           'target nodes drift *more* than random directions'))
        A(f'- per-node similarity: ' + ', '.join(f'`{c["condition"].split(":",1)[1]}` {c["sim_mean"]}'
                                                 for c in sorted(singles, key=lambda c: -c['sim_mean'])))
        A(f'- per-randdir similarity: ' + ', '.join(f'`{c["condition"]}` {c["sim_mean"]}' for c in rdirs))
    if rdirs:
        A(f'- **random-direction controls**: changed ' +
          ', '.join(f'`{c["condition"]}` {c["changed"]}' for c in rdirs) +
          f' — this is the floor any trait-specific effect has to clear')
        rmax = max(c['changed'] for c in rdirs)
        above = [c for c in singles if c['changed'] > rmax]
        A(f'- single nodes above that floor ({rmax}): ' +
          (', '.join(f'`{c["condition"].split(":",1)[1]}`' for c in above) if above
           else '**none** — at this dose no node beats a norm-matched random direction'))
    A('')
    A('## Changed counts by category / by context level\n')
    A('| condition | ' + ' | '.join(cat_keys) + ' |')
    A('|---|' + '---|' * len(cat_keys))
    for row in cat_tab:
        A(f"| `{row['condition']}` | " + ' | '.join(str(row[k]) for k in cat_keys) + ' |')
    A('')
    A('| condition | ' + ' | '.join(f'ctx {k}' for k in ctx_keys) + ' |')
    A('|---|' + '---|' * len(ctx_keys))
    for row in ctx_tab:
        A(f"| `{row['condition']}` | " + ' | '.join(str(row[k]) for k in ctx_keys) + ' |')
    A('')
    A('## Caveats\n')
    A('- single dose only, so nothing here is a dose-response curve; the earlier runs are at dose 1.0 '
      'and cannot be compared row by row with this one.')
    A('- `changed` is text inequality under greedy decoding: it says the steering did something, '
      'not that it did the claimed thing.')
    A('- 6 prompts per category (3 for logic/arithmetic): category columns are counts out of 6, '
      'so a difference of 1–2 is noise-level and should not be read as a category effect.')
    A('- trait / task-correctness / structure scores require `rubric.md` + a blind rater; '
      '`analyze.py` exports the blinded sheet for that.')

    text = '\n'.join(lines) + '\n'
    (outdir / 'summary.md').write_text(text, encoding='utf8')
    summary = {
        'run_id': man.get('run_id'), 'dose': man.get('dose'), 'phase': man.get('phase'),
        'package_version': man.get('package_version'), 'rows': len(rows),
        'prompts': len(prompt_order), 'conditions': len(by_cond),
        'zero_hook_exact': f'{zh_match}/{len(zh)}',
        'baseline_tokens_mean': round(base_tok, 1),
        'baseline_degenerate': base_deg,
        'conditions_table': cond_rows,
        'changed_by_category': cat_tab, 'changed_by_context': ctx_tab,
        'method': {'changed': 'response != baseline of same prompt (greedy)',
                   'degenerate': 'repeated 8-gram >=3 OR template leakage OR empty'},
    }
    (outdir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    print(text)
    print(f'wrote {outdir}/summary.md, summary.json, conditions.csv, by_category.csv, by_context.csv')

if __name__ == '__main__':
    sys.exit(main())
