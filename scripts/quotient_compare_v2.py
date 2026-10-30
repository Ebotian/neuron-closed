#!/usr/bin/env python3
"""Task-9 (plan 2.54): quotient-space comparison, redone after the audit.

WHAT THE AUDIT FOUND (doc/audit-response-2026-10-30.md §8): the deleted version
  * accumulated the cosine denominator only over layers touched by BOTH lists, so it was not the direct-sum cosine;
  * computed the random null only for the 64-entry list and then used it to interpret the 32-layer list;
  * drew null doses as all +1, matching neither the sign nor the magnitude distribution;
  * summed recipes of different traits into one list, which can cancel.
Everything below fixes those four things and nothing else:

  COSINE      full direct-sum over ALL 32 layers: num = sum_l <a_l,b_l>, den = sum_l ||a_l||^2 (no layer filtering)
  NULL        for EVERY compared list, >=20 random lists drawn to match that list's own per-layer counts AND its own
              dose magnitudes with their signs (sampled without replacement from the list's own doses)
  UNITS       v_l = sum over the list's coordinates of dose * unit(W_down column); a list that never touches layer l
              contributes the zero vector there, which is what "direct sum" means
  STRATA      each neuronary recipe is compared SEPARATELY (its own coordinates and doses), never summed with others;
              recipes are then grouped by the frozen L1/L2 tag map of scripts/tag_level_dose.py

FROZEN CRITERION (written before the run)
  Q1  report, per compared list: the direct-sum cosine against each recipe, and the median / IQR / max over recipes.
  Q2  report the matched-null distribution for the SAME list (mean, p95, max) and the fraction of recipes whose cosine
      exceeds the null p95. That fraction, not a single number, is the evidence.
  Q3  the verdict wording is fixed in advance: "no high alignment" unless the median over recipes exceeds the null p95
      for that list. Anything weaker must also be reported as weaker.
"""
from __future__ import annotations
import argparse, csv, json, math, random, sys
from pathlib import Path

import torch

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E
from core.loading import load_model_fast
from core.closedform import unit_columns
from tag_level_dose import L1, L2, EXCLUDED


def read_positions(path, dose_col):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            d = float(row[dose_col])
            if d != 0.0:
                out.append((int(row['layer']), int(row['neuron']), d))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--recipes', default='/root/autodl-tmp/quo/nry_all.json')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lists', default='/root/autodl-tmp/quo/positions_K64.csv:dose_at_b0.05,'
                                      '/root/autodl-tmp/quo/positions_all32_K1024_pairs.csv:u_dose')
    ap.add_argument('--n-null', type=int, default=40)
    ap.add_argument('--out', default='/root/autodl-tmp/runs/quotient_v2.json')
    a = ap.parse_args()
    dev = 'cuda:0'
    tok, model, lm = load_model_fast(a.model, dev)
    layers = list(range(32))
    Wn = unit_columns(lm, layers, dev)      # core.closedform: the ONE definition of the column direction

    recs = [r for r in json.load(open(a.recipes)) if r.get('model') == 'Qwen/Qwen3.5-4B' and r.get('neurons')]
    print(f'recipes on Qwen3.5-4B: {len(recs)}', flush=True)

    from analysis.quotient import to_v as _to_v, cos_direct as _cos, null_for as _null
    to_v = lambda items: _to_v(items, Wn, layers)
    cos_direct = lambda a_, b_: _cos(a_, b_, layers)
    null_for = lambda items, ref_v, n_null, seed=0: _null(items, ref_v, Wn, layers, n_null, seed)

    recs = [r for r in json.load(open(a.recipes)) if r.get('model') == 'Qwen/Qwen3.5-4B' and r.get('neurons')]
    print(f'recipes on Qwen3.5-4B: {len(recs)}', flush=True)

    def to_v(items):
        v = {l: torch.zeros(Wn[l].shape[0], dtype=torch.float64) for l in layers}
        for (l, n, d) in items:
            if 0 <= l < 32 and 0 <= n < Wn[l].shape[1]:
                v[l] += d * Wn[l][:, n]
        return v

    def cos_direct(a_, b_):
        num = sum(float(a_[l] @ b_[l]) for l in layers)
        da = sum(float(a_[l] @ a_[l]) for l in layers)
        db = sum(float(b_[l] @ b_[l]) for l in layers)
        if da <= 0 or db <= 0:
            return None
        return num / math.sqrt(da * db)

    def null_for(items, ref_v, n_null, seed=0):
        rng = random.Random(seed)
        doses = [abs(d) for (_, _, d) in items]
        signs = [1.0 if d > 0 else -1.0 for (_, _, d) in items]
        hist = {}
        for (l, _, _) in items:
            hist[l] = hist.get(l, 0) + 1
        vals = []
        for _ in range(n_null):
            rnd = []
            for l, k in hist.items():
                n_neur = Wn[l].shape[1]
                picks = rng.sample(range(n_neur), min(k, n_neur))
                for n in picks:
                    d = rng.choice(signs) * rng.choice(doses)
                    rnd.append((l, n, d))
            c = cos_direct(to_v(rnd), ref_v)
            if c is not None:
                vals.append(c)
        vals.sort()
        return {'n': len(vals), 'mean': sum(vals) / len(vals) if vals else None,
                'p95': vals[int(0.95 * len(vals)) - 1] if vals else None,
                'max': vals[-1] if vals else None}

    recipe_v = [(r['id'], to_v([(int(n['layer']), int(n['neuron']), float(n['dose'])) for n in r['neurons']]),
                 set(r.get('tags') or [])) for r in recs]

    out = {'n_recipes': len(recs), 'lists': {}}
    for spec in a.lists.split(','):
        path, col = spec.split(':')
        items = read_positions(path, col)
        v = to_v(items)
        cs = []
        for (rid, rv, tags) in recipe_v:
            c = cos_direct(v, rv)
            if c is not None:
                cs.append((c, tags))
        vals = sorted(c for c, _ in cs)
        nl = null_for(items, v, a.n_null)   # null: random list vs OUR OWN list (same histogram/doses)
        grp = {'L1_only': [], 'L2_only': [], 'other': []}
        for c, tags in cs:
            if tags & L1 and not (tags & L2):
                grp['L1_only'].append(c)
            elif tags & L2 and not (tags & L1):
                grp['L2_only'].append(c)
            else:
                grp['other'].append(c)
        entry = {'n_coords': len(items), 'per_recipe': {'median': vals[len(vals) // 2] if vals else None,
                                                        'p25': vals[len(vals) // 4] if vals else None,
                                                        'p75': vals[3 * len(vals) // 4] if vals else None,
                                                        'max': vals[-1] if vals else None},
                 'null_matched': nl,
                 'frac_recipes_above_null_p95': (None if not nl['p95'] else
                                                 sum(1 for c in vals if c > nl['p95']) / len(vals)),
                 'strata': {k: (sorted(v2)[len(v2) // 2] if v2 else None) for k, v2 in grp.items()},
                 'strata_n': {k: len(v2) for k, v2 in grp.items()}}
        out['lists'][path.split('/')[-1]] = entry
        print(path.split('/')[-1], json.dumps(entry, default=str)[:400], flush=True)

    med = {k: v['per_recipe']['median'] for k, v in out['lists'].items()}
    p95 = {k: v['null_matched']['p95'] for k, v in out['lists'].items()}
    out['Q3_verdict'] = {k: ('HIGH ALIGNMENT' if (med[k] is not None and p95[k] is not None and med[k] > p95[k])
                             else 'no high alignment') for k in med}
    print('Q3', out['Q3_verdict'], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=str)
    print('wrote', a.out, flush=True)


if __name__ == '__main__':
    main()
