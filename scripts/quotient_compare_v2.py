#!/usr/bin/env python3
"""Compare position lists in the quotient space, per recipe, against a matched null.

Fixes the four defects the audit found in the first version: direct-sum cosine over ALL layers, a null matched to EACH list's own
per-layer counts and signed dose magnitudes, one comparison per recipe instead of summing across traits, and frozen L1/L2 strata.
Arithmetic lives in analysis/quotient.py; the rule that decides the verdict is `q3_verdict` there.
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
    ap = argparse.ArgumentParser()  # noqa: E501
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
    from analysis.quotient import report_list, q3_verdict
    for spec in a.lists.split(','):
        path, col = spec.split(':')
        items = read_positions(path, col)
        entry = report_list(items, recipe_v, Wn, layers, L1, L2, a.n_null)
        out['lists'][path.split('/')[-1]] = entry
        print(path.split('/')[-1], json.dumps(entry, default=str)[:400], flush=True)
    out['Q3_verdict'] = q3_verdict(out['lists'])
    print('Q3', out['Q3_verdict'], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, 'w'), indent=1, default=str)
    print('wrote', a.out, flush=True)


if __name__ == '__main__':
    main()
