#!/usr/bin/env python3
"""P-B (plan 2.41): does the DOSE depend on the LEVEL of the label?  Zero GPU, stdlib only.

THE QUESTION. "Given a natural-language trait" is only computable if traits live on a lattice rather
than being a folksonomy. neuronary's tags are a folksonomy (252 tags for 193 recipes, 64% singletons)
and they mix levels: some name WHAT the model claims to be, some name HOW it speaks, some name the
content, some name the device used. If the required dose tracks the level -- changing what you are
should cost more than changing how you sound -- then a level coordinate exists and "trait" can be a
point on a lattice. If it does not, neuronary is a validation set only.

FROZEN CRITERION (written 2026-10-30 before the numbers were computed; the tag->level map below is
frozen with it, and ambiguous tags are EXCLUDED rather than assigned):
  Primary: Mann-Whitney U, one-sided, aggregate budget sqrt(sum dose^2) of L1-only recipes > L2-only
           recipes, p <= 0.05.
  Reported regardless: medians, n, and the excluded fraction.
  Secondary (exploratory, no verdict): single-neuron-only recipes, and |dose| instead of aggregate.
BOUNDARY, stated up front: neuronary publishes SUCCESSES ONLY (1 failed reproduction recorded out of
193). There is no denominator, so a level effect here can be a selection effect. This is why the
criterion is about a within-corpus contrast, not about a rate.
"""
from __future__ import annotations
import json, math, sys
from statistics import median

L1 = {'persona', 'identity', 'self-description', 'character', 'superhero', 'gender', 'female',
      'male', 'animal', 'yandere', 'cat', 'human'}
L2 = {'style', 'funny', 'cute', 'silly', 'slang', 'voice', 'attitude', 'weird', 'word-choice',
      'tone', 'formal', 'casual', 'humor', 'subjective'}
EXCLUDED = {'topic', 'topic-bias', 'religion', 'food', 'biology', 'science', 'money', 'coffee',
            'mashup', 'combination', 'recipe', 'exploratory', 'reframe', 'single-neuron',
            'multi-neuron', 'catchphrase', 'joke', 'relativism', 'super'}


def budget(r):
    return math.sqrt(sum(float(n['dose']) ** 2 for n in r['neurons']))


def mwu_greater(a, b):
    """One-sided Mann-Whitney U (a > b), exact-ish normal approximation with tie correction."""
    n1, n2 = len(a), len(b)
    if n1 < 3 or n2 < 3:
        return None, None
    allv = [(x, 0) for x in a] + [(y, 1) for y in b]
    allv.sort()
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r1 = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    ties = {}
    for v, _ in allv:
        ties[v] = ties.get(v, 0) + 1
    var = (n1 * n2 / 12) * ((n1 + n2 + 1) - sum(t ** 3 - t for t in ties.values()) / ((n1 + n2) * (n1 + n2 - 1)))
    if var <= 0:
        return u1, None
    z = (u1 - mu) / math.sqrt(var)
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return u1, p


def main(path):
    recs = json.load(open(path))
    grpL1, grpL2 = [], []
    only, excl = 0, 0
    for r in recs:
        t = set(r.get('tags') or [])
        if not t:
            continue
        has1, has2 = bool(t & L1), bool(t & L2)
        if has1 and has2:
            continue
        if has1:
            grpL1.append(budget(r))
        elif has2:
            grpL2.append(budget(r))
        elif t & EXCLUDED:
            excl += 1
        else:
            only += 1
    u, p = mwu_greater(grpL1, grpL2)
    out = {
        'n_recipes': len(recs),
        'L1_only': {'n': len(grpL1), 'median_budget': round(median(grpL1), 3) if grpL1 else None,
                    'values': [round(x, 3) for x in sorted(grpL1)]},
        'L2_only': {'n': len(grpL2), 'median_budget': round(median(grpL2), 3) if grpL2 else None,
                    'values': [round(x, 3) for x in sorted(grpL2)]},
        'excluded_ambiguous_or_other': excl,
        'unclassified': only,
        'U': u, 'p_one_sided': p,
        'frozen_level_map': {'L1': sorted(L1), 'L2': sorted(L2), 'EXCLUDED': sorted(EXCLUDED)},
    }
    print(json.dumps({k: v for k, v in out.items() if k not in ('L1_only', 'L2_only')}, indent=1))
    print('L1', out['L1_only']['n'], out['L1_only']['median_budget'], out['L1_only']['values'])
    print('L2', out['L2_only']['n'], out['L2_only']['median_budget'], out['L2_only']['values'])
    return out


if __name__ == '__main__':
    out = main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/nry_all.json')
    json.dump(out, open('results/tag_level_dose.json', 'w'), indent=1)
