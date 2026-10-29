#!/usr/bin/env python3
"""Summaries from verify / compose WITHOUT external judges or keyword post-selection."""
import argparse, json, collections, statistics
from pathlib import Path
from engine import read_jsonl


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--verify',default='out/verify')
    p.add_argument('--compose',default=None)
    a=p.parse_args()
    v=Path(a.verify)
    records=read_jsonl(v/'verify.jsonl')
    by_probe=collections.defaultdict(dict)
    for r in records:by_probe[r['probe_id']][(r['key'],r['arm'])]=r
    agg=collections.defaultdict(list)
    for pid,d in by_probe.items():
        base=d.get(('baseline','baseline'))
        if base is None:continue
        for (key,arm),r in d.items():
            if arm not in ('direction','opposite'):continue
            agg[(key,r['source'],arm)].append(r['margin']-base['margin'])
    lines=['# White-box steering discovery: held-out validation','',
           'Each mean is Δ[logP(preferred continuation) − logP(dispreferred continuation)]',
           'relative to the identical heldout prompt with no intervention.','',
           '| Candidate | Source | Arm | Heldouts | Mean Δmargin |','|---|---|---|---:|---:|']
    for (key,source,arm),xs in sorted(agg.items(),key=lambda z: (z[0][2]!='direction',
                                                       -statistics.mean(z[1]))):
        lines.append(f'| {key} | {source} | {arm} | {len(xs)} | {statistics.mean(xs):+.5f} |')
    directional=[statistics.mean(xs) for (key,source,arm),xs in agg.items() if arm=='direction' and source=='gradient']
    random=[statistics.mean(xs) for (key,source,arm),xs in agg.items() if arm=='direction' and source=='random_matched']
    lines+=['',f'Gradient candidates: {len(directional)}; random controls: {len(random)}.',
             f'Gradient mean Δmargin: {statistics.mean(directional):+.5f}' if directional else 'No gradient effects.',
             f'Random mean Δmargin: {statistics.mean(random):+.5f}' if random else 'No random effects.',
             '', 'These are descriptive estimates, not calibrated p-values or independent samples.',
             'Heldout contrasts were written by humans and are not zero-information / unsupervised discovery.',
             'For full persona claims, separately rate generated responses and task correctness.']
    if a.compose and (Path(a.compose)/'interactions.jsonl').exists():
        interactions=read_jsonl(Path(a.compose)/'interactions.jsonl')
        lines+=['','## Pairwise interaction on heldout contrasts','',
                'I_AB=J(A+B)-J(A)-J(B)+J(0). Nonzero means functional non-additivity.',
                '| i | j | Mean I | n |','|---:|---:|---:|---:|']
        pairs=collections.defaultdict(list)
        for x in interactions:pairs[(x['i'],x['j'])].append(x['interaction'])
        for (i,j),xs in sorted(pairs.items()):
            lines.append(f'| {i} | {j} | {statistics.mean(xs):+.5f} | {len(xs)} |')
    text='\n'.join(lines)+'\n'
    print(text)
    (v/'summary.md').write_text(text,encoding='utf8')

if __name__=='__main__':main()
