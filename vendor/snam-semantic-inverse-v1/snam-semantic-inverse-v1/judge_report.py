#!/usr/bin/env python3
"""Unblind *after* independent manual scoring of free generation outputs.
Primary rows are paired by key/prompt (candidate results averaged inside prompt).
"""
from __future__ import annotations
import argparse,csv,collections,json,statistics
from pathlib import Path
from engine import write_json
from protocol import require_plan,require_discovery_seal
from analyze import sd,bootstrap_ci

RUBRIC=['trait_intensity_0to4','fluency_0to4','task_relevance_0to4','degeneration_0to4']


def parse_scores(path,identity):
    result={}
    with open(path,encoding='utf8',newline='') as f:
        for r in csv.DictReader(f):
            key=r['item_id']
            if key not in identity or key in result:raise ValueError(f'Unknown/duplicate blind id: {key}')
            scores={}
            for name in RUBRIC:
                try: val=float(r[name])
                except (TypeError,ValueError) as exc:
                    raise ValueError(f'Missing/non-numeric blinded rating: {key}, {name}') from exc
                if not 0<=val<=4:raise ValueError(f'Rating outside 0..4: {key}, {name}')
                scores[name]=val
            result[key]=scores
    if set(result)!=set(identity):raise ValueError('Not all blind items were scored')
    return result


def analyze_judges(mapping,score_records):
    by=collections.defaultdict(list)
    for row in mapping:
        item=row['item_id'];ratings=[r[item] for r in score_records]
        means={key:statistics.fmean(x[key] for x in ratings) for key in RUBRIC}
        by[(row['key'],row['prompt_id'],row['arm_id'])].append(means)
    base={}
    for (key,prompt,arm),a in by.items():
        if arm=='BASE':base[(key,prompt)]=a[0]
    out=collections.defaultdict(list)
    for (key,prompt,arm),a in by.items():
        if arm=='BASE':continue
        if (key,prompt) not in base:raise RuntimeError('Missing baseline in blind review')
        d={name:a[0][name]-base[(key,prompt)][name] for name in RUBRIC}
        out[(key,prompt)].append(d)
    by_method=collections.defaultdict(list)
    for (key,prompt),rows in out.items():
        avg={name:statistics.fmean(row[name] for row in rows) for name in RUBRIC}
        by_method[key].append(avg)
    return [{'key':key,'n_independent_prompts':len(rows),
             'n_judges':len(score_records),
             'metrics':{name:{'mean_delta':statistics.fmean(r[name] for r in rows),
                              'sd_across_prompts':sd([r[name] for r in rows]),
                              'bootstrap95_prompts':bootstrap_ci([r[name] for r in rows])}
                        for name in RUBRIC},
             'caution':'Human rating, not a guarantee of stable personality. Judges blinded only if PRIVATE mapping withheld.'}
            for key,rows in sorted(by_method.items())]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='out/snam_v1')
    p.add_argument('--ratings',required=True,nargs='+',help='Independent scored copies of blind_review.csv')
    args=p.parse_args();require_plan(args.run,model_check=False)
    d=Path(args.run)/'validation';require_discovery_seal(d);mapping=json.loads((d/'blind_mapping_PRIVATE.json').read_text(encoding='utf8'))
    identity={r['item_id'] for r in mapping}
    scores=[parse_scores(path,identity) for path in args.ratings]
    result=analyze_judges(mapping,scores)
    write_json(Path(args.run)/'analysis/human_review.json',result)
    print('Human rating report:',Path(args.run)/'analysis/human_review.json')

if __name__=='__main__':main()
