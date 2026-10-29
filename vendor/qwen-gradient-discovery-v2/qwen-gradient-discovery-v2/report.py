#!/usr/bin/env python3
"""V2 heldout effect reports: distinguish gradient ranking from sign selection.

All numbers are descriptive; randomized coordinates, not individual prompts, are
units for the random baseline. Each candidate is aggregated within coordinates.
"""
from __future__ import annotations
import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from engine import read_jsonl, write_json


def ranks(xs):
    order=sorted(range(len(xs)), key=lambda i:xs[i]);out=[0.]*len(xs)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and xs[order[start]]==xs[order[end]]:end+=1
        avg=(start+end-1)/2+1
        for i in order[start:end]:out[i]=avg
        start=end
    return out


def pearson(x,y):
    if len(x)<3:return None
    mx,my=statistics.mean(x),statistics.mean(y)
    a=sum((u-mx)*(v-my) for u,v in zip(x,y))
    b=(sum((u-mx)**2 for u in x)*sum((v-my)**2 for v in y))**0.5
    return a/b if b else None


def summarize(arms,records,dose):
    observed={}
    for r in records:
        key=(r['probe_id'],r['coordinate'],int(r['physical_sign']))
        if key in observed:raise ValueError(f'Duplicate physical measurement: {key}')
        observed[key]=r['margin']
    pids=sorted({x[0] for x in observed})
    per_arm=[]
    for a in arms:
        effects=[];opposites=[];curvatures=[]
        for pid in pids:
            base=observed.get((pid,'baseline',0))
            plus=observed.get((pid,a['coordinate'],+1))
            minus=observed.get((pid,a['coordinate'],-1))
            if any(v is None for v in (base,plus,minus)):continue
            own=plus if a['sign']==+1 else minus
            opp=minus if a['sign']==+1 else plus
            effects.append(own-base)
            opposites.append(opp-base)
            curvatures.append((plus+minus-2*base)/dose**2)
        if not effects:continue
        pred=dose*a['predicted_gradient']*a['sign']
        per_arm.append({**a, 'n_heldout':len(effects),'predicted_delta_discovery':pred,
                        'observed_mean_direction':statistics.mean(effects),
                        'observed_mean_opposite':statistics.mean(opposites),
                        'observed_mean_curvature':statistics.mean(curvatures),
                        'observed_deltas':effects})
    return per_arm


def prediction_metrics(arms):
    if not arms:return {}
    # Metrics should not treat the two random views of the same coordinate as independent.
    x=[a['predicted_delta_discovery'] for a in arms]
    y=[a['observed_mean_direction'] for a in arms]
    slope=sum(a*b for a,b in zip(x,y))/sum(a*a for a in x) if sum(a*a for a in x)>1e-20 else None
    return {'n_distinct_coordinates':len({a['coordinate'] for a in arms}),
            'mae':statistics.mean(abs(a-b) for a,b in zip(x,y)),
            'rmse':(statistics.mean((a-b)**2 for a,b in zip(x,y)))**.5,
            'regression_slope_zero_intercept':slope,
            'spearman_signed':pearson(ranks(x),ranks(y)),
            'spearman_magnitude_to_observed':pearson(ranks([abs(v) for v in x]),ranks(y))}


def build_report(verify_dir:Path):
    meta=json.loads((verify_dir/'manifest.json').read_text(encoding='utf8'))
    if meta.get('schema')!='verification-v2':raise ValueError('Need v2 verification manifest')
    arms=meta['arms']
    raw=read_jsonl(verify_dir/'verify.jsonl')
    seen={(r['probe_id'],r['coordinate'],int(r['physical_sign'])) for r in raw}
    expected={(probe['id'],'baseline',0) for probe in meta['heldout_probes']}
    for probe in meta['heldout_probes']:
        for layer,neuron in meta['physical_coordinates']:
            for sign in (-1,+1):
                expected.add((probe['id'],f'L{layer}N{neuron}',sign))
    missing=expected-seen
    extra=seen-expected
    if missing or extra:
        raise RuntimeError(f'Cannot report incomplete/mixed run: missing={len(missing)}, extra={len(extra)}. '
                           'Resume verify.py first.')
    rows=summarize(arms,raw,meta['dose'])
    groups=collections.defaultdict(list)
    for x in rows:groups[x['group']].append(x)
    lines=['# Qwen gradient discovery v2 — Independent validation', '',
           '**What this tests:** teacher-forced log-probability contrasts, not full personality or real-world decisions.',
           '', 'Each random coordinate receives BOTH physical ± doses. We analyze two polarity views:',
           'its own discovery gradient sign (fair rank control) and its parent Top-K sign (sign-ablation control).',
           '', '| Group | Unique coordinates | Mean observed Δmargin | Mean predicted Δmargin |',
           '|---|---:|---:|---:|']
    report={'groups':{},'arm_effects':rows,'limitations':[
        'Heldout objective may share syntactic or semantic biases with discovery.',
        'Discovery gradients vs heldout finite differences mix Taylor error with transfer error.',
        'Random own vs parent sign are correlated views of the SAME physical random coordinates.',
        'No inferential p-values; repeated prompts on same candidate are not independent samples.',
    ]}
    for group,rr in sorted(groups.items()):
        n=len({r['coordinate'] for r in rr})
        avg=statistics.mean(x['observed_mean_direction'] for x in rr)
        pred=statistics.mean(x['predicted_delta_discovery'] for x in rr)
        lines.append(f'| {group} | {n} | {avg:+.6f} | {pred:+.6f} |')
        report['groups'][group]={'n_coordinates':n,'mean_observed':avg,'mean_predicted':pred,
                                  'prediction_metrics':prediction_metrics(rr)}
    lines+=['','## Heldout transfer calibration (discovery g → heldout ΔJ)']
    for group in ('gradient_top','random_own_sign','random_parent_sign'):
        if group in report['groups']:
            m=report['groups'][group]['prediction_metrics']
            lines.append(f'- {group}: MAE={m["mae"]:.6g}; RMSE={m["rmse"]:.6g}; '
                         f'Spearman signed={m["spearman_signed"]}; slope={m["regression_slope_zero_intercept"]}')
    byid={r['arm_id']:r for r in rows}
    paired=[]
    for r in groups.get('random_own_sign',[]):
        parent_id=r.get('matched_to')
        parent=byid.get(parent_id)
        counterpart=byid.get('P'+r['arm_id'][1:])
        if parent is None or counterpart is None:
            raise RuntimeError(f'Broken random matched-pair mapping for {r["arm_id"]}')
        paired.append({'random_arm':r['arm_id'],'parent_arm':parent_id,
                       'top_minus_random_own':parent['observed_mean_direction']-r['observed_mean_direction'],
                       'random_own_minus_parent':r['observed_mean_direction']-counterpart['observed_mean_direction'],
                       'same_sign_random':r['sign']==counterpart['sign']})
    report['matched_pairs']=paired
    if paired:
        lines+=['','## Paired contrasts (matching by Top-K parent layer)',
                f'- Top minus random OWN sign: mean={statistics.mean(x["top_minus_random_own"] for x in paired):+.6f}',
                f'- Random OWN minus parent sign: mean={statistics.mean(x["random_own_minus_parent"] for x in paired):+.6f}',
                f'- Pairs where both random signs happen to agree: {sum(x["same_sign_random"] for x in paired)}/{len(paired)}',
                'These are descriptive paired-coordinate effects; do not treat two random signs as independent samples.']
    lines+=['','## Per-coordinate results',
            '| Arm | Group | Coordinate | Sign | Pred. Δ | Obs. Δ | Obs. opposite Δ | Curvature |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    for a in rows:
        lines.append(f'| {a["arm_id"]} | {a["group"]} | {a["coordinate"]} | {a["sign"]:+d} | '
                     f'{a["predicted_delta_discovery"]:+.5f} | {a["observed_mean_direction"]:+.5f} | '
                     f'{a["observed_mean_opposite"]:+.5f} | {a["observed_mean_curvature"]:+.3f} |')
    lines+=['','## Interpretation','',
            'Ranking efficacy: compare gradient_top vs random_own_sign.',
            'Sign efficacy: compare the two random views on identical random coordinates.',
            'Calibration: discovery-gradient prediction is out-of-sample transfer, NOT same-input Taylor validation.',
            'See `calibrate.py` for same-input gradients and central finite differences.','']
    report_text='\n'.join(lines)
    write_json(verify_dir/'analysis.json',report)
    (verify_dir/'summary.md').write_text(report_text,encoding='utf8')
    return report_text


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',default='out/verify_v2')
    args=ap.parse_args()
    print(build_report(Path(args.verify)))

if __name__=='__main__':main()
