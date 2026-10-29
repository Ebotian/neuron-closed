#!/usr/bin/env python3
"""Heldout ranking/sign tests; physical random coordinates are the sampling unit.

Discovery-gradient predictions belong to out-of-sample TRANSFER, not same-input
Taylor calibration. See calibrate.py for the latter.
"""
from __future__ import annotations
import argparse
import collections
import itertools
import json
import math
import random
import statistics
from pathlib import Path
from engine import read_jsonl, write_json


def ranks(xs):
    order=sorted(range(len(xs)),key=lambda i:xs[i]); out=[0.]*len(xs)
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
    b=(sum((u-mx)**2 for u in x)*sum((v-my)**2 for v in y))**.5
    return a/b if b else None


def _quantile(vals,p):
    if not vals:return None
    seq=sorted(vals)
    pos=(len(seq)-1)*p
    low=math.floor(pos);high=math.ceil(pos)
    return seq[low]+(seq[high]-seq[low])*(pos-low)


def distribution(vals):
    vals=list(vals)
    if not vals:return {'n':0,'mean':None,'sd':None,'median':None,'q25':None,'q75':None,'iqr':None}
    q25=_quantile(vals,.25);q75=_quantile(vals,.75)
    return {'n':len(vals),'mean':statistics.mean(vals),
            'sd':statistics.stdev(vals) if len(vals)>1 else None,
            'median':statistics.median(vals),'q25':q25,'q75':q75,'iqr':q75-q25}


def exact_sign_test(vals,tolerance=1e-12):
    """Two-sided exact binomial sign test on nonzero paired differences."""
    wins=sum(v>tolerance for v in vals)
    losses=sum(v<-tolerance for v in vals)
    ties=len(vals)-wins-losses
    n=wins+losses
    if n==0:return {'wins':wins,'losses':losses,'ties':ties,'n_effective':n,'p_two_sided':1.}
    tail=sum(math.comb(n,k) for k in range(max(wins,losses),n+1))/2**n
    return {'wins':wins,'losses':losses,'ties':ties,'n_effective':n,
            'p_two_sided':min(1.,2*tail)}


def paired_signflip_test(vals,seed=1729):
    """Exact 2-sided paired sign-flip mean test for <=18 independent pairs.

    For larger samples a seeded Monte Carlo test is used with +1 correction.
    Valid under null exchangeability of signs within independent matched pairs.
    """
    vals=[float(v) for v in vals if abs(v)>1e-12]
    n=len(vals)
    if not n:return {'n':0,'p_two_sided':1.,'method':'degenerate'}
    observed=abs(sum(vals));tol=1e-12
    if n<=18:
        extreme=0
        for signs in itertools.product((-1,1),repeat=n):
            extreme+=abs(sum(v*s for v,s in zip(vals,signs)))>=observed-tol
        return {'n':n,'p_two_sided':extreme/(2**n),'method':'exact-sign-flip'}
    rng=random.Random(seed); count=0;n_draws=100000
    for _ in range(n_draws):
        count+=abs(sum(v*(1 if rng.getrandbits(1) else -1) for v in vals))>=observed-tol
    return {'n':n,'p_two_sided':(1+count)/(1+n_draws),'method':'seeded-monte-carlo',
            'draws':n_draws,'seed':seed}


def bootstrap_ci(vals,seed=1729,draws=10000):
    vals=list(vals)
    if len(vals)<2:return None
    rng=random.Random(seed);n=len(vals)
    estimates=[sum(vals[rng.randrange(n)] for _ in range(n))/n for _ in range(draws)]
    return [_quantile(estimates,.025),_quantile(estimates,.975)]


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
        per_arm.append({**a,'n_heldout':len(effects),'predicted_delta_discovery':pred,
                        'observed_mean_direction':statistics.mean(effects),
                        'observed_mean_opposite':statistics.mean(opposites),
                        'observed_mean_curvature':statistics.mean(curvatures),
                        'observed_deltas':effects})
    return per_arm


def transfer_metrics(arms):
    if not arms:return {}
    x=[a['predicted_delta_discovery'] for a in arms]
    y=[a['observed_mean_direction'] for a in arms]
    denom=sum(a*a for a in x)
    return {'n_distinct_coordinates':len({a['coordinate'] for a in arms}),
            'mae':statistics.mean(abs(a-b) for a,b in zip(x,y)),
            'rmse':statistics.mean((a-b)**2 for a,b in zip(x,y))**.5,
            'regression_slope_zero_intercept':sum(a*b for a,b in zip(x,y))/denom if denom>1e-20 else None,
            'spearman_signed':pearson(ranks(x),ranks(y))}


def build_report(verify_dir:Path):
    meta=json.loads((verify_dir/'manifest.json').read_text(encoding='utf8'))
    if meta.get('schema')!='verification-v2':raise ValueError('Need verification-v2 manifest')
    arms=meta['arms'];raw=read_jsonl(verify_dir/'verify.jsonl')
    keys=[(r['probe_id'],r['coordinate'],int(r['physical_sign'])) for r in raw]
    if len(keys)!=len(set(keys)):
        raise RuntimeError('Duplicate physical measurements; refuse to report')
    seen=set(keys)
    expected={(probe['id'],'baseline',0) for probe in meta['heldout_probes']}
    for probe in meta['heldout_probes']:
        for layer,neuron in meta['physical_coordinates']:
            for sign in (-1,+1):expected.add((probe['id'],f'L{layer}N{neuron}',sign))
    if expected!=seen:
        raise RuntimeError(f'Incomplete/mixed verification run: missing={len(expected-seen)},extra={len(seen-expected)}')
    rows=summarize(arms,raw,meta['dose']);groups=collections.defaultdict(list)
    for x in rows:groups[x['group']].append(x)
    lines=['# Gradient Discovery v3 — heldout ranking & sign tests', '',
           '**Outcome:** teacher-forced likelihood contrast, NOT real-world persona or capability.', '',
           '| Group | Coordinates | Mean observed Δ | SD (across coordinates) | Median | IQR |',
           '|---|---:|---:|---:|---:|---:|']
    report={'schema':'heldout-analysis-v3','groups':{},'arm_effects':rows,
            'limitations':['Inference units are unique sampled coordinates, not token/prompt observations.',
                           'Random own/parent signs are correlated views of identical coordinates.',
                           'Any repeated matched Top-K parents invalidate naive independent-pair inference.',
                           'Gradient predictions on discovery vs heldout are TRANSFER, not Taylor calibration.',
                           'Exact sign-flip assumes independent matched pairs and sign exchangeability.',
                           'Small n gives weak inferential power; choose follow-up before inspecting outcomes.']}
    for group,rr in sorted(groups.items()):
        values=[x['observed_mean_direction'] for x in rr]
        stats=distribution(values)
        report['groups'][group]={'n_coordinates':len({r['coordinate'] for r in rr}),
                                  'effect_distribution':stats,'transfer_diagnostic':transfer_metrics(rr)}
        fmt=lambda v: 'NA' if v is None else f'{v:+.6f}'
        lines.append(f'| {group} | {stats["n"]} | {fmt(stats["mean"])} | {fmt(stats["sd"])} | '
                     f'{fmt(stats["median"])} | {fmt(stats["iqr"])} |')
    byid={r['arm_id']:r for r in rows};paired=[]
    for r in groups.get('random_own_sign',[]):
        parent=byid.get(r.get('matched_to'))
        counterpart=byid.get('P'+r['arm_id'][1:])
        if parent is None or counterpart is None:raise RuntimeError('Broken random matched-pair mapping')
        paired.append({'random_arm':r['arm_id'],'parent_arm':r['matched_to'],
                       'top_minus_random_own':parent['observed_mean_direction']-r['observed_mean_direction'],
                       'random_own_minus_parent':r['observed_mean_direction']-counterpart['observed_mean_direction'],
                       'same_sign_random':r['sign']==counterpart['sign']})
    report['matched_pairs']=paired
    report['paired_inference']={}
    if paired:
        independent=len({x['parent_arm'] for x in paired})==len(paired)
        lines+=['','## Matched pair comparisons','',
                'Primary comparison: Top-K vs **same-layer random with OWN gradient sign**.',
                'Secondary diagnostic: random OWN sign vs random PARENT sign; same physical random coordinates.','',
                '| Comparison | Pairs | Mean paired Δ | SD | IQR | Wins:Losses:Ties | Exact sign p (2-sided) | Sign-flip p (2-sided) |',
                '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,key in [('ranking','top_minus_random_own'),('sign_diagnostic','random_own_minus_parent')]:
            values=[r[key] for r in paired]
            stats=distribution(values)
            sign=exact_sign_test(values);flip=paired_signflip_test(values) if independent else None
            report['paired_inference'][name]={'distribution':stats,'sign_test':sign,
                                               'signflip_test':flip,'bootstrap_mean_95pct':bootstrap_ci(values)
                                               if independent else None,
                                               'independent_parent_matches':independent}
            lines.append(f'| {name} | {len(values)} | {stats["mean"]:+.6f} | '
                         f'{stats["sd"] if stats["sd"] is not None else "NA"} | '
                         f'{stats["iqr"]:.6f} | {sign["wins"]}:{sign["losses"]}:{sign["ties"]} | '
                         f'{sign["p_two_sided"]:.6f} | '
                         f'{flip["p_two_sided"]:.6f}' if flip else
                         f'| {name} | {len(values)} | {stats["mean"]:+.6f} | NA | {stats["iqr"]:.6f} | '
                         f'{sign["wins"]}:{sign["losses"]}:{sign["ties"]} | descriptive only | NA (reused parents) |')
        lines+=['',f'Independent Top-K parent matches: **{independent}**.',
                'Two-sided 7/8 sign victories yield p=0.0703125; 8/8 yield p=0.0078125 (no ties).',
                'This is an exploratory unadjusted test, not a family-wise corrected confirmatory result.','']
    random_vals=[r['observed_mean_direction'] for r in groups.get('random_own_sign',[])]
    if random_vals:
        q95=_quantile(random_vals,.95)
        signal=[r['observed_mean_direction'] for r in groups.get('gradient_top',[])]
        report['random_noise_floor']={'random_effect_distribution':distribution(random_vals),
                                      'random_q95':q95,'top_fraction_exceeding_random_q95':
                                      sum(v>q95 for v in signal)/len(signal) if signal else None,
                                      'warning':'Empirical random floor from a small sample, not a universal noise threshold.'}
        lines+=['## Empirical noise floor', '',
                f'- Same-layer random OWN-sign: SD={distribution(random_vals)["sd"]}; '
                f'IQR={distribution(random_vals)["iqr"]:.6f}; empirical Q95={q95:+.6f}.',
                '- Random Q95 from eight arms is highly uncertain; do not use as a significance cutoff.', '']
    lines+=['## Out-of-sample gradient transfer (NOT calibration)','',
            'Discovery |g| predicts *heldout* Δmargin; this includes distribution shift and Taylor error.',
            'MAE, slope and ranks are retained in `analysis.json` under `transfer_diagnostic`, not in the headline.',
            'For **same-input Taylor calibration**, run `calibrate.py` and inspect its separate summary.', '',
            '## Per-coordinate descriptive effects','',
            '| Arm | Group | Coordinate | Sign | Observed Δ | Opposite Δ | Curvature |',
            '|---|---|---|---:|---:|---:|---:|']
    for a in rows:
        lines.append(f'| {a["arm_id"]} | {a["group"]} | {a["coordinate"]} | {a["sign"]:+d} | '
                     f'{a["observed_mean_direction"]:+.5f} | {a["observed_mean_opposite"]:+.5f} | '
                     f'{a["observed_mean_curvature"]:+.3f} |')
    text='\n'.join(lines)+'\n'
    write_json(verify_dir/'analysis.json',report)
    (verify_dir/'summary.md').write_text(text,encoding='utf8')
    return text


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--verify',default='out/verify_v3')
    args=ap.parse_args()
    print(build_report(Path(args.verify)))

if __name__=='__main__':main()
