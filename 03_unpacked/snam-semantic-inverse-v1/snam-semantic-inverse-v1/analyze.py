#!/usr/bin/env python3
"""Analysis units are heldout prompts; candidates are repeated within prompt.
Bootstrap prompts, not token positions or independent-looking neuron coordinates.
All p-values labeled exploratory; manual freeform scores are never hallucinated.
"""
from __future__ import annotations
import argparse,collections,json,math,random,statistics
from pathlib import Path
from protocol import require_plan,require_discovery_seal
from engine import write_json


def mean(xs):return statistics.fmean(xs) if xs else None

def sd(xs):return statistics.stdev(xs) if len(xs)>1 else None

def iqr(xs):
    if len(xs)<2:return None
    z=sorted(xs)
    def quantile(q):
        t=(len(z)-1)*q;k=int(t);f=t-k
        return z[k] if k==len(z)-1 else z[k]*(1-f)+z[k+1]*f
    return quantile(.75)-quantile(.25)

def exact_sign_p(differences):
    """Two-sided exact binomial sign test; zeros are excluded. No iid claim inferred."""
    vals=[v for v in differences if abs(v)>1e-12]
    n=len(vals)
    if n==0:return None
    k=min(sum(v>0 for v in vals),sum(v<0 for v in vals))
    return min(1.,2*sum(math.comb(n,i) for i in range(k+1))/2**n)

def bootstrap_ci(x,seed=0,reps=3000):
    if len(x)<2:return None
    rng=random.Random(seed);b=[sum(x[rng.randrange(len(x))] for _ in x)/len(x) for _ in range(reps)]
    b.sort();return [b[int(.025*(reps-1))],b[int(.975*(reps-1))]]


def group_analysis(rows):
    baseline={(r['key'],r['test_trait'],r['prompt_id']):r for r in rows if r['group']=='baseline'}
    entries=[]
    for r in rows:
        if r['group']=='baseline':continue
        base=baseline.get((r['key'],r['test_trait'],r['prompt_id']))
        if base is None:raise RuntimeError(f'Missing baseline {r["key"]}:{r["prompt_id"]}')
        z=dict(r)
        z['delta_lexical']=r['lexical_score']-base['lexical_score']
        z['delta_transfer']=r['teacher_forced_margin']-base['teacher_forced_margin']
        # These are cross-context predictions, NOT same-input finite differences.
        z['predicted_lexical_transfer']=(r['slope_discovery']*r['interventions'][0]['dose']
                if r['slope_discovery'] is not None and len(r['interventions'])==1
                and r['solver']=='jacobian' else None)
        entries.append(z)
    groups=collections.defaultdict(list)
    for r in entries:groups[(r['key'],r['source_trait'],r['test_trait'],r['group'])].append(r)
    out=[]
    for (key,src,other,group),rec in sorted(groups.items()):
        prompts=collections.defaultdict(list)
        for r in rec:prompts[r['prompt_id']].append(r)
        # Average candidate responses *within each prompt*; don't inflate n with neurons.
        dlex=[mean([r['delta_lexical'] for r in rr]) for rr in prompts.values()]
        dtrans=[mean([r['delta_transfer'] for r in rr]) for rr in prompts.values()]
        record={'key':key,'source_trait':src,'test_trait':other,'group':group,
                'n_prompts':len(prompts),'n_candidate_prompt_cells':len(rec),
                'mean_delta_lexical':mean(dlex),'sd_delta_lexical':sd(dlex),
                'iqr_delta_lexical':iqr(dlex),'prompt_bootstrap95_lexical':bootstrap_ci(dlex),
                'mean_delta_transfer':mean(dtrans),'sd_delta_transfer':sd(dtrans),
                'iqr_delta_transfer':iqr(dtrans),'prompt_bootstrap95_transfer':bootstrap_ci(dtrans),
                'positive_prompt_fraction_transfer':mean([float(x>0) for x in dtrans]),
                'unit':'heldout prompt; candidate scores averaged within prompt'}
        if group=='random_own_sign':
            record['random_abs_transfer_q95_empirical']=sorted(abs(r['delta_transfer']) for r in rec)[int(.95*(len(rec)-1))]
        if group=='top' and any(x['predicted_lexical_transfer'] is not None for x in rec):
            record['cross_context_mae_not_calibration']=mean([abs(x['predicted_lexical_transfer']-x['delta_lexical'])
                                                    for x in rec if x['predicted_lexical_transfer'] is not None])
        out.append(record)
    # Fair paired rank comparison: same prompt AND matching Top/Random pair.
    pairs=[]
    idx=collections.defaultdict(dict)
    for r in entries:
        if r['group'] in ('top','random_own_sign'):
            idx[(r['key'],r['test_trait'],r['prompt_id'],r['matched_pair'])][r['group']]=r
    grouped=collections.defaultdict(list)
    for (key,test_trait,prompt,pair),v in idx.items():
        if 'top' in v and 'random_own_sign' in v:
            grouped[(key,test_trait,prompt)].append(v['top']['delta_transfer']-v['random_own_sign']['delta_transfer'])
    summary=collections.defaultdict(list)
    for (key,t,prompt),ds in grouped.items():
        summary[(key,t)].append({'prompt_id':prompt,'mean_top_minus_matched_random':mean(ds),'n_pairs':len(ds)})
    for (key,t),records in sorted(summary.items()):
        v=[r['mean_top_minus_matched_random'] for r in records]
        pairs.append({'key':key,'test_trait':t,'n_heldout_prompts':len(v),'prompt_level_effects':records,
                      'mean_top_minus_random':mean(v),'sd':sd(v),'iqr':iqr(v),
                      'bootstrap95':bootstrap_ci(v), 'exploratory_two_sided_exact_sign_p':exact_sign_p(v),
                      'inference_warning':'Prompt templates may be dependent; no confirmatory p-value, and multiple comparisons exist.'})
    interactions=[]
    grouped_inter=collections.defaultdict(dict)
    for r in rows:
        combo=r.get('combo_id')
        if combo is None:continue
        ckey=(r['key'],r['test_trait'],r['prompt_id'],combo)
        grouped_inter[ckey][r['arm_id']]=r
    per_combo=collections.defaultdict(list)
    for (key,trait,prompt,combo),by_arm in grouped_inter.items():
        combo_row=by_arm.get(combo)
        a=by_arm.get(combo+'P0');b=by_arm.get(combo+'P1')
        base=baseline.get((key,trait,prompt))
        if all(x is not None for x in (combo_row,a,b,base)):
            per_combo[(key,trait,combo)].append({
                'prompt_id':prompt,
                'interaction_lexical':combo_row['lexical_score']-a['lexical_score']-b['lexical_score']+base['lexical_score'],
                'interaction_transfer':combo_row['teacher_forced_margin']-a['teacher_forced_margin']-b['teacher_forced_margin']+base['teacher_forced_margin']})
    for (key,trait,combo),recs in sorted(per_combo.items()):
        interactions.append({'key':key,'test_trait':trait,'combo_id':combo,
             'n_prompts':len(recs),'mean_interaction_lexical':mean([r['interaction_lexical'] for r in recs]),
             'mean_interaction_transfer':mean([r['interaction_transfer'] for r in recs]),
             'sd_interaction_transfer':sd([r['interaction_transfer'] for r in recs]),
             'definition':'J(A+B)-J(A)-J(B)+J(0), identical signed pair component doses; exploratory'})
    return out,pairs,interactions


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='out/snam_v1')
    p.add_argument('--reuse-fingerprint',action='store_true')
    args=p.parse_args();require_plan(args.run,reuse=args.reuse_fingerprint,model_check=False)
    require_discovery_seal(Path(args.run)/'validation')
    path=Path(args.run)/'validation/cells'
    if not path.exists():raise FileNotFoundError('No validation results: run validate.py first')
    rows=[json.loads(f.read_text(encoding='utf8')) for f in sorted(path.glob('*.json'))]
    if not rows:raise ValueError('No result cells')
    report,pairs,interactions=group_analysis(rows)
    out=Path(args.run)/'analysis';out.mkdir(exist_ok=True)
    write_json(out/'by_group.json',report)
    write_json(out/'paired_top_vs_random.json',pairs)
    write_json(out/'pair_interactions.json',interactions)
    lines=['# SNAM analysis — exploratory, not a claim of stable personality','',
           'Unit: heldout prompt, candidate scores averaged within prompt. All targets are lexical proxies or author-written forced contrasts.','',
           '| method | tested trait | group | n prompts | Δ forced margin | SD | Δ lexical |',
           '|---|---|---|---:|---:|---:|---:|']
    for r in report:
        f=lambda v:'—' if v is None else f'{v:+.4g}'
        lines.append(f'| {r["key"]} | {r["test_trait"]} | {r["group"]} | {r["n_prompts"]} | '+
                     f'{f(r["mean_delta_transfer"])} | {f(r["sd_delta_transfer"])} | {f(r["mean_delta_lexical"])} |')
    lines.extend(['','## Matched Top vs random (own sign)','',
         '| method | tested trait | n prompts | mean paired Δ | SD | exploratory sign p |',
         '|---|---|---:|---:|---:|---:|'])
    for r in pairs:
        f=lambda v:'—' if v is None else f'{v:.4g}'
        lines.append(f'| {r["key"]} | {r["test_trait"]} | {r["n_heldout_prompts"]} | '+
                     f'{f(r["mean_top_minus_random"])} | {f(r["sd"])} | {f(r["exploratory_two_sided_exact_sign_p"])} |')
    lines+=['','## Interpretation limits','',
            '- Weight-only scores are *proxy units*, not calibrated behavioral log-probability changes.',
            '- Jacobian observed-vs-predicted errors on heldout are **transfer errors**, not same-prompt calibration.',
            '- Matched same-prompt finite differences belong exclusively to `calibrate.py`.',
            '- n refers to unique prompts, not tested coordinates, polarity views, tokens or seed variants.',
            '- No claim of independent prompts if contexts share fixed templates. Sign p values are diagnostic.',
            '- Pair non-additivity is computed from exact component doses in `pair_interactions.json`.',
            '- Without completed blinded freeform ratings, no claim of a visibly altered writing style.']
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Report:',out/'SUMMARY.md')

if __name__=='__main__':main()
