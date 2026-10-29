#!/usr/bin/env python3
"""Pure CPU leave-one-probe-out stability and leakage-free heldout-gradient checks."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import torch
from audit import file_hash
from engine import write_json


def concat_layers(layer_dict, item):
    parts=[]; names=[]
    for l in sorted(layer_dict):
        x=layer_dict[l][item]
        parts.append(x)
        names.extend((int(l),n) for n in range(x.shape[-1]))
    return torch.cat(parts,dim=-1),names


def top_indices(scores,k):
    return set(torch.topk(scores.abs(),min(k,scores.numel()),sorted=False).indices.tolist())


def stable_stats(archive, ks=(8,64,256)):
    per,names=concat_layers(archive,'per_probe')
    if per.ndim!=2:raise ValueError('per_probe tensor must be [nprobes,nneurons]')
    p,n=per.shape
    if p<2:raise ValueError('At least two probes needed for LOO; preferably three or more')
    mean=per.mean(0)
    # Do not use a round-trip through the full-data ranking to choose LOO candidates.
    loo=[(per.sum(0)-per[i])/(p-1) for i in range(p)]
    summary={'num_probes':p,'num_coordinates':n,'warnings':[], 'top_k':{}}
    if p<3:summary['warnings'].append('Fewer than three probes; LOO only diagnostic.')
    for k in ks:
        k=min(k,n)
        full=top_indices(mean,k)
        sets=[top_indices(z,k) for z in loo]
        overlap=[len(full&z)/k for z in sets]
        jaccards=[len(a&b)/len(a|b) for a,b in itertools.combinations(sets,2)]
        # Out-of-fold sign consistency & effect: selected from train folds, measured on excluded probe.
        oof=[]; signhits=[]
        for i,z in enumerate(loo):
            ids=torch.tensor(sorted(sets[i]),dtype=torch.long)
            signs=torch.where(z[ids]>=0,1.,-1.)
            heldout=per[i,ids]
            oof.append(float((signs*heldout).mean().item()))
            signhits.append(float(((signs*heldout)>0).float().mean().item()))
        summary['top_k'][str(k)]={
            'overlap_full_LOO':overlap, 'mean_overlap':sum(overlap)/len(overlap),
            'pairwise_LOO_jaccard':jaccards,
            'out_of_fold_mean_signed_gradient':oof,
            'out_of_fold_positive_fraction':signhits}
    top=sorted(top_indices(mean,min(64,n)),key=lambda i:float(-abs(mean[i])))
    featured=[]
    for i in top:
        vec=per[:,i]
        abs_total=float(vec.abs().sum().item())
        dom=float(vec.abs().max().item()/abs_total) if abs_total else 0
        f={'layer':names[i][0],'neuron':names[i][1], 'mean_gradient':float(mean[i]),
           'dominance_max_abs_probe_fraction':dom,
           'probe_gradients':[float(x) for x in vec]}
        featured.append(f)
    summary['top64_probe_dominance']=featured
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scan-dir',default='out/scan_v2')
    ap.add_argument('--trait',default='hostility')
    ap.add_argument('--strict',action='store_true',help='Exit nonzero if LOO top-64 overlap below specified gate')
    ap.add_argument('--min-loo-overlap',type=float,default=0.10)
    args=ap.parse_args()
    sd=Path(args.scan_dir)
    archive=torch.load(sd/f'{args.trait}_all_gradients.pt',map_location='cpu',weights_only=True)
    summary=stable_stats(archive)
    write_json(sd/f'{args.trait}_stability.json',summary)
    result=[]
    result.append(f'# LOO stability: {args.trait} ({summary["num_probes"]} discovery probes)')
    result.append('This uses *only* saved discovery gradients; no GPU or generation.')
    for k,row in summary['top_k'].items():
        result.append(f'- Top-{k}: mean overlap with full set = {row["mean_overlap"]:.3f}; '
                      f'LOO oof sign fractions = {row["out_of_fold_positive_fraction"]}')
    result.extend(['', 'Low overlap is diagnostic, not proof that all candidates are noise.',
                   'High overlap is not evidence of semantic construct validity.',
                   ''])
    (sd/f'{args.trait}_stability.md').write_text('\n'.join(result),encoding='utf8')
    print('\n'.join(result))
    if args.strict and summary['top_k'].get('64',{}).get('mean_overlap',0)<args.min_loo_overlap:
        raise SystemExit('LOO stability gate failed. Reconsider probes before verification.')

if __name__=='__main__':main()
