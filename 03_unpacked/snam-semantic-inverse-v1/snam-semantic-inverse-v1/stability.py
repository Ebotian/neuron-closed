#!/usr/bin/env python3
"""LOO on discovery gradients; CPU only. No inference or heldout data used."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path
import torch
from protocol import require_plan,require_discovery_seal
from engine import write_json
from inverse import top_coordinates


def loo_analysis(per_probe,top_k):
    layers=sorted(per_probe)
    n=per_probe[layers[0]].shape[0]
    if n<2:raise ValueError('LOO needs at least 2 discovery prompts (smoke with n=1 is not eligible)')
    if any(x.shape[0]!=n for x in per_probe.values()):raise ValueError('Misaligned prompt dimensions')
    full={l:v.mean(0) for l,v in per_probe.items()}
    top=top_coordinates(full,top_k)
    key={(r['layer'],r['neuron']):r['sign'] for r in top}
    size=len(key)
    rows=[]
    for i in range(n):
        leave={l:(v.sum(0)-v[i])/(n-1) for l,v in per_probe.items()}
        lo=top_coordinates(leave,top_k)
        selected={(r['layer'],r['neuron']):r['sign'] for r in lo}
        overlap=sum(k in selected for k in key)/size
        signed=sum(k in selected and selected[k]==s for k,s in key.items())/size
        # On the held-out discovery probe, are selected directions aligned with its gradient?
        sg=0.;hits=0
        for c in lo:
            g=float(per_probe[c['layer']][i,c['neuron']])
            sg+=c['sign']*g
            hits+=int(c['sign']*g>0)
        rows.append({'left_out_probe_index':i,'topk_retention':overlap,
                     'signed_topk_retention':signed,'loo_selected_sign_hit_on_left_out_probe':hits/len(lo),
                     'mean_signed_leftout_slope':sg/len(lo)})
    return {'n_discovery_probes':n,'top_k':size,'loo':rows,
            'mean_topk_retention':statistics.fmean(r['topk_retention'] for r in rows),
            'min_topk_retention':min(r['topk_retention'] for r in rows),
            'mean_leftout_sign_hit':statistics.fmean(r['loo_selected_sign_hit_on_left_out_probe'] for r in rows),
            'warning':'LOO is internal discovery reliability, NOT a heldout behavioral test.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='out/snam_v1')
    p.add_argument('--traits',default='hostility')
    p.add_argument('--compilers',default='lexical,distribution,counterfactual')
    p.add_argument('--top-k',type=int,default=64)
    args=p.parse_args();require_plan(args.run,model_check=False)
    stage=Path(args.run)/'discovery';require_discovery_seal(stage)
    for trait in args.traits.split(','):
        for comp in args.compilers.split(','):
            path=stage/f'{trait}_{comp}_jacobian_per_probe.pt'
            if not path.is_file():
                raise FileNotFoundError(f'No per-prompt Jacobian gradients: {path}; run discover.py first')
            per=torch.load(path,map_location='cpu',weights_only=True)
            z=loo_analysis(per,args.top_k)
            out=Path(args.run)/'analysis'
            out.mkdir(exist_ok=True)
            write_json(out/f'{trait}_{comp}_loo.json',z)
            print(f'{trait}:{comp} LOO@{args.top_k} mean={z["mean_topk_retention"]:.3f} min={z["min_topk_retention"]:.3f}',flush=True)

if __name__=='__main__':main()
