#!/usr/bin/env python3
"""Matched-prompt Taylor validation: gradient and central finite difference on SAME input.
This is not discovery->heldout transfer. Restricted to discovery contexts.
"""
import argparse,json,statistics
from pathlib import Path
import torch
from audit import json_digest
from protocol import require_plan,rows_for,stage_plan,require_discovery_seal,seal_discovery
from engine import load_model,column_norms,write_json
from discover import readout_gradients
from validate import lexical_score,load_target


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='out/snam_v1')
    p.add_argument('--traits',default='hostility')
    p.add_argument('--compilers',default='lexical,distribution,counterfactual')
    p.add_argument('--top-k',type=int,default=4)
    p.add_argument('--eps-values',default='0.02,0.1,0.5',
                   help='Comma separated finite-difference doses to diagnose BF16 quantization and nonlinearity')
    p.add_argument('--discovery-probe-index',type=int,default=0)
    p.add_argument('--reuse-fingerprint',action='store_true')
    args=p.parse_args()
    epses=[float(x) for x in args.eps_values.split(',')]
    if not epses or any(e<=0 for e in epses):raise ValueError('--eps-values must be positive')
    plan=require_plan(args.run,reuse=args.reuse_fingerprint)
    disc=Path(args.run)/'discovery'
    require_discovery_seal(disc)
    dm=json.loads((disc/'manifest.json').read_text(encoding='utf8'))
    if 'jacobian' not in dm['parameters']['solvers']:
        raise RuntimeError('Calibration requires a Jacobian discovery run')
    st=stage_plan(args.run,'calibration',{'traits':args.traits,'compilers':args.compilers,
                'top_k':args.top_k,'eps_values':epses,'discovery_probe_index':args.discovery_probe_index,
                'discovery_manifest_sha256':json_digest(dm)})
    if (st/'SEAL.json').exists():
        require_discovery_seal(st);print('Calibration already sealed; no recomputation.');return
    tok,model,lm=load_model(plan['model_dir']);layers=dm['parameters']['layers']
    norms=column_norms(lm,layers);p50={l:1.0 for l in layers}
    outcomes=[]
    for trait in args.traits.split(','):
        rows=rows_for(plan,trait,'discovery',dm['parameters']['discovery_limit'])
        if not 0<=args.discovery_probe_index<len(rows):raise ValueError('No such discovery probe')
        r=rows[args.discovery_probe_index]
        for comp in args.compilers.split(','):
            target=load_target(disc/f'{trait}_{comp}_target.json')
            per=readout_gradients(tok,model,lm,layers,[r],{comp:target},p50,norms,'cuda:0',dm['parameters']['max_seq'])[comp]
            top=json.loads((disc/f'{trait}_{comp}_jacobian_top.json').read_text())[:args.top_k]
            for c in top:
                l=int(c['layer']);n=int(c['neuron']);g=float(per[l][0,n]);
                for eps in epses:
                    plus=[{'layer':l,'neuron':n,'dose':eps}]
                    minus=[{'layer':l,'neuron':n,'dose':-eps}]
                    pp=lexical_score(tok,model,lm,r['prompt'],target,plus,p50,norms,'cuda:0')
                    pm=lexical_score(tok,model,lm,r['prompt'],target,minus,p50,norms,'cuda:0')
                    fd=(pp-pm)/(2*eps)
                    outcomes.append({'trait':trait,'compiler':comp,'probe':r['id'],'layer':l,'neuron':n,
                                     'eps':eps,'autograd_derivative':g,'finite_difference':fd,
                                     'relative_error':abs(fd-g)/max(abs(g),abs(fd),1e-8),
                                     'sign_match':(fd*g>0)})
    write_json(st/'results.json',outcomes)
    seal_discovery(st)
    print(f'Matched-input calibration: {len(outcomes)} cells; no heldout observations used. Saved {st}')

if __name__=='__main__':main()
