#!/usr/bin/env python3
"""Factorial compiler (3) x solver (2) for Qwen3.5-4B.

Weight-only: exact matrix projection but only an UNCALIBRATED proxy across layers.
Jacobian: true gradient for fixed next-token lexical objective on discovery prompts.
Never consult heldout during candidate selection.
"""
from __future__ import annotations
import argparse,gc,json,os,time,traceback
from pathlib import Path
import torch
from audit import guarded_manifest
from engine import (load_model,parse_layers,column_norms,
                    prompt_prefix,capture_mlp_inputs,write_json)
from protocol import ROOT,require_plan,rows_for,stage_plan,seal_discovery,require_discovery_seal
from semantic import compile_target,readout_vector,next_token_objective,canonical_target,validate_target
from inverse import score_weights,top_coordinates,combinations


def readout_gradients(tokenizer,model,lm,layers,prompts,targets,p50,norms,device,max_seq):
    """One forward per prompt, multiple VJPs on shared retained autograd graph.
    Each intervention is broadcast to ALL token positions, so gradients SUM over tokens.
    """
    collected={key:[] for key in targets}
    order=list(targets)
    for row in prompts:
        ids=prompt_prefix(tokenizer,row['prompt'])
        if len(ids)>max_seq:raise ValueError(f'Too long discovery prompt {row["id"]}: {len(ids)}')
        inp=torch.tensor([ids],device=device,dtype=torch.long)
        with torch.enable_grad(),capture_mlp_inputs(lm,layers) as captures:
            logits=model(input_ids=inp,use_cache=False).logits[0,-1,:]
            act=[captures[l] for l in layers]
            for j,key in enumerate(order):
                obj=next_token_objective(logits,targets[key])
                grads=torch.autograd.grad(obj,act,retain_graph=j<len(order)-1)
                record={l:g.detach().float().sum(dim=tuple(range(g.ndim-1))).cpu()
                        *(p50[l]/norms[l].clamp(min=1e-12)) for l,g in zip(layers,grads)}
                if any(not torch.isfinite(r).all().item() for r in record.values()):
                    raise FloatingPointError(f'Non-finite gradient at {row["id"]}, {key}')
                collected[key].append(record)
        del logits
    return {key:{l:torch.stack([r[l] for r in rows]) for l in layers}
            for key,rows in collected.items()}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='out/snam_v1')
    p.add_argument('--traits',default='hostility')
    p.add_argument('--layers',default='13-14')
    p.add_argument('--compilers',default='lexical,distribution,counterfactual')
    p.add_argument('--solvers',default='weight,jacobian')
    p.add_argument('--discovery-limit',type=int,default=3)
    p.add_argument('--top-k',type=int,default=64)
    p.add_argument('--distribution-top',type=int,default=256)
    p.add_argument('--temperature',type=float,default=.06)
    p.add_argument('--max-seq',type=int,default=384)
    p.add_argument('--budget',type=float,default=.5)
    p.add_argument('--target-delta',type=float,default=.1)
    p.add_argument('--reuse-fingerprint',action='store_true')
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for the real Qwen experiment')
    plan=require_plan(args.run,reuse=args.reuse_fingerprint)
    allowed={'lexical','distribution','counterfactual'}; comps=args.compilers.split(',')
    solvers=args.solvers.split(',')
    if not set(comps)<=allowed or not set(solvers)<={'weight','jacobian'}:raise ValueError('Bad methods')
    if len(set(comps))!=len(comps) or len(set(solvers))!=len(solvers):raise ValueError('Duplicate methods')
    spec=json.loads(Path(plan['semantic_file']).read_text(encoding='utf8'))['traits']
    traits=list(spec) if args.traits=='all' else args.traits.split(',')
    if any(t not in spec for t in traits):raise ValueError('Unknown trait')
    if args.discovery_limit<=0 or args.top_k<=0:raise ValueError('Limits must be positive')
    layers=parse_layers(args.layers)
    stage=stage_plan(args.run,'discovery',{'traits':traits,'layers':layers,'compilers':comps,'solvers':solvers,
               'discovery_limit':args.discovery_limit,'top_k':args.top_k,'distribution_top':args.distribution_top,
               'temperature':args.temperature,'max_seq':args.max_seq,'budget':args.budget,'target_delta':args.target_delta})
    if (stage/'SEAL.json').exists():
        require_discovery_seal(stage)
        print('Discovery already sealed; no recomputation.')
        return
    tok,model,lm=load_model(plan['model_dir'])
    norms=column_norms(lm,layers);p50={l:1.0 for l in layers}
    write_json(stage/'normalization.json',{'p50':p50,'layers':layers,
         'definition':'alpha=dose/||W_down[:,neuron]||; purely weight-derived residual-output norm normalization'})
    emb_in=model.get_input_embeddings().weight.detach()
    emb_out=model.get_output_embeddings().weight.detach()
    if emb_in.shape[1]!=emb_out.shape[1]:
        raise RuntimeError('Input and output embedding dimensions differ; cannot use specified cosine compiler')
    print(f'Loaded model. {len(layers)} layers, {len(traits)} traits, {comps} x {solvers}',flush=True)
    for trait in traits:
        rows=rows_for(plan,trait,'discovery',args.discovery_limit)
        targets={}
        for c in comps:
            t=compile_target(c,spec[trait],tok,emb_in,emb_out,rows,args.distribution_top,args.temperature)
            validate_target(t);targets[c]=t
            write_json(stage/f'{trait}_{c}_target.json',canonical_target(t))
        for comp,t in targets.items():
            if 'weight' in solvers:
                v=readout_vector(t,emb_out)
                scores=score_weights(lm,layers,v,p50,norms)
                _save_result(stage,trait,comp,'weight',scores,args)
                del v
        if 'jacobian' in solvers:
            gc.collect();torch.cuda.empty_cache()
            try:
                per=readout_gradients(tok,model,lm,layers,rows,targets,p50,norms,'cuda:0',args.max_seq)
            except Exception:
                err=stage/f'{trait}_backward_traceback.txt'
                err.write_text(traceback.format_exc(),encoding='utf8')
                print(f'GPU backward failed; traceback in {err}. No fallback results fabricated.',flush=True)
                raise
            for comp,by_layer in per.items():
                scores={l:by_layer[l].mean(0) for l in layers}
                _save_result(stage,trait,comp,'jacobian',scores,args,per_probe=by_layer)
        del targets;gc.collect();torch.cuda.empty_cache()
    seal_discovery(stage)
    print('Discovery completed and sealed:',stage,flush=True)


def _save_result(stage,trait,comp,solver,scores,args,per_probe=None):
    key=f'{trait}_{comp}_{solver}'
    path=stage/f'{key}_scores.pt'
    if path.exists():
        old=torch.load(path,map_location='cpu',weights_only=True)
        if any(not torch.equal(old[l],scores[l]) for l in scores):
            raise RuntimeError(f'Conflicting result on resume: {path}. Use a new --run.')
    else:
        torch.save(scores,path)
    if per_probe is not None:
        pp=stage/f'{key}_per_probe.pt'
        if pp.exists():
            old=torch.load(pp,map_location='cpu',weights_only=True)
            if any(not torch.equal(old[l],per_probe[l]) for l in per_probe):
                raise RuntimeError(f'Conflicting gradient probe results: {pp}')
        else:torch.save(per_probe,pp)
    top=top_coordinates(scores,args.top_k)
    write_json(stage/f'{key}_top.json',top)
    synth=combinations(top,max_support=2,pool=min(32,len(top)),budget=args.budget,target_delta=args.target_delta)
    write_json(stage/f'{key}_combos.json',synth[:100])
    print(f' {key}: top L{top[0]["layer"]}N{top[0]["neuron"]}, sign={top[0]["sign"]:+d}, slope={top[0]["slope"]:+.4g}',flush=True)

if __name__=='__main__':main()
