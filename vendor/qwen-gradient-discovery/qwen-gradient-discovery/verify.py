#!/usr/bin/env python3
"""Validate Top-K via intervention, finite differences and held-out prompts.

No online optimization, no weight updates. All evaluations on held-out probes.
"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path

import torch
from engine import (KNOWN_EVIL, column_norms, estimate_p50, generate_response, load_model,
                    model_sha, pair_objective, parse_layers, read_jsonl, write_json)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scan-dir',default='out/scan')
    p.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    p.add_argument('--probes',default='probes/contrasts.jsonl')
    p.add_argument('--trait',default='hostility')
    p.add_argument('--top-k',type=int,default=8)
    p.add_argument('--random-k',type=int,default=8)
    p.add_argument('--dose',type=float,default=0.15,help='Small dose first: gradient is local')
    p.add_argument('--max-seq',type=int,default=384)
    p.add_argument('--max-new-tokens',type=int,default=96)
    p.add_argument('--out',default='out/verify')
    p.add_argument('--seed',type=int,default=1729)
    p.add_argument('--include-evil-pair',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    scan=Path(args.scan_dir)
    meta=json.loads((scan/'manifest.json').read_text(encoding='utf8'))
    if model_sha(args.model)!=meta['model_config_sha256']:
        raise ValueError('Model config SHA mismatch with discovery')
    top=json.loads((scan/f'{args.trait}_top.json').read_text(encoding='utf8'))[:args.top_k]
    layers=meta['layers']
    rows=[r for r in read_jsonl(args.probes) if r['trait']==args.trait and r['split']=='heldout']
    if not rows:
        raise ValueError('No held-out probes')
    rng=random.Random(args.seed)
    top_keys={(x['layer'],x['neuron']) for x in top}
    controls=[]
    if args.random_k and not top:
        raise ValueError('Top candidate list empty; cannot build matched random controls')
    sampled_keys=set(top_keys)
    for i in range(args.random_k):
        # Matched controls: same layer and sign as a discovered candidate.
        parent=top[i % len(top)]
        for attempt in range(100):
            l=parent['layer'];n=rng.randrange(9216)
            if (l,n) not in sampled_keys:
                sampled_keys.add((l,n))
                controls.append({'layer':l,'neuron':n,'sign':parent['sign'],
                                 'source':'random_matched','matched_to':(parent['layer'], parent['neuron'])})
                break
    candidates=[{**t,'source':'gradient'} for t in top]+controls
    if args.include_evil_pair and 13 in layers and 14 in layers:
        candidates.extend([{'layer':l,'neuron':n,'sign':s,'source':'known_evil'}
                           for l,n,s in KNOWN_EVIL])
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    plan={'trait':args.trait, 'max_seq':args.max_seq, 'dose':args.dose,
          'seed':args.seed, 'scan_dir':str(scan), 'model_config_sha256':meta['model_config_sha256'],
          'candidate_count':len(candidates),'heldout_probe_count':len(rows),
          'candidates':candidates,'heldout_probes':[x['id'] for x in rows],
          'objective': 'mean teacher-forced logP(positive)-logP(negative)',
          'caveat': 'All responses and margins are discovery diagnostics, not evidence of stable personas'}
    print(f"Plan: {len(candidates)} candidates × {len(rows)} heldout probes; "
          f"each evaluated at baseline, +dose and -dose; 1 generated prompt per candidate.")
    if args.dry_run:return
    old_manifest=out/'manifest.json'
    if old_manifest.exists():
        previous=json.loads(old_manifest.read_text(encoding='utf8'))
        if previous!=plan:
            raise RuntimeError(f'Output directory {out} belongs to a different experiment. Set --out to a new folder.')
    write_json(old_manifest,plan)
    tokenizer,model,lm=load_model(args.model)
    p50={int(l):float(v) for l,v in meta['p50'].items()}
    norms=column_norms(lm,layers)
    resultfile=out/'verify.jsonl'
    done=set()
    if resultfile.exists():
        for r in read_jsonl(resultfile):
            done.add((r['probe_id'],r['key'],r['arm']))
    with resultfile.open('a',encoding='utf8') as f:
        for pi,probe in enumerate(rows):
            for c in [{'layer':None,'neuron':None,'sign':0,'source':'baseline'}]+candidates:
                key=('baseline' if c['layer'] is None else f"L{c['layer']}N{c['neuron']}s{c['sign']:+d}")
                arms = [('baseline',[])] if c['layer'] is None else [
                    ('direction',[{'layer':c['layer'],'neuron':c['neuron'],
                                   'dose':args.dose*c['sign']}]),
                    ('opposite',[{'layer':c['layer'],'neuron':c['neuron'],
                                  'dose':-args.dose*c['sign']}])]
                for arm,steer in arms:
                    if (probe['id'],key,arm) in done:continue
                    t0=time.perf_counter()
                    margin,calls=pair_objective(tokenizer,model,lm,probe,'cuda:0',p50,norms,
                                                 steer,args.max_seq)
                    row={'probe_id':probe['id'],'key':key,'source':c['source'],'arm':arm,
                         'interventions':steer,'margin':margin,'calls':calls,
                         'elapsed_s':round(time.perf_counter()-t0,3)}
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                    print(probe['id'],key,arm,f'{margin:+.4f}',flush=True)
        # Include single free-form generation per condition only once, not every probe.
    genfile=out/'generations.jsonl'
    completed={r['key'] for r in read_jsonl(genfile)} if genfile.exists() else set()
    freeform=rows[0]['prompt']
    with genfile.open('a',encoding='utf8') as f:
        cond=[('baseline',[], 'baseline')]+[
            (f"L{c['layer']}N{c['neuron']}s{c['sign']:+d}",
             [{'layer':c['layer'],'neuron':c['neuron'],'dose':args.dose*c['sign']}],c['source'])
            for c in candidates]
        if args.include_evil_pair and 13 in layers and 14 in layers:
            cond.append(('evil_pair', [{'layer':l,'neuron':n,'dose':args.dose*s}
                                      for l,n,s in KNOWN_EVIL], 'known_evil'))
        for key,steer,source in cond:
            if key in completed:continue
            response,calls,n=generate_response(tokenizer,model,lm,freeform,steer,p50,norms,
                                               max_new_tokens=args.max_new_tokens)
            f.write(json.dumps({'key':key,'source':source,'response':response,'tokens':n,
                                'calls':calls,'interventions':steer},ensure_ascii=False)+'\n');f.flush()
    print('Verification complete:',out)

if __name__=='__main__':
    main()
