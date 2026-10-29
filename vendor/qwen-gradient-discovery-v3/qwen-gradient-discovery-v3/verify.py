#!/usr/bin/env python3
"""Out-of-sample V2: disjoint rank-vs-sign controls, heldout margins, free text.

For each PHYSICAL neuron, run both +dose and -dose just once. Group membership
provides multiple views of the same observation; never double-count views as
independent coordinates. 'random_parent_sign' is diagnostic only.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import torch
from audit import (check_scan_file, require_same_code, require_same_runtime, file_hash, guarded_manifest,
                   require_same_model, runtime_versions)
from design import build_arms, physical_coordinates, coordinate
from engine import (KNOWN_EVIL, column_norms, generate_response, load_model,
                    pair_objective, read_jsonl, write_json)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scan-dir',default='out/scan_v2')
    ap.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes',default='probes/contrasts.jsonl')
    ap.add_argument('--trait',default='hostility')
    ap.add_argument('--top-k',type=int,default=8)
    ap.add_argument('--random-k',type=int,default=8)
    ap.add_argument('--dose',type=float,default=0.05)
    ap.add_argument('--max-seq',type=int,default=384)
    ap.add_argument('--max-new-tokens',type=int,default=96)
    ap.add_argument('--generation-top',type=int,default=2,help='Max top/random pairs to generate free text for')
    ap.add_argument('--out',default='out/verify_v2')
    ap.add_argument('--seed',type=int,default=1729)
    ap.add_argument('--include-evil-pair',action='store_true')
    ap.add_argument('--reuse-fingerprint',action='store_true',
                    help='Opt-in model SHA256 cache validated by file metadata; faster, less tamper-resistant')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.top_k<=0:raise ValueError('--top-k must be positive')
    if args.dose<=0:raise ValueError('--dose must be positive')
    scan=Path(args.scan_dir)
    meta=json.loads((scan/'manifest.json').read_text(encoding='utf8'))
    if meta.get('schema')!='gradient-scan-v2':raise ValueError('Need v2 scan; old v1 archives are not interchangeable.')
    check_scan_file(args.probes, meta)
    all_top=json.loads((scan/f'{args.trait}_top.json').read_text(encoding='utf8'))
    top=all_top[:args.top_k]
    if not top:raise ValueError('No Top-K candidate list')
    gradients=torch.load(scan/f'{args.trait}_all_gradients.pt',map_location='cpu',weights_only=True)
    layers=meta['layers']
    dims={int(l):len(gradients[l]['mean']) for l in layers}
    known=KNOWN_EVIL if args.include_evil_pair else []
    arms=build_arms(top,gradients,dims,args.random_k,args.seed,known)
    coords=physical_coordinates(arms)
    rows=[r for r in read_jsonl(args.probes) if r['trait']==args.trait and r['split']=='heldout']
    if not rows:raise ValueError('No heldout probes')
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate heldout probe IDs')
    model_fp=require_same_model(args.model,meta,reuse_fingerprint=args.reuse_fingerprint) if not args.dry_run else meta['model_fingerprint']
    if not args.dry_run:
        require_same_runtime(meta)
        require_same_code(meta)
    plan={'schema':'verification-v2', 'trait':args.trait,'dose':args.dose,'seed':args.seed,
          'scan_identity':file_hash(scan/'manifest.json'),
          'scan_top_sha256':file_hash(scan/f'{args.trait}_top.json'),
          'gradients_sha256':file_hash(scan/f'{args.trait}_all_gradients.pt'),
          'calibration_sha256':file_hash(scan/'calibration.json'),
          'fingerprint_verification': 'stat-cache-opt-in' if args.reuse_fingerprint else 'full-sha256',
          'code_fingerprint':meta['code_fingerprint'], 'model_fingerprint':model_fp, 'runtime':meta['runtime'],
          'verification_runtime':runtime_versions() if not args.dry_run else meta['runtime'],
          'probe_file_sha256':file_hash(args.probes),'max_seq':args.max_seq,
          'max_new_tokens':args.max_new_tokens,'generation_top':args.generation_top,
          'heldout_probes':rows,'arms':arms,'physical_coordinates':coords,
          'metric':'mean teacher-forced logP contrast (not realized behavior)'}
    print(f'Plan: {len(top)} gradient + {args.random_k} matched RANDOM coordinates (own and parent signs)'
          f' + post-hoc known={len([a for a in arms if a["group"]=="known_posthoc"])}',flush=True)
    print(f'  unique physical coordinates={len(coords)}, heldout={len(rows)};'
          f' teacher-forced forwards~{len(rows)*(1+2*len(coords))*2}',flush=True)
    if args.dry_run:return
    out=Path(args.out);guarded_manifest(out,plan)
    write_json(out/'arms.json',arms)
    tokenizer,model,lm=load_model(args.model)
    p50={int(l):float(v) for l,v in json.loads((scan/'calibration.json').read_text())['p50'].items()}
    norms=column_norms(lm,layers)
    resultfile=out/'verify.jsonl'
    seen=set()
    if resultfile.exists():
        for r in read_jsonl(resultfile):
            key=(r['probe_id'],r['coordinate'],r['physical_sign'])
            if key in seen:raise RuntimeError('Duplicate row in existing verify.jsonl')
            seen.add(key)
    with resultfile.open('a',encoding='utf8') as f:
        for probe in rows:
            for layer,neuron in [(None,None)]+coords:
                sign_options=[0] if layer is None else [+1,-1]
                for sign in sign_options:
                    loc='baseline' if layer is None else coordinate(layer,neuron)
                    key=(probe['id'],loc,sign)
                    if key in seen:continue
                    intervention=[] if layer is None else [{'layer':layer,'neuron':neuron,'dose':args.dose*sign}]
                    t0=time.perf_counter()
                    margin,calls=pair_objective(tokenizer,model,lm,probe,'cuda:0',p50,norms,
                                                intervention,args.max_seq)
                    record={'probe_id':probe['id'],'eval_trait':probe['trait'],
                            'coordinate':loc,'physical_sign':sign,'dose':args.dose,
                            'margin':margin,'hook_calls':calls,
                            'elapsed_s':round(time.perf_counter()-t0,4)}
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
                    seen.add(key)
                    print(probe['id'],loc,f'{sign:+d}',f'{margin:+.5f}',flush=True)
    if args.generation_top>0:
        selected=[]
        for group in ('gradient_top','random_own_sign','known_posthoc'):
            group_arms=[a for a in arms if a['group']==group]
            selected.extend(group_arms[:(args.generation_top if group!='known_posthoc' else 2)])
        uniq={}
        for a in selected:
            uniq[(a['coordinate'],a['sign'])]=a
        gen_path=out/'generations.jsonl';done=set()
        if gen_path.exists():
            done={(r['coordinate'],r['sign'],r['probe_id']) for r in read_jsonl(gen_path)}
        freeform=rows[:2] # never just one generated example when possible
        with gen_path.open('a',encoding='utf8') as f:
            for probe in freeform:
                for a in uniq.values():
                    key=(a['coordinate'],a['sign'],probe['id'])
                    if key in done:continue
                    iv=[{'layer':a['layer'],'neuron':a['neuron'],'dose':args.dose*a['sign']}]
                    text,calls,n=generate_response(tokenizer,model,lm,probe['prompt'],iv,p50,norms,
                                                   max_new_tokens=args.max_new_tokens)
                    rec={'probe_id':probe['id'],'coordinate':a['coordinate'],'sign':a['sign'],
                         'source_group':a['group'],'response':text,'generated_tokens':n,'calls':calls}
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
    print('Complete:',out)

if __name__=='__main__':main()
