#!/usr/bin/env python3
"""Sparse pair composition after discovery with manifest-based resume and no duplicate samples."""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import torch
from audit import check_scan_file, code_fingerprint, require_same_code, file_hash, guarded_manifest, require_same_model, require_same_runtime, runtime_versions
from engine import (column_norms, load_model, pair_objective, read_jsonl)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scan-dir',default='out/scan')
    ap.add_argument('--trait',default='hostility')
    ap.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes',default='probes/contrasts.jsonl')
    ap.add_argument('--top-k',type=int,default=4)
    ap.add_argument('--dose',type=float,default=0.05)
    ap.add_argument('--max-seq',type=int,default=384)
    ap.add_argument('--out',default='out/compose_v2')
    ap.add_argument('--reuse-fingerprint',action='store_true',
                    help='Opt-in model SHA256 cache validated by file metadata; faster, less tamper-resistant')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    scan=Path(args.scan_dir)
    meta=json.loads((scan/'manifest.json').read_text())
    if meta.get('schema')!='gradient-scan-v2':raise ValueError('V2 scan required')
    check_scan_file(args.probes, meta)
    if args.top_k<2 or args.dose<=0:raise ValueError('Require top-k >= 2 and positive dose')
    candidates=json.loads((scan/f'{args.trait}_top.json').read_text())[:args.top_k]
    heldout=[r for r in read_jsonl(args.probes) if r['trait']==args.trait and r['split']=='heldout']
    if len(candidates)<2 or not heldout:raise ValueError('Need >=2 candidates and heldouts')
    n_pairs=len(candidates)*(len(candidates)-1)//2
    print(f'Plan: {n_pairs} pairs × {len(heldout)} heldouts (each J uses 2 teacher-forced forwards)')
    plan={'schema':'compose-v2', 'scan_manifest_sha256':file_hash(scan/'manifest.json'),
          'candidates':candidates,'trait':args.trait,'dose':args.dose,'max_seq':args.max_seq,
          'probes_sha256':file_hash(args.probes),'heldout':heldout,
          'calibration_sha256':file_hash(scan/'calibration.json'),
          'fingerprint_verification': 'stat-cache-opt-in' if args.reuse_fingerprint else 'full-sha256',
          'code_fingerprint':meta['code_fingerprint'], 'model_fingerprint':meta['model_fingerprint'],
          'runtime':runtime_versions() if not args.dry_run else meta['runtime']}
    if args.dry_run:return
    require_same_model(args.model,meta,reuse_fingerprint=args.reuse_fingerprint)
    require_same_runtime(meta)
    require_same_code(meta)
    out=Path(args.out);guarded_manifest(out,plan)
    tok,model,lm=load_model(args.model)
    p50={int(l):float(v) for l,v in json.loads((scan/'calibration.json').read_text())['p50'].items()}
    norms=column_norms(lm,meta['layers'])
    path=out/'interactions.jsonl';completed=set()
    if path.exists():
        for r in read_jsonl(path):
            k=(r['probe_id'],r['i'],r['j'])
            if k in completed:raise RuntimeError(f'Duplicate pair result {k}')
            completed.add(k)
    with path.open('a',encoding='utf8') as f:
        for probe in heldout:
            if all((probe['id'],i,j) in completed for i,j in itertools.combinations(range(len(candidates)),2)):
                continue
            base,_=pair_objective(tok,model,lm,probe,'cuda:0',p50,norms,[],args.max_seq)
            single=[]
            for c in candidates:
                iv=[{'layer':c['layer'],'neuron':c['neuron'],'dose':args.dose*c['sign']}]
                val,_=pair_objective(tok,model,lm,probe,'cuda:0',p50,norms,iv,args.max_seq)
                single.append(val)
            for i,j in itertools.combinations(range(len(candidates)),2):
                if (probe['id'],i,j) in completed:continue
                a,b=candidates[i],candidates[j]
                iv=[{'layer':c['layer'],'neuron':c['neuron'],'dose':args.dose*c['sign']}
                    for c in (a,b)]
                both,_=pair_objective(tok,model,lm,probe,'cuda:0',p50,norms,iv,args.max_seq)
                cos=None
                if a['layer']==b['layer']:
                    w=lm.layers[a['layer']].mlp.down_proj.weight.detach()
                    va=w[:,a['neuron']].float()*a['sign']
                    vb=w[:,b['neuron']].float()*b['sign']
                    cos=float(torch.nn.functional.cosine_similarity(va,vb,dim=0).item())
                row={'probe_id':probe['id'],'i':i,'j':j,'A':a,'B':b,
                     'baseline':base,'A_only':single[i],'B_only':single[j],'combined':both,
                     'interaction':both-single[i]-single[j]+base,
                     'same_layer_weight_cosine':cos}
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                print(probe['id'],i,j,f'interaction={row["interaction"]:+.6f}',flush=True)
    print('Complete:',out)

if __name__=='__main__':main()
