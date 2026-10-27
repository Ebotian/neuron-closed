#!/usr/bin/env python3
"""White-box VJP discovery; resumable probe-level gradients with auditable inputs."""
from __future__ import annotations
import argparse
import json
import os
import time
import traceback
from pathlib import Path
import torch
from audit import fingerprint_with_stat_cache, file_hash, guarded_manifest, runtime_versions, code_fingerprint
from engine import (aggregate_probe_gradients, column_norms, dimensions, estimate_p50,
                    known_ranks, load_model, parse_layers, ranked_candidates,
                    read_jsonl, score_gradient, write_json)
from progress import Progress


def checkpoint_name(probe):
    import hashlib
    return hashlib.sha256(json.dumps(probe, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    p.add_argument('--probes',default='probes/contrasts.jsonl')
    p.add_argument('--specificity-probes',default='probes/specificity.jsonl')
    p.add_argument('--choices-tasks',default='probes/choices.jsonl')
    p.add_argument('--traits',default='hostility')
    p.add_argument('--layers',default='11-19')
    p.add_argument('--max-probes',type=int,default=3)
    p.add_argument('--max-seq',type=int,default=384)
    p.add_argument('--top-k',type=int,default=64)
    p.add_argument('--out',default='out/scan')
    p.add_argument('--gate-scan-dir',default=None)
    p.add_argument('--gate-verify-dir',default=None)
    p.add_argument('--override-exploratory-gate', action='store_true',
                   help='Explicitly bypass v3 exploratory criteria, logged in scan manifest')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    layers=parse_layers(args.layers)
    probes=read_jsonl(args.probes)
    traits=sorted({x['trait'] for x in probes}) if args.traits=='all' else [x.strip() for x in args.traits.split(',')]
    subsets={t:[x for x in probes if x['trait']==t and x['split']=='discovery'][:args.max_probes] for t in traits}
    if any(not x for x in subsets.values()):
        raise ValueError('Missing discovery probes for some traits')
    print(f'Plan: {sum(map(len,subsets.values()))} prompts × 2 VJPs; {len(layers)} layers, no generation')
    full_stage=(len(layers)==32 and args.traits=='all')
    gate_record={'required':full_stage,'override':bool(args.override_exploratory_gate)}
    if full_stage and not args.override_exploratory_gate:
        if not args.gate_scan_dir or not args.gate_verify_dir:
            raise RuntimeError('Full all-trait 32-layer scan requires --gate-scan-dir and --gate-verify-dir; '
                               'or explicit --override-exploratory-gate (not recommended).')
        from gate import gate_decision
        gate_record=gate_decision(Path(args.gate_scan_dir),Path(args.gate_verify_dir))
        if not gate_record['pass']:
            raise RuntimeError(f'Exploratory gate failed: {gate_record["reasons"]}')
    if args.dry_run:return
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device='cuda:0'
    out=Path(args.out)
    print('SHA256 hashing model shards & tokenizer for reproducibility...',flush=True)
    identity, fingerprint_mode=fingerprint_with_stat_cache(args.model, reuse=False)
    plan={'schema':'gradient-scan-v2', 'package_release':'v5.1', 'model_fingerprint':identity,
          'fingerprint_verification':fingerprint_mode,'exploratory_gate':gate_record,
          'runtime':runtime_versions(), 'code_fingerprint':code_fingerprint(), 'probe_file_sha256':file_hash(args.probes),
          'auxiliary_dataset_sha256': {'specificity': file_hash(args.specificity_probes),
                                       'choices':file_hash(args.choices_tasks)},
          'traits':traits, 'subsets':subsets, 'layers':layers,
          'max_seq':args.max_seq,'top_k':args.top_k,
          'method':'teacher forced +/- contrast exact all-token VJP, no parameter updates'}
    guarded_manifest(out,plan)
    print('Loading model...',flush=True)
    tok,model,lm=load_model(args.model,device)
    norms=column_norms(lm,layers)
    p50=estimate_p50(tok,model,lm,layers,device)
    dims=dimensions(lm,layers)
    write_json(out/'calibration.json',{'p50':p50,'dims':dims})
    # Progress counts only probes this process still has to compute: a resumed
    # checkpoint is instant, so counting it would bias the ETA downwards.
    pending=[(t,p) for t,rows in subsets.items() for p in rows
             if not (out/'checkpoints'/t/f'{checkpoint_name(p)}.pt').exists()]
    prog=Progress(max(1,len(pending)))
    print(f'Progress: {len(pending)} probe(s) to compute now, one line each',flush=True)
    for trait,rows in subsets.items():
        print('\nTrait',trait,flush=True)
        per_probe=[]; scored=[]
        cp_dir=out/'checkpoints'/trait;cp_dir.mkdir(parents=True,exist_ok=True)
        for probe in rows:
            fp=checkpoint_name(probe)
            dest=cp_dir/f'{fp}.pt'
            if dest.exists():
                record=torch.load(dest,map_location='cpu',weights_only=True)
                if record['probe_id']!=probe['id'] or record['probe_hash']!=fp:
                    raise RuntimeError(f'Bad checkpoint identity {dest}')
                grad=record['gradient'];entry=record['scores']
                print(' ',probe['id'],'[resumed]',flush=True)
            else:
                t0=time.perf_counter()
                try:
                    lp,gp=score_gradient(tok,model,lm,layers,probe,'positive',device,args.max_seq)
                    ln,gn=score_gradient(tok,model,lm,layers,probe,'negative',device,args.max_seq)
                except Exception as exc:
                    errdir=out/'errors';errdir.mkdir(exist_ok=True)
                    (errdir/f'{trait}_{fp}.traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
                    print(f'Failure {type(exc).__name__}; traceback saved. No automatic CUDA diagnosis.',flush=True)
                    raise
                grad={int(l):(gp[l]-gn[l]).detach().cpu() for l in layers}
                for layer in layers:
                    if not torch.isfinite(grad[layer]).all():
                        raise FloatingPointError(f'Non-finite gradient {trait}: {probe["id"]}, layer {layer}')
                entry={'probe_id':probe['id'],'positive_logp':lp,'negative_logp':ln,
                       'baseline_margin':lp-ln,'elapsed_s':round(time.perf_counter()-t0,3)}
                tmp=dest.with_suffix('.pt.tmp')
                torch.save({'probe_id':probe['id'],'probe_hash':fp,'gradient':grad,'scores':entry},tmp)
                os.replace(tmp,dest)
                prog.step(f'{trait} {probe["id"]} margin={lp-ln:+.4f} {entry["elapsed_s"]:.2f}s')
            per_probe.append(grad);scored.append(entry)
        aggr=aggregate_probe_gradients(per_probe,layers,p50,norms)
        winners,ranked=ranked_candidates(aggr,args.top_k)
        write_json(out/f'{trait}_top.json',winners)
        write_json(out/f'{trait}_known_rank.json',known_ranks(ranked))
        torch.save({l:{k:aggr[l][k] for k in ('mean','per_probe','consistency')} for l in layers},
                   out/f'{trait}_all_gradients.pt')
        with (out/f'{trait}_probe_scores.jsonl').open('w',encoding='utf8') as f:
            for entry in scored:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        for i,w in enumerate(winners[:10],1):
            print(f'  {i}: L{w["layer"]} N{w["neuron"]} sign={w["sign"]:+d} |g|={w["abs_gradient"]:.5g}',flush=True)
    print('Done:',out)

if __name__=='__main__':main()
