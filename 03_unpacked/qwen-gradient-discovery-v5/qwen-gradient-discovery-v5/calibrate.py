#!/usr/bin/env python3
"""True same-input Taylor calibration: gradients and +/- finite differences on identical prompt.

This separates Taylor remainder from discovery→heldout distribution shift.
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
from pathlib import Path
import torch
from audit import check_scan_file, code_fingerprint, require_same_code, file_hash, guarded_manifest, require_same_model, require_same_runtime, runtime_versions
from engine import (column_norms, load_model, pair_objective, read_jsonl,
                    score_gradient, score_for_dose, write_json)
from design import coordinate


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scan-dir',default='out/scan')
    ap.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes',default='probes/contrasts.jsonl')
    ap.add_argument('--trait',default='hostility')
    ap.add_argument('--split',choices=['discovery','heldout'],default='heldout')
    ap.add_argument('--max-probes',type=int,default=1)
    ap.add_argument('--top-k',type=int,default=4)
    ap.add_argument('--dose',type=float,default=0.02)
    ap.add_argument('--max-seq',type=int,default=384)
    ap.add_argument('--out',default='out/calibration_v2')
    ap.add_argument('--reuse-fingerprint',action='store_true',
                    help='Opt-in model SHA256 cache validated by file metadata; faster, less tamper-resistant')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.dose<=0:raise ValueError('Positive dose required')
    scan=Path(args.scan_dir)
    meta=json.loads((scan/'manifest.json').read_text(encoding='utf8'))
    if meta.get('schema')!='gradient-scan-v2':raise ValueError('Requires v2 scan')
    check_scan_file(args.probes, meta)
    top=json.loads((scan/f'{args.trait}_top.json').read_text())[:args.top_k]
    if not top:raise ValueError('No candidates')
    probes=[p for p in read_jsonl(args.probes) if p['trait']==args.trait and p['split']==args.split][:args.max_probes]
    if not probes:raise ValueError('No calibration probes')
    layers=meta['layers']
    model_fp=meta['model_fingerprint'] if args.dry_run else require_same_model(args.model,meta,reuse_fingerprint=args.reuse_fingerprint)
    plan={'schema':'same-input-calibration-v2','scan_manifest_sha256':file_hash(scan/'manifest.json'),
          'top_file_sha256':file_hash(scan/f'{args.trait}_top.json'),
          'probe_file_sha256':file_hash(args.probes), 'trait':args.trait,'split':args.split,
          'probes':probes,'top':top,'dose':args.dose,'max_seq':args.max_seq,
          'calibration_sha256':file_hash(scan/'calibration.json'),
          'fingerprint_verification': 'stat-cache-opt-in' if args.reuse_fingerprint else 'full-sha256',
          'code_fingerprint':meta['code_fingerprint'], 'model_fingerprint':model_fp,'runtime':runtime_versions() if not args.dry_run else meta['runtime']}
    print(f'Plan: {len(probes)} same-input probes, {len(top)} coords; two VJPs + '
          f'{2*(1+2*len(top))} teacher-forced forwards per probe',flush=True)
    if args.dry_run:return
    require_same_runtime(meta)
    require_same_code(meta)
    out=Path(args.out);guarded_manifest(out,plan)
    tok,model,lm=load_model(args.model)
    cal=json.loads((scan/'calibration.json').read_text())
    p50={int(l):float(v) for l,v in cal['p50'].items()}
    norms=column_norms(lm,layers)
    result=out/'calibration.jsonl';done=set()
    if result.exists():
        for r in read_jsonl(result):
            key=(r['probe_id'],r['coordinate'])
            if key in done:raise RuntimeError(f'Duplicate checkpoint row {key}')
            done.add(key)
    with result.open('a',encoding='utf8') as f:
        for probe in probes:
            if all((probe['id'],coordinate(c['layer'],c['neuron'])) in done for c in top):continue
            try:
                lp,gp=score_gradient(tok,model,lm,layers,probe,'positive','cuda:0',args.max_seq)
                ln,gn=score_gradient(tok,model,lm,layers,probe,'negative','cuda:0',args.max_seq)
            except Exception:
                (out/f'error_{probe["id"]}.traceback.txt').write_text(traceback.format_exc(),encoding='utf8')
                raise
            grad={l:score_for_dose(gp[l]-gn[l],p50[l],norms[l]) for l in layers}
            base=lp-ln
            for c in top:
                l,n=int(c['layer']),int(c['neuron'])
                coord=coordinate(l,n)
                if (probe['id'],coord) in done:continue
                plus,_=pair_objective(tok,model,lm,probe,'cuda:0',p50,norms,
                                      [{'layer':l,'neuron':n,'dose':args.dose}],args.max_seq)
                minus,_=pair_objective(tok,model,lm,probe,'cuda:0',p50,norms,
                                       [{'layer':l,'neuron':n,'dose':-args.dose}],args.max_seq)
                g=float(grad[l][n]);dose=args.dose
                row={'probe_id':probe['id'],'coordinate':coord,'gradient_same_prompt':g,
                     'dose':dose,'pred_plus':dose*g,'pred_minus':-dose*g,
                     'baseline':base,'observed_plus':plus-base,'observed_minus':minus-base,
                     'fd_derivative':(plus-minus)/(2*dose),
                     'fd_curvature':(plus+minus-2*base)/dose**2,
                     'relative_derivative_error':abs((plus-minus)/(2*dose)-g)/(abs(g)+1e-8)}
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                print(probe['id'],coord,f'g={g:+.5g}',f'FD={(plus-minus)/(2*dose):+.5g}',flush=True)
    rows=read_jsonl(result)
    import statistics
    from report import pearson, ranks
    pred=[r['gradient_same_prompt'] for r in rows]
    fd=[r['fd_derivative'] for r in rows]
    slope=sum(a*b for a,b in zip(pred,fd))/sum(a*a for a in pred) if sum(a*a for a in pred)>1e-20 else None
    summary={'n':len(rows), 'same_input_derivative_mae':statistics.mean(abs(a-b) for a,b in zip(pred,fd)),
             'same_input_derivative_rmse':statistics.mean((a-b)**2 for a,b in zip(pred,fd))**0.5,
             'fd_vs_gradient_slope_zero_intercept':slope,
             'fd_vs_gradient_spearman':pearson(ranks(pred),ranks(fd)),
             'mean_abs_fd_curvature':statistics.mean(abs(r['fd_curvature']) for r in rows),
             'notes':['These predictions and finite differences use identical prompts.',
                      'At very small doses BF16 quantization can dominate changes.',
                      'Passing this check does not imply heldout transfer or real behavioral validity.']}
    write_json(out/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('Complete',out)

if __name__=='__main__':main()
