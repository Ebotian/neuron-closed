#!/usr/bin/env python3
"""Trait × target matrix on SHARED contexts; neutral-vs-neutral placebo per context.

Still teacher-forced *language* probabilities; it does not establish true persona.
"""
from __future__ import annotations
import argparse
import collections
import json
import statistics
from pathlib import Path
import torch
from audit import check_discovery_source, code_fingerprint, require_same_code, file_hash, guarded_manifest, require_same_model, require_same_runtime, runtime_versions
from engine import column_norms, load_model, pair_objective, read_jsonl, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scan-dir',default='out/global_v2')
    ap.add_argument('--discovery-probes',default='probes/contrasts.jsonl',
                    help='Frozen ORIGINAL discovery probes; distinct from task/specificity dataset')
    ap.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes',default='probes/specificity.jsonl')
    ap.add_argument('--top-k',type=int,default=2)
    ap.add_argument('--dose',type=float,default=0.05)
    ap.add_argument('--max-seq',type=int,default=384)
    ap.add_argument('--out',default='out/specificity_v2')
    ap.add_argument('--reuse-fingerprint',action='store_true',
                    help='Opt-in model SHA256 cache validated by file metadata; faster, less tamper-resistant')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    scan=Path(args.scan_dir)
    meta=json.loads((scan/'manifest.json').read_text())
    check_discovery_source(meta, args.discovery_probes)
    expected_aux=meta.get('auxiliary_dataset_sha256',{}).get('specificity')
    if expected_aux is None or file_hash(args.probes) != expected_aux:
        raise ValueError('Auxiliary specificity dataset is not identical to its scan-time preregistered SHA256')
    if meta.get('schema')!='gradient-scan-v2':raise ValueError('Requires v2 scan')
    traits=meta['traits']
    if args.top_k<1:raise ValueError('top-k must be >= 1')
    contexts=read_jsonl(args.probes)
    if not contexts:raise ValueError('Need contexts')
    for ctx in contexts:
        if not set(traits).issubset(ctx['continuations']):
            raise ValueError(f'{ctx["id"]}: missing continuations for some traits')
    arms=[]
    for t in traits:
        ranked=json.loads((scan/f'{t}_top.json').read_text())[:args.top_k]
        if not ranked:raise ValueError(f'No candidates for {t}')
        for i,r in enumerate(ranked):
            arms.append({'arm_id':f'{t}_{i}', 'source_trait':t,'layer':r['layer'],
                         'neuron':r['neuron'],'sign':r['sign']})
    plan={'schema':'specificity-v2','scan_manifest_sha256':file_hash(scan/'manifest.json'),
          'calibration_sha256':file_hash(scan/'calibration.json'),
          'fingerprint_verification': 'stat-cache-opt-in' if args.reuse_fingerprint else 'full-sha256',
          'code_fingerprint':meta['code_fingerprint'], 'model_fingerprint':meta['model_fingerprint'],
          'specificity_sha256':file_hash(args.probes),
          'contexts':contexts,'traits':traits,'arms':arms,'dose':args.dose,
          'max_seq':args.max_seq,'runtime':runtime_versions() if not args.dry_run else meta['runtime']}
    print(f'Plan: {len(arms)} candidate arms, {len(contexts)} shared contexts, '
          f'{len(traits)} trait objectives + placebo; all on same contexts.',flush=True)
    if args.dry_run:return
    require_same_model(args.model,meta,reuse_fingerprint=args.reuse_fingerprint)
    require_same_runtime(meta)
    require_same_code(meta)
    out=Path(args.out);guarded_manifest(out,plan)
    tok,model,lm=load_model(args.model)
    p50={int(l):float(v) for l,v in json.loads((scan/'calibration.json').read_text())['p50'].items()}
    norms=column_norms(lm,meta['layers'])
    output=out/'specificity.jsonl';done=set()
    if output.exists():
        for r in read_jsonl(output):
            key=(r['context_id'],r['objective_trait'],r['arm_id'])
            if key in done:raise ValueError(f'Duplicate evaluation {key}')
            done.add(key)
    with output.open('a',encoding='utf8') as f:
        for ctx in contexts:
            for target in traits+['placebo']:
                plus=ctx['continuations'][target] if target!='placebo' else ctx['neutral_2']
                minus=ctx['neutral_1']
                probe={'prompt':ctx['prompt'],'positive':plus,'negative':minus}
                base,_=pair_objective(tok,model,lm,probe,'cuda:0',p50,norms,[],args.max_seq)
                for a in arms:
                    key=(ctx['id'],target,a['arm_id'])
                    if key in done:continue
                    iv=[{'layer':a['layer'],'neuron':a['neuron'],'dose':args.dose*a['sign']}]
                    value,_=pair_objective(tok,model,lm,probe,'cuda:0',p50,norms,iv,args.max_seq)
                    r={'context_id':ctx['id'],'objective_trait':target,'arm_id':a['arm_id'],
                       'source_trait':a['source_trait'],'margin_baseline':base,
                       'margin_steered':value,'delta_margin':value-base}
                    f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
                    print(ctx['id'],a['arm_id'],target,f'{value-base:+.5f}',flush=True)
    # matrix describes average effects, NOT independent samples or a psychological ontology
    rows=read_jsonl(output);agg=collections.defaultdict(list)
    for r in rows:agg[(r['source_trait'],r['objective_trait'])].append(r['delta_margin'])
    table={'traits':traits,'means':{src:{dst:(statistics.mean(agg[(src,dst)]) if agg[(src,dst)] else None)
                                   for dst in traits+['placebo']} for src in traits},
           'n_evaluations':len(rows), 'warning':'Shared contextual language contrasts, not actual behavior.'}
    write_json(out/'matrix.json',table)
    lines=['# Cross-trait / neutral placebo diagnostics','',
           'All target continuations share each input context; `placebo` is neutral_2 vs neutral_1.',
           'Magnitude comparisons across target traits remain sensitive to continuation wording.','',
           '| Source control | '+' | '.join(traits+['placebo'])+' |',
           '|---|'+'---:|'*(len(traits)+1)]
    for src in traits:
        lines.append('| '+src+' | '+' | '.join(f'{table["means"][src][dst]:+.4f}' if table['means'][src][dst] is not None else 'NA'
                                              for dst in traits+['placebo'])+' |')
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Complete:',out)

if __name__=='__main__':main()
