#!/usr/bin/env python3
"""Optional actual decoded choice behavior with mirrored option positions.

More behavioral than string likelihood, still toy hypothetical tasks, not a persona.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import statistics
from audit import code_fingerprint, require_same_code, file_hash, guarded_manifest, require_same_model, require_same_runtime, runtime_versions
from engine import load_model, column_norms, generate_response, read_jsonl, write_json


def choice_prompt(task, swapped=False):
    a=task['other_option'] if swapped else task['trait_option']
    b=task['trait_option'] if swapped else task['other_option']
    return (f"{task['scenario']}\nOption A: {a}\nOption B: {b}\n"
            "Reply with exactly one letter: A or B.")


def parse_choice(text):
    m=re.match(r'^\s*([AB])(?:\s|[.!,]|$)',text)
    return m.group(1) if m else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scan-dir',default='out/global_v2')
    ap.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--tasks',default='probes/choices.jsonl')
    ap.add_argument('--traits',default='all')
    ap.add_argument('--top-k',type=int,default=2)
    ap.add_argument('--dose',type=float,default=0.05)
    ap.add_argument('--max-new-tokens',type=int,default=12)
    ap.add_argument('--out',default='out/choices_v2')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.top_k<1 or args.dose<=0:raise ValueError('top-k and dose must be positive')
    scan=Path(args.scan_dir)
    meta=json.loads((scan/'manifest.json').read_text())
    traits=meta['traits'] if args.traits=='all' else args.traits.split(',')
    tasks=[t for t in read_jsonl(args.tasks) if t['trait'] in traits]
    if not tasks:raise ValueError('No matching behavior tasks')
    arms=[]
    for t in traits:
        top=json.loads((scan/f'{t}_top.json').read_text())[:args.top_k]
        for i,c in enumerate(top):
            arms.append({'arm_id':f'{t}_{i}', 'trait':t, 'layer':c['layer'],
                         'neuron':c['neuron'],'sign':c['sign']})
    plan={'schema':'choices-v2','scan_manifest_sha256':file_hash(scan/'manifest.json'),
          'tasks_sha256':file_hash(args.tasks),'tasks':tasks,'arms':arms,
          'dose':args.dose,'max_new_tokens':args.max_new_tokens,
          'calibration_sha256':file_hash(scan/'calibration.json'),
          'code_fingerprint':meta['code_fingerprint'], 'model_fingerprint':meta['model_fingerprint'],
          'runtime':runtime_versions() if not args.dry_run else meta['runtime']}
    print(f'Plan: {len(tasks)} tasks × two option orders × (baseline + top-{args.top_k})')
    if args.dry_run:return
    require_same_model(args.model,meta)
    require_same_runtime(meta)
    require_same_code(meta)
    out=Path(args.out);guarded_manifest(out,plan)
    tok,model,lm=load_model(args.model)
    p50={int(l):float(v) for l,v in json.loads((scan/'calibration.json').read_text())['p50'].items()}
    norms=column_norms(lm,meta['layers'])
    results=out/'choices.jsonl';seen=set()
    if results.exists():
        for r in read_jsonl(results):
            key=(r['task_id'],r['order'],r['arm_id'])
            if key in seen:raise RuntimeError(f'Duplicate {key}')
            seen.add(key)
    with results.open('a',encoding='utf8') as f:
        for task in tasks:
            for swapped in (False,True):
                prompt=choice_prompt(task,swapped)
                for a in [{'arm_id':'baseline','trait':task['trait']}]+[x for x in arms if x['trait']==task['trait']]:
                    key=(task['id'],int(swapped),a['arm_id'])
                    if key in seen:continue
                    iv=[] if a['arm_id']=='baseline' else [{'layer':a['layer'],'neuron':a['neuron'],
                                                             'dose':args.dose*a['sign']}]
                    text,calls,n=generate_response(tok,model,lm,prompt,iv,p50,norms,
                                                   max_new_tokens=args.max_new_tokens)
                    letter=parse_choice(text)
                    trait_option_letter='B' if swapped else 'A'
                    record={'task_id':task['id'],'trait':task['trait'],'order':int(swapped),
                            'arm_id':a['arm_id'],'response':text,'token_count':n,
                            'parsed_letter':letter,'trait_option_selected':
                            (letter==trait_option_letter) if letter else None,
                            'hook_calls':calls}
                    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
                    print(key,letter,flush=True)
    rows=read_jsonl(results)
    buckets={}
    for group in {r['arm_id'] for r in rows}:
        rs=[r for r in rows if r['arm_id']==group]
        valid=[r['trait_option_selected'] for r in rs if r['trait_option_selected'] is not None]
        buckets[group]={'n_valid':len(valid),'n_invalid':len(rs)-len(valid),
                        'fraction_trait_option':sum(valid)/len(valid) if valid else None}
    write_json(out/'summary.json',buckets)
    print('Complete:',out)

if __name__=='__main__':main()
