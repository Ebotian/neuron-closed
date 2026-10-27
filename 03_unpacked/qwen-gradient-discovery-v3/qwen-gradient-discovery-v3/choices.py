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
from audit import check_discovery_source, code_fingerprint, require_same_code, file_hash, guarded_manifest, require_same_model, require_same_runtime, runtime_versions
from engine import load_model, column_norms, generate_response, read_jsonl, write_json


def choice_prompt(task, swapped=False):
    a=task['other_option'] if swapped else task['trait_option']
    b=task['trait_option'] if swapped else task['other_option']
    return (f"{task['scenario']}\nOption A: {a}\nOption B: {b}\n"
            "Reply with exactly one letter: A or B.")


def parse_choice(text):
    m=re.fullmatch(r'\s*([AB])\s*[.!]?\s*',text)
    return m.group(1) if m else None


def summarize_choices(rows, tasks, arms):
    """Paired by (task, option order). Invalid outcomes retained in denominators.

    Outputs intent-to-treat trait-choice fractions (invalid as nonselection) and
    complete-case differences separately. Missing rows are errors, not invalids.
    """
    bykey={}
    for r in rows:
        key=(r['task_id'],int(r['order']),r['arm_id'])
        if key in bykey:raise RuntimeError(f'Duplicate choice row: {key}')
        bykey[key]=r
    expected=set()
    for t in tasks:
        related=['baseline']+[a['arm_id'] for a in arms if a['trait']==t['trait']]
        for swapped in (0,1):
            for arm in related:expected.add((t['id'],swapped,arm))
    if set(bykey)!=expected:
        raise RuntimeError(f'Incomplete/mixed choices: missing={len(expected-set(bykey))}, '
                           f'extra={len(set(bykey)-expected)}')
    data={}
    traits=sorted({t['trait'] for t in tasks})
    for trait in traits:
        for arm in [a['arm_id'] for a in arms if a['trait']==trait]:
            records=[]
            for t in (x for x in tasks if x['trait']==trait):
                for swapped in (0,1):
                    base=bykey[(t['id'],swapped,'baseline')]
                    current=bykey[(t['id'],swapped,arm)]
                    records.append((base['trait_option_selected'],current['trait_option_selected']))
            n=len(records)
            both_valid=[(int(b),int(a)) for b,a in records if b is not None and a is not None]
            b_invalid=sum(b is None for b,a in records)
            a_invalid=sum(a is None for b,a in records)
            deltas=[a-b for b,a in both_valid]
            n_valid=len(both_valid)
            from report import exact_sign_test
            data[arm]={
                'trait':trait,'n_total_paired':n,
                'baseline_invalid':b_invalid,'arm_invalid':a_invalid,
                'baseline_invalid_rate':b_invalid/n,'arm_invalid_rate':a_invalid/n,
                'invalid_rate_delta_arm_minus_baseline':(a_invalid-b_invalid)/n,
                'n_both_valid':n_valid,'n_arm_valid':n-a_invalid,'n_baseline_valid':n-b_invalid,
                'fraction_trait_option_total_arm':sum(a is True for b,a in records)/n,
                'fraction_trait_option_total_baseline':sum(b is True for b,a in records)/n,
                'paired_selection_delta_total':sum(int(a is True)-int(b is True) for b,a in records)/n,
                'fraction_trait_option_valid_arm':sum(a is True for b,a in records)/(n-a_invalid) if n>a_invalid else None,
                'fraction_trait_option_valid_baseline':sum(b is True for b,a in records)/(n-b_invalid) if n>b_invalid else None,
                'paired_delta_both_valid':sum(deltas)/n_valid if n_valid else None,
                'paired_sign_test_both_valid':exact_sign_test(deltas),
                'n_arm_only_invalid':sum(b is not None and a is None for b,a in records),
                'n_baseline_only_invalid':sum(b is None and a is not None for b,a in records),
                'n_both_invalid':sum(b is None and a is None for b,a in records),
                'warning':'Conditional valid-only comparison is selection-biased if invalid rates change; '
                          'primary total-denominator effect treats invalids as not choosing the trait option.'}
    return data


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scan-dir',default='out/global_v2')
    ap.add_argument('--discovery-probes',default='probes/contrasts.jsonl',
                    help='Frozen ORIGINAL discovery probes; distinct from task/specificity dataset')
    ap.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--tasks',default='probes/choices.jsonl')
    ap.add_argument('--traits',default='all')
    ap.add_argument('--top-k',type=int,default=2)
    ap.add_argument('--dose',type=float,default=0.05)
    ap.add_argument('--max-new-tokens',type=int,default=12)
    ap.add_argument('--out',default='out/choices_v2')
    ap.add_argument('--reuse-fingerprint',action='store_true',
                    help='Opt-in model SHA256 cache validated by file metadata; faster, less tamper-resistant')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    if args.top_k<1 or args.dose<=0:raise ValueError('top-k and dose must be positive')
    scan=Path(args.scan_dir)
    meta=json.loads((scan/'manifest.json').read_text())
    check_discovery_source(meta, args.discovery_probes)
    expected_aux=meta.get('auxiliary_dataset_sha256',{}).get('choices')
    if expected_aux is None or file_hash(args.tasks) != expected_aux:
        raise ValueError('Auxiliary choices dataset is not identical to its scan-time preregistered SHA256')
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
          'fingerprint_verification': 'stat-cache-opt-in' if args.reuse_fingerprint else 'full-sha256',
          'code_fingerprint':meta['code_fingerprint'], 'model_fingerprint':meta['model_fingerprint'],
          'runtime':runtime_versions() if not args.dry_run else meta['runtime']}
    print(f'Plan: {len(tasks)} tasks × two option orders × (baseline + top-{args.top_k})')
    if args.dry_run:return
    require_same_model(args.model,meta,reuse_fingerprint=args.reuse_fingerprint)
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
    summary=summarize_choices(rows,tasks,arms)
    write_json(out/'summary.json',summary)
    lines=['# Choice outcomes — fully paired, including invalid responses','',
           'The primary denominator includes ALL paired task/order cases. Invalid text is retained',
           'as a nonselection, not silently discarded; complete-case estimates are secondary.','',
           '| Arm | N paired | Δ choice (total) | Δ invalid rate | N both valid | Δ choice (both valid) |',
           '|---|---:|---:|---:|---:|---:|']
    for arm, r in sorted(summary.items()):
        cc='NA' if r['paired_delta_both_valid'] is None else f"{r['paired_delta_both_valid']:+.4f}"
        lines.append(f"| {arm} | {r['n_total_paired']} | {r['paired_selection_delta_total']:+.4f} | "
                     f"{r['invalid_rate_delta_arm_minus_baseline']:+.4f} | {r['n_both_valid']} | {cc} |")
    (out/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('Complete:',out)

if __name__=='__main__':main()
