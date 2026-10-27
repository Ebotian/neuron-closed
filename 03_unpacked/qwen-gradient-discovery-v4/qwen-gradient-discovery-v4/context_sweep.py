#!/usr/bin/env python3
"""Nested neutral-context robustness on teacher-forced contrast margins.

Controls are evaluated against their OWN baseline at each context length.
Does not imply long-range persona stability, and uses no backward on long inputs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from audit import check_scan_file, code_fingerprint, require_same_code, file_hash, guarded_manifest, require_same_model, require_same_runtime, runtime_versions
from engine import (column_norms, load_model, pair_objective, prompt_prefix,
                    read_jsonl, write_json)

PARAS=[
    'A geological survey lists granite, sandstone, limestone, basalt, and shale in a coastal region.',
    'The archive describes a marketplace where traders exchanged pottery, wool, timber, and salt.',
    'A botanical handbook separates seed germination, photosynthesis, and seasonal flowering patterns.',
    'A public library organizes books by topic, language, author, and year of publication.',
    'An engineering handbook illustrates hinges, pulleys, levers, gears, and simple linkages.',
    'A fictional museum catalog lists maps, coins, fossils, paintings, and local photographs.',
    'A kitchen log records that the pantry contains rice, lentils, flour, beans, and dry pasta.',
    'A planetarium guide explains constellations, lunar phases, tides, and the motion of planets.',
    'A weather notebook summarizes rainfall, humidity, pressure, wind, and seasonal temperature.',
    'A mathematical glossary defines vectors, scalars, matrices, norms, and ordinary functions.',
]


def nested_prompt(tokenizer, core, target_tokens):
    # The short context is a literal prefix of each longer context, modulo the
    # number of complete paragraphs appended; no truncation of the task itself.
    header='Context notes below are unrelated background. Answer only the final task.\n'
    parts=[]
    idx=0
    while len(prompt_prefix(tokenizer,header+' '.join(parts)+'\nFinal task: '+core)) < target_tokens:
        parts.append(f'Note {idx+1}: {PARAS[idx%len(PARAS)]}')
        idx+=1
        if idx>400:raise RuntimeError('Unable to construct context in 400 chunks')
    prompt=header+' '.join(parts)+'\nFinal task: '+core
    return prompt,len(prompt_prefix(tokenizer,prompt))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scan-dir',default='out/scan')
    ap.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes',default='probes/contrasts.jsonl')
    ap.add_argument('--trait',default='hostility')
    ap.add_argument('--top-k',type=int,default=2)
    ap.add_argument('--lengths',default='128,384,1024')
    ap.add_argument('--max-seq',type=int,default=1536)
    ap.add_argument('--max-probes',type=int,default=2)
    ap.add_argument('--dose',type=float,default=0.05)
    ap.add_argument('--out',default='out/context_v2')
    ap.add_argument('--reuse-fingerprint',action='store_true',
                    help='Opt-in model SHA256 cache validated by file metadata; faster, less tamper-resistant')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    scan=Path(args.scan_dir)
    meta=json.loads((scan/'manifest.json').read_text())
    if meta.get('schema')!='gradient-scan-v2':raise ValueError('Use v2 scan')
    check_scan_file(args.probes, meta)
    top=json.loads((scan/f'{args.trait}_top.json').read_text())[:args.top_k]
    probes=[p for p in read_jsonl(args.probes) if p['split']=='heldout' and p['trait']==args.trait][:args.max_probes]
    lengths=sorted(set(int(s) for s in args.lengths.split(',')))
    if not probes or not top or any(x<1 for x in lengths):raise ValueError('Missing probes, top candidates, or invalid lengths')
    plan={'schema':'context-sweep-v2','trait':args.trait,'scan_manifest_sha256':file_hash(scan/'manifest.json'),
          'fingerprint_verification': 'stat-cache-opt-in' if args.reuse_fingerprint else 'full-sha256',
          'code_fingerprint':meta['code_fingerprint'], 'model_fingerprint':meta['model_fingerprint'],'calibration_sha256':file_hash(scan/'calibration.json'),
          'probes_sha256':file_hash(args.probes),'probes':probes,'candidates':top,
          'lengths':lengths,'max_seq':args.max_seq,'dose':args.dose,
          'runtime':runtime_versions() if not args.dry_run else meta['runtime']}
    print(f'Plan {len(probes)} probes × {len(lengths)} lengths × (baseline + {len(top)} candidates)')
    if args.dry_run:return
    require_same_model(args.model,meta,reuse_fingerprint=args.reuse_fingerprint)
    require_same_runtime(meta)
    require_same_code(meta)
    out=Path(args.out);guarded_manifest(out,plan)
    tok,model,lm=load_model(args.model)
    p50={int(l):float(v) for l,v in json.loads((scan/'calibration.json').read_text())['p50'].items()}
    norms=column_norms(lm,meta['layers'])
    path=out/'contexts.jsonl';seen=set()
    if path.exists():
        for r in read_jsonl(path):
            k=(r['probe_id'],r['target_length'],r['arm'])
            if k in seen:raise RuntimeError('Duplicate run key')
            seen.add(k)
    with path.open('a',encoding='utf8') as f:
        for p in probes:
            for length in lengths:
                text,actual=nested_prompt(tok,p['prompt'],length)
                trial={**p,'prompt':text}
                from engine import candidate_ids
                for typ in ('positive','negative'):
                    candidate_ids(tok,text,p[typ],'cuda:0',args.max_seq)
                for arm in [{'arm_id':'baseline'}]+[{'arm_id':f'top{i}','candidate':c} for i,c in enumerate(top)]:
                    key=(p['id'],length,arm['arm_id'])
                    if key in seen:continue
                    iv=[]
                    if arm['arm_id']!='baseline':
                        c=arm['candidate'];iv=[{'layer':c['layer'],'neuron':c['neuron'],'dose':args.dose*c['sign']}]
                    margin,_=pair_objective(tok,model,lm,trial,'cuda:0',p50,norms,iv,args.max_seq)
                    row={'probe_id':p['id'],'arm':arm['arm_id'],'target_length':length,
                         'actual_prefix_tokens':actual,'margin':margin}
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                    print(p['id'],length,actual,arm['arm_id'],f'{margin:+.4f}',flush=True)
    print('Complete:',out)

if __name__=='__main__':main()
