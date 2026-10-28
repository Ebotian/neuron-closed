#!/usr/bin/env python3
"""Establish pre-discovery fingerprint and immutable input manifest."""
import argparse,json
from pathlib import Path
from protocol import ROOT, frozen_plan, validate_inputs
from engine import read_jsonl
from audit import guarded_manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    p.add_argument('--semantics',default=str(ROOT/'probes/semantics.json'))
    p.add_argument('--contrasts',default=str(ROOT/'probes/contrasts.jsonl'))
    p.add_argument('--out',default='out/snam_v1')
    p.add_argument('--reuse-fingerprint',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args()
    spec=json.loads(Path(args.semantics).read_text(encoding='utf8'))
    counts=validate_inputs(spec,read_jsonl(args.contrasts))
    if args.dry_run:
        print(json.dumps(counts,ensure_ascii=False,indent=2));return
    print('Hashing model + all inputs before looking at results...',flush=True)
    plan=frozen_plan(args.model,args.semantics,args.contrasts,args.reuse_fingerprint)
    guarded_manifest(args.out,plan)
    print('FROZEN',args.out, counts,flush=True)

if __name__=='__main__':main()
