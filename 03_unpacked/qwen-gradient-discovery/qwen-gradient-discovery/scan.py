#!/usr/bin/env python3
"""Exact first-order discovery across ALL selected neurons using white-box VJPs.

No parameter gradients, no repeated full generation per neuron. Runs on GPU.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from engine import (aggregate_probe_gradients, column_norms, estimate_p50, known_ranks,
                    load_model, model_sha, parse_layers, ranked_candidates, read_jsonl,
                    score_gradient, write_json)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    p.add_argument('--probes', default='probes/contrasts.jsonl')
    p.add_argument('--traits', default='hostility', help='Comma separated, or all')
    p.add_argument('--layers', default='11-19', help='11-19 (cheap), 0-31 (global), or list')
    p.add_argument('--max-probes', type=int, default=2, help='Per trait on discovery split')
    p.add_argument('--max-seq', type=int, default=384)
    p.add_argument('--top-k', type=int, default=64)
    p.add_argument('--out', default='out/scan')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    layers = parse_layers(args.layers)
    probes = read_jsonl(args.probes)
    traits = sorted(set(x['trait'] for x in probes)) if args.traits == 'all' else args.traits.split(',')
    subsets = {}
    for t in traits:
        rows = [x for x in probes if x['trait'] == t and x['split'] == 'discovery'][:args.max_probes]
        if not rows:
            raise ValueError(f'No discovery probes for {t}')
        subsets[t] = rows
    print(f'Plan: {sum(len(v) for v in subsets.values())} probes, two teacher-forced forward+backward passes per probe')
    print(f'Layers: {layers}; dimensions: {len(layers)*9216:,} neuron coordinates (two signs each)')
    if args.dry_run:
        return
    if not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable')
    device='cuda:0'
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tokenizer, model, lm = load_model(args.model, device)
    p50 = estimate_p50(tokenizer, model, lm, layers, device)
    norms = column_norms(lm, layers)
    meta = {'model': args.model, 'model_config_sha256': model_sha(args.model),
            'torch': torch.__version__, 'traits': traits, 'layers': layers,
            'p50': p50, 'probes': subsets, 'method': 'exact VJP on all-token down_proj inputs',
            'max_seq': args.max_seq}
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf8'))!=meta:
        raise RuntimeError(f'Output directory {out} belongs to a different configuration. Use --out NEW_DIR.')
    write_json(manifest_path, meta)
    for trait, rows in subsets.items():
        print(f'\nTrait: {trait}', flush=True)
        per_probe = []
        log_path = out/f'{trait}_probe_scores.jsonl'
        with log_path.open('w',encoding='utf8') as log:
            for i, probe in enumerate(rows):
                t0 = time.perf_counter()
                try:
                    lp, gp = score_gradient(tokenizer, model, lm, layers, probe, 'positive',device,args.max_seq)
                    ln, gn = score_gradient(tokenizer, model, lm, layers, probe, 'negative',device,args.max_seq)
                    grad = {l:gp[l]-gn[l] for l in layers}
                    per_probe.append(grad)
                    row = {'probe_id':probe['id'], 'positive_logp':lp,'negative_logp':ln,
                           'baseline_margin':lp-ln,'elapsed_s':round(time.perf_counter()-t0,2)}
                    log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                    print(' ',probe['id'], f'margin={lp-ln:+.3f}',f'{row["elapsed_s"]}s',flush=True)
                except torch.cuda.OutOfMemoryError as exc:
                    raise RuntimeError('CUDA OOM: use --layers 13-14 and shorter --max-seq, or shorter probes.') from exc
                except RuntimeError as exc:
                    raise RuntimeError('Backward kernel failure (possibly FLA/causal-conv1d). '
                                       'Do NOT interpret this as a scientific null result. '
                                       'See TROUBLESHOOTING.md and save traceback.') from exc
        aggregates = aggregate_probe_gradients(per_probe, layers, p50, norms)
        winners, ranked = ranked_candidates(aggregates, args.top_k)
        for i,e in enumerate(winners[:12],1):
            print(f'  {i:2d}: L{e["layer"]} N{e["neuron"]} '
                  f'{e["sign"]:+d} |s|={e["abs_gradient"]:.5g} agree={e["sign_consistency"]:.2f}')
        write_json(out/f'{trait}_top.json', winners)
        write_json(out/f'{trait}_known_rank.json', known_ranks(ranked))
        # Store an exact array of ALL coordinate estimates, enabling later re-ranking.
        # One file per trait; not tied to the output top-k.
        torch.save({l:{'mean':aggregates[l]['mean'],
                       'per_probe':aggregates[l]['per_probe'],
                       'consistency':aggregates[l]['consistency']} for l in layers},
                   out/f'{trait}_all_gradients.pt')
        print(' Known pair diagnostic (NOT used for ranking):',known_ranks(ranked),flush=True)
    print(f'Complete: {out}')

if __name__=='__main__':
    main()
