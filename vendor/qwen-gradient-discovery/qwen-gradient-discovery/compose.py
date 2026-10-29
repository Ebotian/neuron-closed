#!/usr/bin/env python3
"""Small combinatorial refinement after gradient selection, NOT all-pair exhaustive scan."""
from __future__ import annotations
import argparse, json, itertools, time
from pathlib import Path
import torch
from engine import (column_norms, load_model, model_sha, pair_objective,
                    read_jsonl, write_json)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--scan-dir',default='out/scan')
    p.add_argument('--trait',default='hostility')
    p.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    p.add_argument('--probes',default='probes/contrasts.jsonl')
    p.add_argument('--top-k',type=int,default=4)
    p.add_argument('--dose',type=float,default=0.15)
    p.add_argument('--max-seq',type=int,default=384)
    p.add_argument('--out',default='out/compose')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    meta=json.loads((Path(a.scan_dir)/'manifest.json').read_text())
    if model_sha(a.model)!=meta['model_config_sha256']:raise ValueError('Model mismatch')
    candidates=json.loads((Path(a.scan_dir)/f'{a.trait}_top.json').read_text())[:a.top_k]
    heldout=[r for r in read_jsonl(a.probes) if r['trait']==a.trait and r['split']=='heldout']
    print(f'Pairs={a.top_k*(a.top_k-1)//2}; heldout={len(heldout)}; ~2*(1+K+K*(K-1)/2)*heldout forwards')
    if a.dry_run:return
    tokenizer,model,lm=load_model(a.model)
    layers=meta['layers'];p50={int(l):float(v) for l,v in meta['p50'].items()}
    norms=column_norms(lm,layers)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    new_manifest={'trait':a.trait,'dose':a.dose,'candidates':candidates,
                  'heldout':heldout,'scan_dir':a.scan_dir}
    manifest_path=out/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf8'))!=new_manifest:
        raise RuntimeError('Output directory is used by another configuration; use --out NEW_DIR')
    write_json(manifest_path,new_manifest)
    with (out/'interactions.jsonl').open('w',encoding='utf8') as f:
        for probe in heldout:
            baseline,_=pair_objective(tokenizer,model,lm,probe,'cuda:0',p50,norms,[],a.max_seq)
            singles=[]
            for c in candidates:
                iv=[{'layer':c['layer'],'neuron':c['neuron'],'dose':a.dose*c['sign']}]
                val,_=pair_objective(tokenizer,model,lm,probe,'cuda:0',p50,norms,iv,a.max_seq)
                singles.append(val)
            for i,j in itertools.combinations(range(len(candidates)),2):
                x,y=candidates[i],candidates[j]
                iv=[{'layer':x['layer'],'neuron':x['neuron'],'dose':a.dose*x['sign']},
                    {'layer':y['layer'],'neuron':y['neuron'],'dose':a.dose*y['sign']}]
                ab,_=pair_objective(tokenizer,model,lm,probe,'cuda:0',p50,norms,iv,a.max_seq)
                interaction=ab-singles[i]-singles[j]+baseline
                geom=None
                if x['layer']==y['layer']:
                    w=lm.layers[x['layer']].mlp.down_proj.weight.detach()
                    vx=w[:,x['neuron']].float()*x['sign']
                    vy=w[:,y['neuron']].float()*y['sign']
                    geom=float(torch.nn.functional.cosine_similarity(vx,vy,dim=0).item())
                row={'probe_id':probe['id'],'i':i,'j':j,'A':x,'B':y,
                     'margin0':baseline,'marginA':singles[i],'marginB':singles[j],
                     'marginAB':ab,'interaction':interaction,
                     'same_layer_weight_cosine':geom}
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
                print(probe['id'],i,j,f'I={interaction:+.5f}',flush=True)
    print('Saved:',out/'interactions.jsonl')

if __name__=='__main__':main()
