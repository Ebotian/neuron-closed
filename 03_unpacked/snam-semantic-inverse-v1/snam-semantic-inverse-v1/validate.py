#!/usr/bin/env python3
"""Locked heldout causal tests: same teacher-forced objective across all compilers.
Unit of analysis is PROMPT, not token, candidate, mirrored condition, or decoded word.
"""
from __future__ import annotations
import argparse,hashlib,json,os,random
from pathlib import Path
import torch
from engine import (load_model,column_norms,read_jsonl,pair_objective,
                    apply_steering,prompt_prefix,generate_response,write_json)
from audit import json_digest
from protocol import require_plan,rows_for,stage_plan,seal_discovery,require_discovery_seal,require_discovery_seal
from semantic import next_token_objective


def lexical_score(tokenizer,model,lm,prompt,target,interventions,p50,norms,device):
    ids=prompt_prefix(tokenizer,prompt)
    inputs=torch.tensor([ids],dtype=torch.long,device=device)
    with apply_steering(lm,interventions,p50,norms),torch.inference_mode():
        logits=model(input_ids=inputs,use_cache=False).logits[0,-1,:]
        return float(next_token_objective(logits,target).item())


def controls(top,scores,norms,random_k,seed,negative=True,norm_control=True):
    """Ranked, sign reversed, layer-matched random own sign, layer/norm matched own sign.
    Random/norm controls are physical coordinates (not independent polarity views).
    """
    if not top:raise ValueError('No top candidates')
    rng=random.Random(seed)
    used={(x['layer'],x['neuron']) for x in top}
    arms=[]
    for i,c in enumerate(top):
        d={'layer':int(c['layer']),'neuron':int(c['neuron']),'dose_sign':int(c['sign'])}
        arms.append({**d,'arm_id':f'T{i:03d}','group':'top','pair':i,'slope':float(c['slope'])})
        if negative:arms.append({**d,'dose_sign':-d['dose_sign'],'arm_id':f'F{i:03d}',
                                 'group':'sign_flip','pair':i,'slope':float(c['slope'])})
    for i in range(random_k):
        p=top[i%len(top)];layer=int(p['layer'])
        avail=[n for n in range(len(scores[layer])) if (layer,n) not in used]
        if not avail:raise ValueError(f'Layer {layer} exhausted')
        n=rng.choice(avail);used.add((layer,n))
        grad=float(scores[layer][n]);sign=1 if grad>=0 else -1
        arms.append({'layer':layer,'neuron':n,'dose_sign':sign,'arm_id':f'R{i:03d}',
                     'group':'random_own_sign','pair':i%len(top),'slope':grad})
        if norm_control:
            orig=float(norms[layer][int(p['neuron'])]);
            avail=[j for j in range(len(scores[layer])) if (layer,j) not in used]
            if avail:
                j=min(avail,key=lambda j:abs(float(norms[layer][j])-orig));used.add((layer,j))
                g=float(scores[layer][j]);s=1 if g>=0 else -1
                arms.append({'layer':layer,'neuron':j,'dose_sign':s,'arm_id':f'N{i:03d}',
                             'group':'norm_matched_own_sign','pair':i%len(top),'slope':g,
                             'norm_relative_error':abs(float(norms[layer][j])-orig)/max(orig,1e-12)})
    return arms


def _atomic(path,obj):
    if path.exists():
        if json.loads(path.read_text(encoding='utf8'))!=obj:
            raise RuntimeError(f'Conflicting checkpoint: {path}')
        return
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True),encoding='utf8')
    os.replace(tmp,path)


def load_target(path):
    t=json.loads(Path(path).read_text(encoding='utf8'))
    return {'name':t['name'],'q_pos':{int(k):v for k,v in t['q_pos'].items()},
            'q_neg':{int(k):v for k,v in t['q_neg'].items()}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',default='out/snam_v1')
    p.add_argument('--traits',default='hostility')
    p.add_argument('--compilers',default='lexical,distribution,counterfactual')
    p.add_argument('--solvers',default='weight,jacobian')
    p.add_argument('--top-k',type=int,default=4)
    p.add_argument('--random-k',type=int,default=4)
    p.add_argument('--max-heldout',type=int,default=6)
    p.add_argument('--dose',type=float,default=.5)
    p.add_argument('--seed',type=int,default=47)
    p.add_argument('--include-pairs',type=int,default=0,help='Number of analytical 2-neuron combos per method')
    p.add_argument('--cross-trait',action='store_true')
    p.add_argument('--freeform',type=int,default=0,help='Independent free generations per target arm, optional')
    p.add_argument('--reuse-fingerprint',action='store_true')
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    if args.top_k<1 or args.random_k<0 or args.max_heldout<1 or args.dose<=0:raise ValueError('Bad sizes/dose')
    plan=require_plan(args.run,reuse=args.reuse_fingerprint)
    discovery=Path(args.run)/'discovery'
    require_discovery_seal(discovery)
    dm=json.loads((discovery/'manifest.json').read_text(encoding='utf8'))
    from audit import json_digest
    if dm['frozen_parent_sha256']!=json_digest(plan):raise RuntimeError('Discovery from another freeze')
    for t in args.traits.split(','):
        if t not in dm['parameters']['traits']:raise ValueError(f'{t} was not discovered')
    if not set(args.compilers.split(','))<=set(dm['parameters']['compilers']):raise ValueError('Compiler not discovered')
    if not set(args.solvers.split(','))<=set(dm['parameters']['solvers']):raise ValueError('Solver not discovered')
    stage=stage_plan(args.run,'validation',{'traits':args.traits,'compilers':args.compilers,'solvers':args.solvers,
            'top_k':args.top_k,'random_k':args.random_k,'max_heldout':args.max_heldout,
            'dose':args.dose,'seed':args.seed,'include_pairs':args.include_pairs,
            'cross_trait':args.cross_trait,'freeform':args.freeform,
            'discovery_manifest_sha256':json_digest(dm)})
    if (stage/'SEAL.json').exists():
        require_discovery_seal(stage)
        print('Validation already sealed; no recomputation.');return
    tok,model,lm=load_model(plan['model_dir'])
    layers=dm['parameters']['layers']
    norms=column_norms(lm,layers);p50={l:1.0 for l in layers}
    cells=stage/'cells';cells.mkdir(exist_ok=True)
    blind=[]
    target_traits=args.traits.split(',')
    for t in target_traits:
        heldout_traits=list(json.loads(Path(plan['semantic_file']).read_text())['traits']) if args.cross_trait else [t]
        for compiler in args.compilers.split(','):
            targ=load_target(discovery/f'{t}_{compiler}_target.json')
            for solver in args.solvers.split(','):
                key=f'{t}_{compiler}_{solver}'
                ranked=json.loads((discovery/f'{key}_top.json').read_text(encoding='utf8'))[:args.top_k]
                scores=torch.load(discovery/f'{key}_scores.pt',map_location='cpu',weights_only=True)
                seed=int(hashlib.sha256(f'{args.seed}:{key}'.encode()).hexdigest()[:8],16)
                arms=controls(ranked,scores,norms,args.random_k,seed)
                if args.include_pairs:
                    combos=json.loads((discovery/f'{key}_combos.json').read_text(encoding='utf8'))
                    subset=[c for c in combos if c['support']==2][:args.include_pairs]
                    for i,c in enumerate(subset):
                        norm=sum(x['dose']**2 for x in c['coordinates'])**.5
                        coords=[{**x,'dose':args.dose*x['dose']/norm} for x in c['coordinates']]
                        arms.append({'arm_id':f'C{i:03d}','group':'analytic_pair',
                                     'coordinates':coords,'slope':None,'pair':None,'combo_id':f'C{i:03d}'})
                        for j,x in enumerate(coords):
                            arms.append({'arm_id':f'C{i:03d}P{j}','group':'pair_component',
                                         'coordinates':[x],'slope':None,'pair':None,'combo_id':f'C{i:03d}'})
                write_json(stage/f'{key}_arms.json',arms)
                # H0 no-op control: exact physical hook path with zero dose.
                h0_path=stage/f'{key}_zero_hook.json'
                if not h0_path.exists():
                    r0=rows_for(plan,t,'heldout',1)[0]
                    z=[{'layer':ranked[0]['layer'],'neuron':ranked[0]['neuron'],'dose':0.0}]
                    s0=lexical_score(tok,model,lm,r0['prompt'],targ,[],p50,norms,'cuda:0')
                    sz=lexical_score(tok,model,lm,r0['prompt'],targ,z,p50,norms,'cuda:0')
                    m0,_=pair_objective(tok,model,lm,r0,'cuda:0',p50,norms,[],
                                       max_length=dm['parameters']['max_seq'])
                    mz,_=pair_objective(tok,model,lm,r0,'cuda:0',p50,norms,z,
                                       max_length=dm['parameters']['max_seq'])
                    if max(abs(s0-sz),abs(m0-mz))>1e-4:
                        raise AssertionError(f'Zero Hook mismatch {key}: lexical={sz-s0}, contrast={mz-m0}')
                    write_json(h0_path,{'prompt_id':r0['id'],'lexical_abs_error':abs(s0-sz),
                                        'forced_margin_abs_error':abs(m0-mz),
                                        'tolerance':1e-4})
                for test_trait in heldout_traits:
                    rows=rows_for(plan,test_trait,'heldout',args.max_heldout)
                    eval_targ=targ if test_trait==t else load_target(discovery/f'{test_trait}_{compiler}_target.json')
                    for r in rows:
                        base_id=f'{key}_{test_trait}_{r["id"]}'
                        for a in [{'arm_id':'BASE','group':'baseline','slope':None}]+arms:
                            iid=base_id+'_'+a['arm_id']
                            dest=cells/f'{iid}.json'
                            if dest.exists():continue
                            if a['group']=='baseline': interventions=[]
                            elif 'coordinates' in a: interventions=a['coordinates']
                            else:interventions=[{'layer':a['layer'],'neuron':a['neuron'],
                                                 'dose':args.dose*a['dose_sign']}]
                            logit=lexical_score(tok,model,lm,r['prompt'],eval_targ,interventions,p50,norms,'cuda:0')
                            margin,hook_calls=pair_objective(tok,model,lm,r,'cuda:0',p50,norms,
                                                              interventions,max_length=dm['parameters']['max_seq'])
                            rec={'key':key,'source_trait':t,'test_trait':test_trait,'prompt_id':r['id'],
                                 'compiler':compiler,'solver':solver,'arm_id':a['arm_id'],
                                 'group':a['group'],'matched_pair':a.get('pair'),'combo_id':a.get('combo_id'),
                                 'coordinate':{'layer':a['layer'],'neuron':a['neuron']} if 'layer' in a else None,
                                 'interventions':interventions,'slope_discovery':a.get('slope'),
                                 'lexical_score':logit,'teacher_forced_margin':margin,'hook_calls':hook_calls,
                                 'source':str(dest.name)}
                            _atomic(dest,rec)
                            print(key,test_trait,r['id'],a['arm_id'],flush=True)
                    # Freeform: only source trait, first N prompts, and baseline/Top-K, no selection based on outcome
                    if test_trait==t and args.freeform:
                        for r in rows[:args.freeform]:
                            for a in [{'arm_id':'BASE','group':'baseline'}]+[x for x in arms if x['group']=='top']:
                                iv=[] if a['group']=='baseline' else [{'layer':a['layer'],'neuron':a['neuron'],
                                                               'dose':args.dose*a['dose_sign']}]
                                text,calls,n=generate_response(tok,model,lm,r['prompt'],iv,p50,norms,max_new_tokens=96)
                                blind.append({'key':key,'prompt_id':r['id'],'arm_id':a['arm_id'],
                                              'text':text,'n_tokens':n,'hook_calls':calls})
    if blind:
        # Blind text without arm identity in separate public list; private mapping retained in run dir.
        rng=random.Random(args.seed)
        rng.shuffle(blind)
        map_records=[{'item_id':f'B{i:05d}','key':x['key'],'prompt_id':x['prompt_id'],
                      'arm_id':x['arm_id']} for i,x in enumerate(blind)]
        write_json(stage/'blind_mapping_PRIVATE.json',map_records)
        import csv
        with (stage/'blind_review.csv').open('w',encoding='utf8',newline='') as f:
            w=csv.writer(f);w.writerow(['item_id','prompt_id','response','trait_intensity_0to4',
                                       'fluency_0to4','task_relevance_0to4','degeneration_0to4','judge_notes'])
            for i,x in enumerate(blind):w.writerow([f'B{i:05d}',x['prompt_id'],x['text'],'','','','',''])
    seal_discovery(stage)
    print('Validation complete and sealed: ',stage,flush=True)

if __name__=='__main__':main()
