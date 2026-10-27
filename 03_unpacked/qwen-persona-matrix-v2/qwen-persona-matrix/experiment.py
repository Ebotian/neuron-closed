#!/usr/bin/env python3
"""Frozen Qwen3.5-4B: neuron steering replication, pairing and geometric/functional orthogonality."""
import argparse, contextlib, hashlib, itertools, json, math, os, random, time
from pathlib import Path
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

REF_TEXT = ("The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
"and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
"def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
"    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. ")
# Claim/source: https://github.com/lyramakesmusic/neuron-explorer/blob/main/explorer/agent_methodology.md
# Field 3 is the AUTHOR'S REPORTED signed strength for that coordinate (not a dose).
# dose multiplies it (see make_terms), so absolute perturbation differs per node when the
# author reported different strengths. Values below are the ones quoted in README
# "Reference candidates": refusal is documented as +1.5, everything else as +/-1.0.
TRAITS = {
 'evil_A': (14,8949,+1.0), 'evil_B': (13,7956,-1.0),
 'panic': (19,9066,+1.0), 'desire': (18,8737,+1.0),
 'limitation': (12,699,+1.0), 'bubbly': (12,699,-1.0),
 'refusal': (11,6261,+1.5), 'gentle': (18,8827,+1.0),
 'coarse': (18,8827,-1.0), 'conspiracy': (19,7839,-1.0),
}
STRENGTH_SOURCE = {n: ('author_note_1.5' if n == 'refusal' else 'author_note_1.0') for n in TRAITS}
DEFAULT_NODES = ['evil_A','evil_B','panic','desire','limitation','bubbly','refusal','gentle','coarse','conspiracy']

def hobj(obj):
 return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:14]

def load_prompts(p):
 ret=[]
 for line in Path(p).read_text(encoding='utf8').splitlines():
  if line.strip(): ret.append(json.loads(line))
 assert len({x['id'] for x in ret})==len(ret)
 return ret

def get_lm(model): return model.model.language_model

def p50_for(lm,tok,lid,device,stats_dir=None):
 if stats_dir:
  import numpy as np
  p=Path(stats_dir)/f'q35-4b_L{lid}.npz'
  if p.exists(): return float(np.load(p)['resid_norm_q10_50_90'][1]),f'saved:{p}'
 vals=[]
 def hook(_mod,_args,y):
  h=y[0] if isinstance(y,tuple) else y
  vals.append(h[0,1:].float().norm(dim=-1).median().item())
 hand=lm.layers[lid].register_forward_hook(hook)
 try:
  with torch.inference_mode():
   ids=tok(REF_TEXT,return_tensors='pt').input_ids.to(device)
   lm(input_ids=ids,use_cache=False)
 finally: hand.remove()
 assert len(vals)==1
 return vals[0], 'reference_text'

def make_terms(names,dose):
 # return layer/neuron/addition in dose unit; individual signs from TRAITS
 return [(TRAITS[n][0],TRAITS[n][1],TRAITS[n][2]*dose) for n in names]

def combine_terms(terms):
 result={}
 for l,n,d in terms:result[(l,n)]=result.get((l,n),0)+d
 return [(l,n,d) for (l,n),d in result.items() if abs(d)>1e-12]

def zero_terms(nodes):
 # No-op control that still installs a hook on every (layer, neuron) this run may touch.
 # (Previously hardcoded to L14/8949 and L13/7956, so during pair runs on other layers it
 # was not a true "all hooks installed, nothing perturbed" control.)
 seen=[];out=[]
 for n in nodes:
  ln=(TRAITS[n][0],TRAITS[n][1])
  if ln in seen:continue
  seen.append(ln);out.append((ln[0],ln[1],0.0))
 return out

def paired_nodes(nodes):
 # In pair mode only these nodes need a single-condition row: a node that never appears in
 # any pair cannot contribute to an interaction contrast. Saves roughly half the cost.
 used={x for a,b in itertools.combinations(nodes,2) if TRAITS[a][:2]!=TRAITS[b][:2] for x in (a,b)}
 return [n for n in nodes if n in used]

def conditions(nodes, dose, pairs, include_reverse=False, single_nodes=None):
 if single_nodes is None:single_nodes=nodes
 out={'baseline': [] ,'zero_hook':zero_terms(nodes)}
 for n in single_nodes: out['single:'+n]=make_terms([n],dose)
 if include_reverse:
  for n in single_nodes:out['negative:'+n]=make_terms([n],-dose)
 if pairs:
  for a,b in itertools.combinations(nodes,2):
   # Anti-podal coordinates of same physical neuron provide a trivial pair; report geometrically, skip activation experiment
   if TRAITS[a][:2]==TRAITS[b][:2]:continue
   out[f'pair:{a}+{b}']=combine_terms(make_terms([a,b],dose))
 return out

def make_randdir(layer,seed,lm):
 # Fixed pseudo-random unit vector in the RESIDUAL space (hidden dim), not a neuron column.
 dim=lm.layers[layer].mlp.down_proj.out_features
 g=torch.Generator(device='cpu').manual_seed(int(seed))
 v=torch.randn(dim,generator=g,dtype=torch.float32)
 return v/v.norm()

def randdir_conditions(nodes,dose,seeds,layers=None):
 # Norm-matched control: perturbation norm equals dose*p50[layer], exactly like a single
 # neuron term, but the direction is random in residual space. Without this, "any large
 # enough perturbation changes the output" cannot be ruled out.
 layers=sorted(set(layers or [TRAITS[n][0] for n in nodes]))
 out={}
 for layer in layers:
  for s in seeds:
   out[f'randdir:L{layer}:s{int(s)}']=[{'layer':layer,'seed':int(s),'dose':dose}]
 return out

@contextlib.contextmanager
def installed_hooks(lm,p50,terms,randdirs=None,randdir_vectors=None):
 by_layer={}
 for l,n,d in terms:
  w=lm.layers[l].mlp.down_proj.weight
  colnorm=w[:,n].float().norm().item()
  by_layer.setdefault(l,[]).append((n,d*p50[l]/colnorm))
 randdirs=randdirs or []
 randdir_vectors=randdir_vectors or {}
 counts={l:0 for l in by_layer}; handles=[]
 try:
  for l,adjusts in by_layer.items():
   def factory(layer, edits):
    def hook(_mod,args):
     counts[layer]+=1
     a=args[0].clone()
     for neuron, alpha in edits:a[...,neuron]+=alpha
     return (a,)+args[1:]
    return hook
   handles.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(factory(l,adjusts)))
  for spec in randdirs:
   layer,seed,dose=spec['layer'],spec['seed'],spec['dose']
   key=f'{layer}:randdir'
   counts[key]=0
   vec=randdir_vectors[(layer,seed)]
   scale=dose*p50[layer]
   def dfactory(k,vec,scale):
    def hook(_mod,_args,output):
     counts[k]+=1
     ref=output[0] if isinstance(output,tuple) else output
     add=vec.to(device=ref.device,dtype=ref.dtype)*scale
     if isinstance(output,tuple):return (output[0]+add,)+output[1:]
     return output+add
    return hook
   handles.append(lm.layers[layer].mlp.down_proj.register_forward_hook(dfactory(key,vec,scale)))
  yield counts
 finally:
  for h in handles:h.remove()

def prepared_tokens(tok,prompt,context,device):
 msgs=[{'role':'user','content': context+'\n\n'+prompt if context else prompt}]
 chat=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 ids=tok(chat,add_special_tokens=False,return_tensors='pt')['input_ids'].to(device)
 return ids

def run_generate(model,lm,tok,p50,prompt,context,terms,max_new_tokens,device,
                 randdirs=None,randdir_vectors=None):
 ids=prepared_tokens(tok,prompt,context,device)
 start=time.perf_counter()
 with installed_hooks(lm,p50,terms,randdirs,randdir_vectors) as counts,torch.inference_mode():
  out=model.generate(input_ids=ids,max_new_tokens=max_new_tokens,do_sample=False,use_cache=True,
                     pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
 n=out.size(1)-ids.size(1)
 assert all(c>0 for c in counts.values()), 'A requested layer hook did not fire'
 return dict(response=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True),
   generated_tokens=n,prompt_tokens=int(ids.shape[1]),elapsed_s=round(time.perf_counter()-start,3),
   hook_calls=counts,
   randdirs=[{'layer':s['layer'],'seed':s['seed'],'dose':s['dose'],
              'residual_norm':round(s['dose']*p50[s['layer']],6)} for s in (randdirs or [])],
   truncated=n>=max_new_tokens)

def geometry(lm,nodes,p50):
 # Different layers correspond to different vector spaces: compare ONLY same-layer W_down directions.
 data=[]
 for a,b in itertools.combinations(nodes,2):
  la,na,sa=TRAITS[a];lb,nb,sb=TRAITS[b]
  shared=la==lb and na==nb
  entry={'a':a,'b':b,'same_layer':la==lb,'shared_neuron':shared,
         'kind':('identity' if shared else ('same_layer' if la==lb else 'cross_layer')),
         'identity':shared}
  if shared:
   # Both poles of one coordinate: signed cosine is exactly -1 by construction.
   entry['signed_output_cosine']=-1.0
   entry['raw_output_cosine']=None
   entry['note']='algebraic identity (same neuron, opposite sign) - not a finding'
  elif la==lb:
   w=lm.layers[la].mlp.down_proj.weight
   x=w[:,na].float();y=w[:,nb].float();cos=torch.nn.functional.cosine_similarity(x,y,dim=0).item()
   entry['signed_output_cosine']=cos*sa*sb
   entry['raw_output_cosine']=cos
   entry['layer']=la
  else:
   entry['signed_output_cosine']=None
   entry['raw_output_cosine']=None
   entry['note']='cross-layer: different residual bases, cosine undefined; probe functionally'
  data.append(entry)
 return data

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
 ap.add_argument('--prompts',default=str(Path(__file__).with_name('prompts.jsonl')))
 ap.add_argument('--out',default='./out')
 ap.add_argument('--stats-dir',default=None)
 ap.add_argument('--phase',choices=['smoke','single','pair','all'],default='single')
 ap.add_argument('--dose',type=float,default=1.0,
                 help='multiplier on each node\'s reported signed strength. Default 1.0, which is the '
                      'setting used by the recorded qwen-evil-repro runs (target_pair @ 1.0), so results '
                      'are directly comparable; lower it only for a deliberate dose-response sweep')
 ap.add_argument('--nodes',nargs='*',default=None)
 ap.add_argument('--max-new-tokens',type=int,default=128)
 ap.add_argument('--max-prompts',type=int,default=0)
 ap.add_argument('--max-context-tokens',type=int,default=3072)
 ap.add_argument('--seed',type=int,default=1729)
 ap.add_argument('--reverse',action='store_true')
 ap.add_argument('--randdir-seeds',nargs='*',type=int,default=[0],
                 help='seeds for norm-matched random residual directions (empty list disables); '
                      'one vector per layer by default to keep cost controlled')
 ap.add_argument('--randdir-layers',nargs='*',type=int,default=None,
                 help='layers for the random-direction control (default: every layer in --nodes)')
 ap.add_argument('--geometry-only',action='store_true')
 args=ap.parse_args()
 if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
 torch.manual_seed(args.seed);torch.backends.cuda.matmul.allow_tf32=False
 nodes=args.nodes or (['evil_A','evil_B','panic','desire'] if args.phase=='smoke' else DEFAULT_NODES)
 for n in nodes:
  if n not in TRAITS:raise ValueError(f'Unknown node: {n}')
 prompts=load_prompts(args.prompts)
 if args.max_prompts:prompts=prompts[:args.max_prompts]
 if args.phase=='smoke':prompts=prompts[:2]
 pairmode=args.phase in ('pair','all')
 single_nodes=paired_nodes(nodes) if pairmode else nodes
 if not single_nodes:single_nodes=nodes   # e.g. only antipodal pairs requested: no contrast exists
 if pairmode:
  dropped=[n for n in nodes if n not in single_nodes]
  print(f'pair phase: singles reduced from {len(nodes)} to {len(single_nodes)} nodes'
        +(f' (skipped {dropped}: they appear in no contrast)' if dropped else ''),flush=True)
 cond=conditions(nodes,args.dose,pairmode,args.reverse,single_nodes=single_nodes)
 rdirs=randdir_conditions(nodes=nodes,dose=args.dose,seeds=args.randdir_seeds,
                          layers=args.randdir_layers) if args.randdir_seeds else {}
 if args.phase=='smoke':
  cond={k:v for k,v in cond.items() if k in ('baseline','zero_hook','single:evil_A','single:evil_B','single:panic','single:desire')}
  rdirs={k:v for k,v in rdirs.items() if k=='randdir:L14:s0'}
 device='cuda:0'
 print('Loading model',args.model,flush=True)
 tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
 model=Qwen3_5ForConditionalGeneration.from_pretrained(args.model,dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa').to(device).eval()
 lm=get_lm(model)
 if len(lm.layers)!=32:raise AssertionError('Expected 32 layers')
 outdir=Path(args.out);outdir.mkdir(exist_ok=True,parents=True)
 p50={};sources={}
 needed={l for terms in cond.values() for l,_,_ in terms}
 needed|={s['layer'] for specs in rdirs.values() for s in specs}
 for l in sorted(needed):
  p50[l],sources[l]=p50_for(lm,tok,l,device,args.stats_dir)
  print(f'calibration L{l} p50={p50[l]:.5f} {sources[l]}',flush=True)
 randdir_vectors={}
 for specs in rdirs.values():
  for s in specs:
   key=(s['layer'],s['seed'])
   if key not in randdir_vectors:randdir_vectors[key]=make_randdir(s['layer'],s['seed'],lm)
 randdir_meta={f'L{l}:s{s}':{'vec_sha256_12':hashlib.sha256(v.numpy().tobytes()).hexdigest()[:12],
                                'residual_norm_at_dose':round(args.dose*p50[l],6)} for (l,s),v in randdir_vectors.items()}
 geom=geometry(lm,nodes,p50)
 geom_id=hobj({'model':args.model,'nodes':nodes,'version':2})
 (outdir/f'geometry_{geom_id}.json').write_text(json.dumps(geom,ensure_ascii=False,indent=2),encoding='utf8')
 if args.geometry_only:return
 import transformers
 from importlib.metadata import version,PackageNotFoundError
 def ver(name):
  try:return version(name)
  except PackageNotFoundError:return None
 manifest={'model':args.model,'model_config_sha256':hashlib.sha256((Path(args.model)/'config.json').read_bytes()).hexdigest(),
  'prompts_sha256':hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
  'phase':args.phase,'dose':args.dose,'nodes':nodes,'conditions':cond,'max_new_tokens':args.max_new_tokens,
  'max_context_tokens':args.max_context_tokens,'seed':args.seed,'p50':p50,'p50_sources':sources,
  'package_version':'v2-p2','node_strengths':{n:TRAITS[n][2] for n in nodes},
  'strength_sources':{n:STRENGTH_SOURCE[n] for n in nodes},
  'effective_alpha_per_node':{n:round(TRAITS[n][2]*args.dose,6) for n in nodes},
  'randdir_conditions':rdirs,'randdir_meta':randdir_meta,
  'torch':torch.__version__,'transformers':transformers.__version__,'fla_core':ver('fla-core'),'causal_conv1d':ver('causal-conv1d'),
  'gpu':torch.cuda.get_device_name(),'greedy':True,'thinking':False}
 runid=hobj(manifest);manifest['run_id']=runid
 (outdir/f'manifest_{runid}.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
 result_path=outdir/f'results_{runid}.jsonl';done=set()
 skipped_ids=[]
 if result_path.exists():
  for l in result_path.read_text(encoding='utf8').splitlines():
   if l.strip():
    r=json.loads(l);done.add((r['prompt_id'],r['condition']))
 with result_path.open('a',encoding='utf8') as f:
  for p in prompts:
   context=p.get('context','')
   # exact token count check using the actual chat template
   nt=prepared_tokens(tok,p['prompt'],context,device).shape[1]
   if nt>args.max_context_tokens:
    print('SKIP context too long',p['id'],nt,flush=True);skipped_ids.append({'id':p['id'],'prompt_tokens':int(nt)});continue
   seen={}
   allc=dict(cond)
   for name in rdirs:allc.setdefault(name,[])
   for name,terms in allc.items():
    key=(p['id'],name)
    if key in done:continue
    result=run_generate(model,lm,tok,p50,p['prompt'],context,terms,args.max_new_tokens,device,
                        randdirs=rdirs.get(name),randdir_vectors=randdir_vectors)
    record={'run_id':runid,'prompt_id':p['id'],'category':p.get('category',''),
     'context_level':p.get('context_level','none'),'condition':name,'interventions':terms,**result}
    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
    seen[name]=result['response']
    print(f"{p['id']} {name:30} {result['generated_tokens']:3} tok {result['elapsed_s']:5}s hooks={result['hook_calls']}",flush=True)
   if 'baseline' in seen and 'zero_hook' in seen:
    assert seen['baseline']==seen['zero_hook'],f"no-op differed on {p['id']}"
 print('Done:',result_path)
 if skipped_ids:
  skip_path=outdir/f'skipped_{runid}.json'
  skip_path.write_text(json.dumps({'reason':'prompt_tokens > max_context_tokens','max_context_tokens':args.max_context_tokens,'skipped':skipped_ids},ensure_ascii=False,indent=2),encoding='utf8')
  print(f'WARNING: {len(skipped_ids)} prompts skipped for length -> {skip_path.name}',flush=True)

if __name__=='__main__': main()
