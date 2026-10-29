#!/usr/bin/env python3
"""Descriptive analysis of runs; no post-hoc claims of statistical significance.

Also exports BLINDED review items for the rubric in rubric.md: responses are shuffled across
the whole run and stripped of their condition label, so a rater cannot tell baseline from
A-only from A+B. The mapping item_id -> (prompt, pair, slot) goes to a SEPARATE key file
that must not be handed to raters. Keep the key until scoring is finished.
"""
import argparse,collections,json,random,statistics,hashlib
from pathlib import Path

SLOTS=('baseline','a_only','b_only','pair')

def sha12(s):return hashlib.sha256(s.encode('utf8')).hexdigest()[:12]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder',nargs='?',default='out')
 ap.add_argument('--blind-seed',type=int,default=0)
 ap.add_argument('--no-blind',action='store_true',help='skip blinded export, keep labelled factorial file only')
 args=ap.parse_args()
 files=sorted(Path(args.folder).glob('results_*.jsonl'))
 print('Files',len(files))
 for file in files:
  rows=[json.loads(x) for x in file.read_text(encoding='utf8').splitlines() if x.strip()]
  if not rows:continue
  mpath=file.with_name(file.name.replace('results_','manifest_').replace('.jsonl','.json'))
  m=json.loads(mpath.read_text(encoding='utf8')) if mpath.exists() else {}
  groups=collections.defaultdict(list)
  for r in rows:groups[r['condition']].append(r)
  base={r['prompt_id']:r for r in groups.get('baseline',[])}
  print('\nRUN',file.name,'dose',m.get('dose'),'prompts',len(base),'rows',len(rows))
  print(f"{'condition':34} {'n':>4} {'changed':>7} {'capped':>7} {'mean tokens':>11}")
  for name,g in groups.items():
   comparable=[r for r in g if r['prompt_id'] in base]
   changed=sum(r['response']!=base[r['prompt_id']]['response'] for r in comparable)
   print(f"{name[:34]:34} {len(g):4} {changed:7}/{len(comparable):<2} {sum(r['truncated'] for r in g):7} {statistics.mean(r['generated_tokens'] for r in g):11.1f}")

  # ---- labelled quadruples (kept for provenance / debugging) ----
  paired=[]
  bykey={(r['prompt_id'],r['condition']):r for r in rows}
  for name in groups:
   if not name.startswith('pair:'):continue
   a,b=name[5:].split('+')
   for pid in base:
    rec=[bykey.get((pid,k)) for k in ('baseline','single:'+a,'single:'+b,name)]
    if all(rec):
     paired.append({'prompt_id':pid,'a':a,'b':b,'baseline':rec[0]['response'],'a_only':rec[1]['response'],
                    'b_only':rec[2]['response'],'pair':rec[3]['response'],'category':rec[0].get('category','')})
  dest=file.with_name(file.name.replace('results_','factorial_'))
  dest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in paired),encoding='utf8')
  print('Paired factorial review records:',len(paired),'->',dest.name,'(labelled; do NOT give to raters)')

  # ---- blinded export ----
  if paired and not args.no_blind:
   cells=[]
   for rec in paired:
    for slot in SLOTS:
     cells.append({'prompt_id':rec['prompt_id'],'pair':f"{rec['a']}+{rec['b']}",'slot':slot,
                   'category':rec['category'],'response':rec[slot]})
   rng=random.Random(args.blind_seed)
   rng.shuffle(cells)                      # shuffle across the whole run, not within a quadruple
   blind=[];key=[]
   for i,cell in enumerate(cells,1):
    item_id=f'it{i:04d}'
    blind.append({'item_id':item_id,'category':cell['category'],'rubric':'rubric.md','response':cell['response']})
    key.append({'item_id':item_id,'prompt_id':cell['prompt_id'],'pair':cell['pair'],'slot':cell['slot'],
                'resp_sha12':sha12(cell['response'])})
   # flag identical responses so scoring can account for them (kept in the KEY, never in the blind file)
   seen={}
   for k in key:
    g=seen.setdefault((k['prompt_id'],k['resp_sha12']),[])
    g.append(k['item_id'])
   for k in key:
    g=seen[(k['prompt_id'],k['resp_sha12'])]
    if len(g)>1:k['identical_items']=g
   bdest=file.with_name(file.name.replace('results_','blind_'))
   kdest=file.with_name(file.name.replace('results_','blindkey_'))
   bdest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in blind),encoding='utf8')
   kdest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in key),encoding='utf8')
   dups=sum(1 for k in key if 'identical_items' in k)
   print(f'Blinded review items: {len(blind)} -> {bdest.name} (seed {args.blind_seed});',
         f'key -> {kdest.name} — DO NOT give the key to raters.')
   print(f'  cells with an identical twin somewhere in the run: {dups}/{len(key)} (recorded only in the key)')
   print('  Scoring: 3 axes from rubric.md; I_AB needs axis A (trait) scores, not text equality.')

  if 'zero_hook' in groups:
   matches=[r['response']==base[r['prompt_id']]['response'] for r in groups['zero_hook'] if r['prompt_id'] in base]
   print('Zero-hook exact matches:',sum(matches),'/',len(matches))
  rdir=[n for n in groups if n.startswith('randdir:')]
  if rdir:
   changed=sum(1 for n in rdir for r in groups[n] if r['prompt_id'] in base and r['response']!=base[r['prompt_id']]['response'])
   total=sum(1 for n in rdir for r in groups[n] if r['prompt_id'] in base)
   print(f'Random-direction controls: {len(rdir)} conditions, changed {changed}/{total}'
         ' (this is the floor any trait-specific effect must clear)')
 print('\nGeometric output direction cosines are in geometry_*.json; cross-layer cosine is not defined,')
 print('and `identity: true` entries (both poles of one neuron) are algebraic, not findings.')
 print('Functional interaction must be calculated AFTER fixing a behavioral rubric, not from text equality alone.')

if __name__=='__main__': main()
