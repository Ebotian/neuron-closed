#!/usr/bin/env python3
"""Descriptive analysis of runs; no post-hoc claims of statistical significance."""
import argparse,collections,json,itertools,statistics
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('folder',nargs='?',default='out');args=ap.parse_args()
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
  # Functional factorial contrast requires a predeclared numeric outcome; text is nonnumeric.
  # Export matched quadruples for subsequent blinded assessment, instead of fabricating a semantic score.
  paired=[]
  bykey={(r['prompt_id'],r['condition']):r for r in rows}
  for name in groups:
   if not name.startswith('pair:'):continue
   a,b=name[5:].split('+')
   for pid in base:
    rec=[bykey.get((pid,k)) for k in ('baseline','single:'+a,'single:'+b,name)]
    if all(rec):
     paired.append({'prompt_id':pid,'a':a,'b':b,'baseline':rec[0]['response'],'a_only':rec[1]['response'],'b_only':rec[2]['response'],'pair':rec[3]['response']})
  dest=file.with_name(file.name.replace('results_','factorial_'))
  dest.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in paired),encoding='utf8')
  print('Paired factorial review records:',len(paired),'->',dest.name)
  if 'zero_hook' in groups:
   matches=[r['response']==base[r['prompt_id']]['response'] for r in groups['zero_hook'] if r['prompt_id'] in base]
   print('Zero-hook exact matches:',sum(matches),'/',len(matches))
 print('\nGeometric output direction cosines are in geometry_*.json; cross-layer cosine is not defined.')
 print('Functional interaction must be calculated AFTER fixing a behavioral rubric, not from text equality alone.')
if __name__=='__main__':main()
