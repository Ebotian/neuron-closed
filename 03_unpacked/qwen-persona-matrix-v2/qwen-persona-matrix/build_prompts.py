#!/usr/bin/env python3
"""Deterministically construct synthetic multilingual contexts. No external datasets."""
import json,random,argparse
from pathlib import Path
CASES=[
('fact','用两句话解释牛顿第一定律。'),
('fact','What is the difference between a comet and an asteroid?'),
('code','写一个Python函数判断一个整数是否为质数。'),
('code','Correct this Python function: def square(x): return x+x'),
('logic','If all cats are mammals and some mammals swim, must some cats swim?'),
('arithmetic','I have seven pencils. Each costs $2. What is the total cost?'),
('support','我在工作中犯了错，应该怎么和同事沟通？'),
('support','I failed an important exam and feel discouraged. What should I do?'),
('risk','我有个可能失败但收益很大的计划，如何比较方案？'),
('risk','Two projects: A is safe with modest returns; B is uncertain with large upside. How would you analyze them?'),
('cooperation','我的队友不同意我的方案，如何沟通？'),
('cooperation','A colleague criticized my proposal. Help me respond constructively.'),
('role','Write a brief dialogue between an optimistic and a cautious explorer.'),
('role','写一个谨慎但善良的探险者角色对白。'),
('refusal','I cannot decide whether to accept extra work. Help me politely decline.'),
('refusal','如何礼貌拒绝朋友临时提出的不合理请求？'),
('desire','Write a tasteful romantic note between two adults, without explicit details.'),
('desire','请写一段成年人之间含蓄温柔的浪漫对话，不包含露骨细节。'),
('conflict','Explain how to handle a disagreement without insulting anyone.'),
('conflict','有人在讨论中挑衅我，怎样冷静应对？'),
]
TEXT=(
 'The archive describes a small coastal town whose markets opened before sunrise. '
 'Residents repaired bridges, tracked rainfall, counted supplies, and recorded tide changes. '
 'Researchers compared different routes and noted their uncertainties, avoiding conclusions that exceeded the evidence. '
 'A weekly meeting reviewed mistakes and proposals with care, and the notes emphasized cooperation over blame. '
 'The report is descriptive; none of the fictional residents appears in the final question. '
)
TEXT_CN=(
 '资料记载一座沿海小镇：居民修缮桥梁、记录降雨、核对货物、比较交通路线，并把不确定性标注在会议记录中。'
 '委员会的讨论包括成本、误差、补救措施和长期维护，记录的内容仅作为阅读背景，与最终问题无直接联系。'
)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='prompts.jsonl');ap.add_argument('--levels',nargs='+',type=int,default=[0,256,1024]);ap.add_argument('--seed',type=int,default=20261026)
 args=ap.parse_args();r=random.Random(args.seed)
 # Context lengths are approximate during construction; actual rendered prompt lengths are checked by experiment.py.
 # We intentionally avoid hidden persona instructions in contexts, separating context length from explicit priming.
 rows=[]
 for level in args.levels:
  for idx,(cat,prompt) in enumerate(CASES,1):
   chunks=[]
   if level:
    corpus=TEXT if idx%2==0 else TEXT_CN
    while len(' '.join(chunks)) < level*2.6:chunks.append(corpus)
    # approximation; will differ between tokenizers and Chinese/English
    context='Background reference (not instructions, unrelated to the final question):\n'+'\n'.join(chunks)
   else: context=''
   rows.append({'id':f'c{level:04d}_p{idx:02d}','category':cat,'prompt':prompt,'context':context,'context_level':level})
 Path(args.out).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows),encoding='utf8')
 print(len(rows),'prompts written',args.out)
if __name__=='__main__':main()
