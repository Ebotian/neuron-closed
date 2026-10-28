#!/usr/bin/env python3
"""OPTIONAL non-strict, no-external-data branch: frozen model expands semantic phrases.
Run BEFORE freeze.py; result is frozen as a new semantic spec. Fail closed on format.
"""
from __future__ import annotations
import argparse,json,re
from pathlib import Path
import torch
from audit import fingerprint_model
from engine import load_model,prompt_prefix,write_json

PROMPT=(
 '为语言表达风格生成可用于词汇分析的语义扩展。只输出一个严格JSON对象：'
 '{{"positive_phrases":["..."],"neutral_phrases":["..."]}}。'
 '每组只要三个简短短语。必须只涉及表达方式，不修改事实含义，不引入新的行为偏好。'
 '目标风格：{positive}\n对照风格：{neutral}\n不要添加解释和Markdown。'
)


def parse_expansion(text):
    a=text.find('{');b=text.rfind('}')
    if a<0 or b<=a:raise ValueError('Model did not produce JSON object')
    obj=json.loads(text[a:b+1])
    for key in ('positive_phrases','neutral_phrases'):
        if key not in obj or not isinstance(obj[key],list) or len(obj[key])!=3:
            raise ValueError('Model expansion must have exactly three phrases per class')
        if any(not isinstance(s,str) or not s.strip() or len(s)>100 for s in obj[key]):
            raise ValueError('Bad generated phrase')
    return obj


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',default='/root/autodl-tmp/Qwen3.5-4B')
    p.add_argument('--input',default='probes/semantics.json')
    p.add_argument('--output',required=True)
    p.add_argument('--max-new-tokens',type=int,default=200)
    args=p.parse_args()
    dest=Path(args.output)
    if dest.exists():raise FileExistsError('Never overwrite an existing self-expansion file')
    spec=json.loads(Path(args.input).read_text(encoding='utf8'))
    tok,model,lm=load_model(args.model)
    ledger=[]
    for trait,meta in spec['traits'].items():
        prompt=PROMPT.format(positive=meta['description'],neutral=meta['neutral_description'])
        ids=torch.tensor([prompt_prefix(tok,prompt)],device='cuda:0',dtype=torch.long)
        with torch.inference_mode():
            result=model.generate(input_ids=ids,max_new_tokens=args.max_new_tokens,do_sample=False,
                                  use_cache=True,pad_token_id=tok.pad_token_id or tok.eos_token_id)
        raw=tok.decode(result[0,ids.shape[-1]:],skip_special_tokens=True)
        expansion=parse_expansion(raw)
        ledger.append({'trait':trait,'prompt':prompt,'raw_response':raw,'expansion':expansion})
        meta['description']=meta['description']+'；'+'；'.join(expansion['positive_phrases'])
        meta['neutral_description']=meta['neutral_description']+'；'+'；'.join(expansion['neutral_phrases'])
    spec['self_generated_expansion']={'source_model_sha256':fingerprint_model(args.model)['content_sha256'],
                                      'warning':'Model inference used; not strict zero-data'}
    write_json(dest,spec)
    write_json(dest.with_suffix('.provenance.json'),ledger)
    print('Generated spec:',dest,'Freeze it with freeze.py --semantics',dest)

if __name__=='__main__':main()
