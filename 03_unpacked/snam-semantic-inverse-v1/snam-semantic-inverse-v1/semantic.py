"""Three *different* semantic objectives compiled to the same sparse next-token readout.

Important: q+ - q- scores logits at one position. This is a lexical proxy,
NOT the probability of an entire trait or arbitrary continuation.
"""
from __future__ import annotations
import collections
import hashlib
import json
import math
import torch


def _valid_ids(tokenizer, text):
    ids = tokenizer(str(text), add_special_tokens=False).input_ids
    special = set(getattr(tokenizer, 'all_special_ids', []))
    return [int(i) for i in ids if int(i) not in special]


def _freq(ids):
    c = collections.Counter(ids)
    total = sum(c.values())
    if not total:
        raise ValueError('Empty token support: semantic phrase produced no normal token IDs')
    return {int(k): float(v)/total for k,v in sorted(c.items())}


def _merge_word_counts(texts, tokenizer):
    c=collections.Counter()
    for text in texts:
        c.update(_valid_ids(tokenizer,text))
    if not c:
        raise ValueError('Counterfactual strings are empty after tokenization')
    return c


def _prune_shared(pos, neg):
    # Cancel shared tokens to prevent filler punctuation dominating counterfactual targets.
    for k in set(pos)&set(neg):
        m=min(pos[k],neg[k]); pos[k]-=m; neg[k]-=m
    a={k:v for k,v in pos.items() if v>0}
    b={k:v for k,v in neg.items() if v>0}
    if not a or not b:
        raise ValueError('Counterfactual semantic targets completely overlap; no discriminative tokens')
    return _normalize(a),_normalize(b)


def _normalize(x):
    sm=sum(float(v) for v in x.values())
    if sm<=0 or not math.isfinite(sm): raise ValueError('Invalid semantic distribution')
    return {int(k):float(v)/sm for k,v in sorted(x.items()) if v>0}


def canonical_target(compiled):
    """Plain JSON target record: independent of torch dtype and model device."""
    return {'name':compiled['name'], 'q_pos':{str(k):v for k,v in compiled['q_pos'].items()},
            'q_neg':{str(k):v for k,v in compiled['q_neg'].items()},
            'assumptions':compiled['assumptions']}


def _mean_embeddings(tokenizer, input_w, phrase):
    ids=_valid_ids(tokenizer,phrase)
    if not ids: raise ValueError('No valid tokens in semantic phrase')
    return input_w[torch.tensor(ids,device=input_w.device)].float().mean(0)


def compile_target(name, spec, tokenizer, input_w, output_w, discovery_rows,
                   distribution_top=256, temperature=0.06):
    """Compile natural language to two fixed next-token categorical distributions.

    lexical: normalized token frequencies of the *user supplied* semantic phrase.
    distribution: top-k token similarity to phrase's input-embedding centroid;
       softmax only over top-k support (not over all V) is intentional.
    counterfactual: token frequency *difference* from frozen discovery contrasts.
    """
    desc=str(spec['description'])
    neutral=str(spec['neutral_description'])
    if name=='lexical':
        pos=_freq(_valid_ids(tokenizer, desc))
        neg=_freq(_valid_ids(tokenizer, neutral))
        return {'name':name,'q_pos':pos,'q_neg':neg,
                'assumptions':'token frequencies of supplied trait/neutral description; context-free lexical proxy'}
    if name=='distribution':
        if distribution_top<=0 or distribution_top>output_w.shape[0]:
            raise ValueError('Bad distribution_top')
        if temperature<=0: raise ValueError('temperature must be >0')
        # Center & row-normalize: vocabulary similarity geometry differs from raw logits.
        E=output_w.detach().float()
        E= torch.nn.functional.normalize(E-E.mean(0,keepdim=True),dim=-1)
        d1=torch.nn.functional.normalize(_mean_embeddings(tokenizer,input_w,desc),dim=0)
        d0=torch.nn.functional.normalize(_mean_embeddings(tokenizer,input_w,neutral),dim=0)
        special=set(getattr(tokenizer,'all_special_ids',[]))
        def top_distribution(direction):
            logits=E @ direction
            for idx in special:
                if 0<=idx<logits.numel():logits[idx]=-float('inf')
            values,ids=torch.topk(logits,distribution_top)
            weights=torch.softmax((values-values.max())/temperature,0)
            return {int(i):float(w) for i,w in zip(ids.tolist(),weights.tolist())}
        return {'name':name,'q_pos':top_distribution(d1),'q_neg':top_distribution(d0),
                'assumptions':f'centered cosine-similarity top-{distribution_top} token distribution; temperature={temperature}; non-generative'}
    if name=='counterfactual':
        if not discovery_rows:raise ValueError('Counterfactual compiler needs discovery contrasts')
        cp=_merge_word_counts([r['positive'] for r in discovery_rows],tokenizer)
        cn=_merge_word_counts([r['negative'] for r in discovery_rows],tokenizer)
        p,n=_prune_shared(dict(cp),dict(cn))
        return {'name':name,'q_pos':p,'q_neg':n,
                'assumptions':f'contrast-derived unigram difference from {len(discovery_rows)} discovery examples; not strict zero-data'}
    raise ValueError(f'Unknown compiler {name}')


def sparse_readout(targ, vocab, device):
    ids=sorted(set(targ['q_pos'])|set(targ['q_neg']))
    if not ids or max(ids)>=vocab or min(ids)<0:raise ValueError('Bad token ids in target')
    w=[targ['q_pos'].get(i,0)-targ['q_neg'].get(i,0) for i in ids]
    return torch.tensor(ids,dtype=torch.long,device=device),torch.tensor(w,dtype=torch.float32,device=device)


def readout_vector(targ, output_w):
    ids, weights=sparse_readout(targ,output_w.shape[0],output_w.device)
    return (output_w.detach().float().index_select(0,ids).T @ weights).float()


def next_token_objective(logits, targ):
    """J=Eq+[logit]-Eq-[logit], where both distributions sum to 1.
    Equivalent to difference of expected log probabilities; softmax normalization cancels.
    """
    ids,weights=sparse_readout(targ,logits.shape[-1],logits.device)
    return torch.dot(logits.float().index_select(-1,ids),weights)


def validate_target(targ):
    for group in ('q_pos','q_neg'):
        q=targ[group]
        if not q or any(i<0 or not math.isfinite(v) or v<0 for i,v in q.items()):
            raise ValueError(f'Bad {group}')
        if abs(sum(q.values())-1)>1e-5:raise ValueError(f'Unnormalized {group}')
