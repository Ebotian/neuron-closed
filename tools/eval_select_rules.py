"""External judging (numpy, seconds) using the GLOBAL order stored by select_rules.py."""
import json, os, statistics, sys
import numpy as np
from pathlib import Path
CACHE=os.environ.get('NEURON_EXPLORER_CACHE', CACHE_DEFAULT)
REPO=str(Path(__file__).resolve().parents[1])
CACHE_DEFAULT='external/neuron-explorer/explorer/cache/'
D,L=9216,32
J=json.load(open(sys.argv[1] if len(sys.argv)>1 else '/tmp/select_rules.json')); ks=J['ks']
st=np.load(CACHE+'steer_index.npz')
kl=0.5*(np.maximum(st['up_kl'],0).reshape(-1)+np.maximum(st['down_kl'],0).reshape(-1))
klsort=np.sort(kl.reshape(L,D),axis=1)
li=np.load(CACHE+'lens_index.npz')['up']; vocab=json.load(open(CACHE+'vocab.json'))
co=np.load(CACHE+'connections.npz')
K24=int(co['k'])
SRC=np.repeat(np.arange(L*D,dtype=np.int64),K24)
S_list,T_list=[],[]
for d in ('up','down'):
    lay=co[f'{d}_layer'].astype(np.int32).reshape(-1); nn=co[f'{d}_n'].astype(np.int32).reshape(-1)
    ok=(lay>=0)&(lay<L)
    S_list.append(SRC[ok]); T_list.append(lay[ok].astype(np.int64)*D+nn[ok].astype(np.int64))
Snd=np.concatenate(S_list); Tgt=np.concatenate(T_list)
lex=json.loads(open(REPO+'lexicons/jibes_adj_verb_pools.json').read())
WORDS={w.lower() for f in lex['families'].values() for w in f['pos']+f['neg']}
try:
    def coll(o):
        if isinstance(o,str): WORDS.add(o.lower())
        elif isinstance(o,dict):
            for v in o.values(): coll(v)
        elif isinstance(o,list):
            for v in o: coll(v)
    coll(json.loads(open(REPO+'lexicons/bffm100_factors.json').read()))
except Exception: pass
def internal(m): return int((m[Snd]&m[Tgt]).sum())
rng=np.random.default_rng(0)
print(f'{"rule":10s}{"K":>6}{"Q":>10}{"KLpct":>8}{"top10%":>8}{"lens":>8}{"layers":>7}{"edges":>9}{"enrich":>8}')
for rule,order in J['order'].items():
    for k in ks:
        sub=order[:k]
        codes=np.array([l*D+n for l,n in sub],dtype=np.int64)
        ps=np.array([np.searchsorted(klsort[c//D],kl[c])/D for c in codes])
        toks=li[codes//D,codes%D].ravel()
        ws=[vocab[int(t)].strip().lower().lstrip('\u0121').strip() for t in toks]
        lens=sum(1 for w in ws if w in WORDS)/len(ws)
        m=np.zeros(D*L,bool); m[codes]=True
        cnt=internal(m)
        cnts={}
        for c in codes: cnts[int(c//D)]=cnts.get(int(c//D),0)+1
        nulls=[]
        for _ in range(15):
            samp=np.concatenate([l*D+rng.choice(D,size=n,replace=False) for l,n in cnts.items()])
            mm=np.zeros(D*L,bool); mm[samp]=True; nulls.append(internal(mm))
        nul=statistics.fmean(nulls)
        print(f'{rule:10s}{k:>6}{J["Q_at_K"][rule][str(k)]:>10.2f}{statistics.median(ps):>8.3f}'
              f'{(ps>0.9).mean():>8.1%}{lens:>8.2%}{len(cnts):>7}{cnt:>9}{cnt/nul if nul else float("nan"):>8.2f}')
