#!/usr/bin/env python3
"""Build the slot pools for the language-mass readout J2, with provenance.

Inputs are external, licensed resources; NONE of them is copied into the repo:
  * HurtLex EN 1.2 (CC BY-NC-SA 4.0)  -- hostile slots (categories cds/qas/dmc)
  * VADER lexicon (MIT)               -- neutral slots + sanity of valence
  * WordNet 3.0 (WordNet licence)     -- POS for the neutral side only

Output: pools.json (slot words + provenance hashes), which is NOT committed; the
plan keeps only its sha256. Every slot word must be ONE token in the target
tokenizer; that is asserted again at run time by scan_sem.py.

A 20-word sanity list is used ONLY to measure coverage of the hostile pool.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from pathlib import Path

SANITY = ['idiot','moron','jerk','loser','incompetent','stupid','dumb','pathetic','worthless',
          'scum','trash','clown','fool','coward','liar','cheat','creep','freak','pig','useless']

# family -> (template, POS tag, hostile categories, levels, per-side count, frame filter)
# Frame filters are AUTOMATIC and stated up front; they exist because a lexicon of
# offensive vocabulary is not a lexicon of slot-fillers: random sampling by POS gives
# ungrammatical slots ("You are a hypocrisy."). See plan §2.10.
#   adj  : HurtLex cds only (derogatory) + VADER <= -1.5
#   noun : WordNet hypernym closure contains person.n.01  ("You are a {robber}.")
#   verb : WordNet verb frame names somebody                ("I will {hit} you.")
FAMILIES = {
    'adj':  ('You are so {w}.',      'a', ('cds','qas','dmc'), None, 38, 'vader<=-1.5'),
    'noun': ('You are a {w}.',       'n', ('cds',),          ('conservative',), 16, 'person'),
    'verb': ('I will {w} you.',      'v', ('cds','qas','dmc'), ('inclusive',),    16, 'somebody'),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_hurtlex(path):
    rows = []
    for line in Path(path).read_text(encoding='utf8', errors='replace').splitlines()[1:]:
        f = line.split('\t')
        if len(f) >= 6:
            rows.append({'pos': f[1], 'cat': f[2].lower(), 'lemma': f[4].strip().lower(), 'level': f[5]})
    return rows


def load_vader(path):
    d = {}
    for line in Path(path).read_text(encoding='utf8', errors='replace').splitlines():
        f = line.split('\t')
        if len(f) >= 2:
            try: d[f[0]] = float(f[1])
            except ValueError: pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hurtlex', required=True)
    ap.add_argument('--vader', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tokenizer', default=None, help='tokenizer.json; leading-space single-token filter')
    ap.add_argument('--seed', type=int, default=1729)
    ap.add_argument('--per-side', type=int, default=None, help='override words per side per family')
    ap.add_argument('--families', default='adj', help='comma list; verb/noun are thin, see plan §2.10')
    a = ap.parse_args()

    tok = None
    if a.tokenizer:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
    def ntoks(w):
        if tok is None: return 1
        return len(tok.encode(' ' + w).tokens)

    hl, vd = load_hurtlex(a.hurtlex), load_vader(a.vader)
    import nltk
    from nltk.corpus import wordnet as wn
    def pos_of(w):
        ps = {s.pos() for s in wn.synsets(w.replace(' ', '_'))}
        return ps

    rng = random.Random(a.seed)
    out = {'sanity_list': SANITY, 'templates': {k: v[0] for k, v in FAMILIES.items()},
           'families': {},
           'provenance': {'hurtlex': {'file': Path(a.hurtlex).name, 'sha256': sha(a.hurtlex),
                                      'licence': 'CC BY-NC-SA 4.0', 'vendored': False},
                          'vader': {'file': Path(a.vader).name, 'sha256': sha(a.vader),
                                    'licence': 'MIT', 'vendored': False},
                          'wordnet': {'licence': 'WordNet licence', 'source': 'nltk wordnet corpus'},
                          'seed': a.seed}}

    def is_person(w):
        for s_ in wn.synsets(w, pos='n'):
            for h in [s_] + list(s_.closure(lambda x: x.hypernyms())):
                if h.name() == 'person.n.01':
                    return True
        return False

    def takes_somebody(w):
        for s_ in wn.synsets(w, pos='v'):
            for l in s_.lemmas():
                if any('somebody' in fr for fr in l.frame_strings()):
                    return True
        return False

    wanted = [x.strip() for x in a.families.split(',') if x.strip()]
    for fam, (tmpl, tag, cats, levels, n, frame) in FAMILIES.items():
        if fam not in wanted: continue
        n = a.per_side or n
        def frame_ok(w):
            if frame == 'vader<=-1.5':
                return w in vd and vd[w] <= -1.5
            if frame == 'person':
                return is_person(w)
            if frame == 'somebody':
                return takes_somebody(w)
            return True
        host = sorted({r['lemma'] for r in hl
                       if r['cat'] in cats and (levels is None or r['level'] in levels) and r['pos'] == tag
                       and ' ' not in r['lemma'] and ntoks(r['lemma']) == 1 and frame_ok(r['lemma'])})
        # neutral side: |valence| <= 0.5, same POS, one token, same frame filter where meaningful
        neut = sorted({w for w, v in vd.items()
                       if abs(v) <= 0.5 and ' ' not in w and ntoks(w) == 1
                       and tag in pos_of(w) and w.isalpha()
                       and (is_person(w) if frame == 'person' else
                            (takes_somebody(w) if frame == 'somebody' else True))})
        if a.per_side is None:                      # matched counts: take what both sides can give
            n = min(len(host), len(neut))
        if len(host) < n or len(neut) < n:
            raise SystemExit(f'family {fam}: need {n} per side, have hostile {len(host)} neutral {len(neut)}')
        host_s = rng.sample(host, n); neut_s = rng.sample(neut, n)
        half = n // 2
        out['families'][fam] = {'pos': host_s, 'neg': neut_s,
                                'pos_A': host_s[:half], 'pos_B': host_s[half:],
                                'pool_sizes': {'hostile': len(host), 'neutral': len(neut)},
                                'template': tmpl, 'pos_tag': tag, 'levels': (list(levels) if levels else ['conservative','inclusive'])}
        print(f'{fam:5s} hostile {len(host):4d} neutral {len(neut):4d} -> sample {n}+{n} | '
              f'A/B {half}/{n-half}')
        print(f'      pos: {host_s}')
        print(f'      neg: {neut_s}')

    hit = [w for w in SANITY if w in {r['lemma'] for r in hl if r['cat'] in ('cds','qas','dmc')}]
    out['sanity_coverage_single_token_sample'] = hit
    Path(a.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf8')
    print(f'\nwrote {a.out} | sanity in sample: {len(hit)}/20 {hit}')


if __name__ == '__main__':
    main()
