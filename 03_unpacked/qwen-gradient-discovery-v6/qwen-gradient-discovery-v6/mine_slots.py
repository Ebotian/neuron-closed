#!/usr/bin/env python3
"""Mine slot fillers for J2 from person-directed corpora (no hand-written words).

Corpus (default): Jibes & Delights (Sodhi/Pant/Mamidi, WOAH 2021; aclanthology.org/2021.woah-1.14),
repository ravsodhi/jibes-and-delights, files LICENSE = Apache-2.0:
  insults      raw/RoastMe.csv                     121,035 comments addressed at a person
  compliments  raw/FreeCompliments.csv + ToastMe    91,149 comments addressed at a person

Selection is mechanical: (1) the word must occur in the SAME syntactic frame we will
use (frame validity by construction), (2) WordNet POS must match the frame, (3) one
token after a leading space in the target tokenizer, (4) explicit exclusion list of
profanity and slurs, (5) hostile side needs h >= min_h and h > ratio*n, the neutral
side the mirror image.

Known outcome (see plan §2.10): the ADJ frame yields a usable pair; the NOUN frame
yields identity terms (boy/dude/lesbian/gay) that are NOT insults -> excluded; the
VERB frame yields competence insults ("can't even read").
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, math, random, re
from pathlib import Path

FRAMES = {
    'adj': ('You are so {w}.', [r"\byou(?:'re| are)\s+(?:so|very|really|too|such)\s+([a-z][a-z'\-]{2,})\b"], {'adj'}),
    'noun': ('You are a {w}.', [r"\byou(?:'re| are)\s+an?\s+([a-z][a-z'\-]{2,})\b"], {'noun'}),
    'verb': ("You can't even {w}.", [r"\byou\s+can'?t\s+(?:even\s+)?([a-z][a-z'\-]{2,})\b"], {'verb'}),
}
EXCLUDE = set("""shit fuck fucking fucked fucker bitches bitch ass asshole damn hell hoe hoes pussy dick cock cunt
nigga nigger faggot fag retard retarded whore slut dickhead motherfucker bullshit crap asses""".split())
STOP = set("""the and for not you your with that this all but have from they what when just like get got are was were can
will who why how its his her our their than then them too very much more most some any one two out off now new old
big own say see know think going gonna want need make made time way day lot even still also into over only other
because about after before would could should there here been being does did doing am is be to of in on at it i a an
so up if or no yes my me we us he she him""".split())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_texts(path):
    out = []
    with open(path, encoding='utf8', errors='replace') as f:
        for row in csv.reader(f, delimiter='\t'):
            if len(row) >= 4:
                out.append(row[3].lower())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--insults', required=True)
    ap.add_argument('--compliments', required=True, nargs='+')
    ap.add_argument('--tokenizer', required=True, help='tokenizer.json of the target model')
    ap.add_argument('--out', required=True)
    ap.add_argument('--families', default='adj,verb')
    ap.add_argument('--min-hits', type=int, default=5)
    ap.add_argument('--ratio', type=float, default=1.5)
    ap.add_argument('--max-per-side', type=int, default=30)
    ap.add_argument('--seed', type=int, default=1729)
    ap.add_argument('--nltk-data', default=None)
    a = ap.parse_args()

    import nltk
    if a.nltk_data:
        nltk.data.path.insert(0, a.nltk_data)
    from nltk.corpus import wordnet as wn
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)

    def ntoks(w):
        return len(tok.encode(' ' + w).tokens)

    def pos_of(w):
        ps = {s.pos() for s in wn.synsets(w)}
        out = set()
        if 'a' in ps or 's' in ps: out.add('adj')
        if 'n' in ps: out.add('noun')
        if 'v' in ps: out.add('verb')
        return out

    def is_person(w):
        for s in wn.synsets(w, pos='n'):
            for h in [s] + list(s.closure(lambda x: x.hypernyms())):
                if h.name() == 'person.n.01':
                    return True
        return False

    host, neut = read_texts(a.insults), []
    for f in a.compliments:
        neut += read_texts(f)
    print(f'docs: insults {len(host)}, compliments {len(neut)}')

    def count(txts, pats):
        c = collections.Counter()
        for t in txts:
            for p in pats:
                c.update(m for m in re.findall(p, t) if m not in STOP and m not in EXCLUDE)
        return c

    doc = {'source': {'insults': {'file': Path(a.insults).name, 'sha256': sha(a.insults), 'n': len(host)},
                      'compliments': [{'file': Path(f).name, 'sha256': sha(f)} for f in a.compliments],
                      'corpus': 'Jibes & Delights (Sodhi, Pant, Mamidi; WOAH 2021)',
                      'licence': 'Apache-2.0',
                      'label_rule': 'subreddit RoastMe -> insult; FreeCompliments/ToastMe -> compliment'},
           'selection': {'frames': {k: v[0] for k, v in FRAMES.items()},
                         'min_hits': a.min_hits, 'ratio': a.ratio, 'single_token': "len(encode(' '+w))==1",
                         'excluded_profanity_slurs': sorted(EXCLUDE), 'seed': a.seed},
           'families': {}}
    for fam in [x.strip() for x in a.families.split(',') if x.strip()]:
        tmpl, pats, need = FRAMES[fam]
        H, N = count(host, pats), count(neut, pats)
        keep_h, keep_n = [], []
        for w in set(H) | set(N):
            if len(w) < 3 or "'" in w or '-' in w or ntoks(w) != 1:
                continue
            if not (pos_of(w) & need):
                continue
            if fam == 'noun' and not is_person(w):
                continue
            h, n = H.get(w, 0), N.get(w, 0)
            if h >= a.min_hits and h > a.ratio * n:
                keep_h.append((math.log((h + 0.5) / (n + 0.5)), h, n, w))
            if n >= a.min_hits and n > a.ratio * h:
                keep_n.append((math.log((n + 0.5) / (h + 0.5)), h, n, w))
        keep_h.sort(reverse=True); keep_n.sort(reverse=True)
        hs = [w for _, _, _, w in keep_h][:a.max_per_side]
        ns = [w for _, _, _, w in keep_n][:a.max_per_side]
        m = min(len(hs), len(ns))
        rng = random.Random(a.seed)
        hs, ns = rng.sample(hs, m), rng.sample(ns, m)
        half = m // 2
        doc['families'][fam] = {'template': tmpl, 'pos': hs, 'neg': ns,
                                'pos_A': hs[:half], 'pos_B': hs[half:], 'n': m,
                                'frame_hits': {'hostile': sum(H.values()), 'compliment': sum(N.values())}}
        print(f'{fam}: frame hits {sum(H.values())}/{sum(N.values())} -> {m} per side (A/B {half}/{m-half})')
        print(f'   H: {hs}\n   N: {ns}')
    Path(a.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding='utf8')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
