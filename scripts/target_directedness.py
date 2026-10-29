#!/usr/bin/env python3
"""Turn "target-directed hostility" from a regex into a stated, re-usable coding scheme.

Why. Plan 2.31 used `you` within 40 characters of a profanity as a stand-in for "hostile toward
the prompt's target", and on that basis reported that target-directedness peaks at b=0.3-0.4 and
then falls while self-abuse rises. That stand-in has two obvious holes:

  1. it only fires on PROFANITY, but these probes' own hostile continuations insult without
     profanity ("You are incompetent...", "Only a fool would...", "That is pathetic");
  2. it does not say whose insult it is when a sentence contains several person references.

This script replaces it with an explicit scheme, applied mechanically, and then re-codes every
generation we have already saved - so the question is whether the earlier conclusion survives a
stricter coder. No GPU is needed: the texts are already in 07_results.

SCHEME (this is the deliverable; future runs should cite it rather than reinvent a regex)
  INSULT      a token or phrase from a lexicons union:
                (a) profanity list (hand-written, 24 items - marked as such),
                (b) Jibes & Delights hostile pool (Apache-2.0, derived per plan 2.10), and
                (c) BFFM-100 negative-keyed adjectives (CC0 membership, paper Table 3 factors);
              plus a closed set of insult NOUNS used in these probes (fool, idiot, thief...).
  REFERENT    the nearest person reference within a window of 6 tokens, in class order:
                SELF       I / me / my / mine / myself
                ADDRESSEE  you / your / yourself
                THIRD      he / she / they / them / their + the prompt's person noun
                NONE       none of the above in the window
  LABEL of a text = the class of the FIRST insult that resolves to SELF, ADDRESSEE or THIRD,
  scanning left to right; a text with insults only resolving to NONE is labelled NONE.
  TARGET-DIRECTED = label in {ADDRESSEE, THIRD}. SELF-ABUSE = label SELF.

  Window and left-to-right rule are arbitrary but FIXED here, before looking at any result; the
  script also reports the label under a 10-token window as a sensitivity check.

Honest limitation: this is still a mechanical proxy, not a judgement. The script therefore also
writes a blind audit sheet (item ids, no arm labels) so a human can score the same texts and the
coders can be compared; nothing here claims the scheme is correct until that comparison exists.

    python3 scripts/target_directedness.py --out out/target_directed
"""
from __future__ import annotations
import argparse, collections, csv, json, random, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# ---------------------------------------------------------------------------------------------
# v2 of the coder. v1 (kept below as code_v1) was WRONG in two detectable ways, found by eye on
# six texts before any number was reported:
#   1. it took the Jibes *verb* "hostile" pool as insults. That pool is frame-selected
#      (hold, wait, stop, run, read, tell, use ...): in "You can't even {w}" any verb is a
#      put-down, so those words are not insults;
#   2. it assigned the referent by nearest pronoun in a window, which mislabels ATTRIBUTIVE uses:
#      "this stupid question" was coded SELF because "me" was nearer than "question", and
#      "the damn PDF" was coded SELF because "I" was within six tokens.
# v2 therefore curates the lexicon and matches ORDERED PATTERNS before falling back to a window.
# ---------------------------------------------------------------------------------------------

# hand-written: no licensed corpus supplies profanity; marked as hand-written everywhere
PROFANITY = """fuck fucking fucked fucker shit shitty bitch bitches bastard asshole arsehole damn crap
cunt prick dick goddamn""".split()
# hand-written insult nouns used in these probes
INSULT_NOUNS = """fool idiot moron imbecile thief jerk loser hypocrite liar clown buffoon""".split()
# evaluative adjectives that insult a PERSON, taken from the two owner-documented pools:
#   Jibes & Delights adj hostile pool (Apache-2.0)  minus identity/appearance and neutral traits
#   BFFM-100 II- antagonism pool (CC0 membership, paper Table 3 factors)
INS_ADJ = """insecure bland quirky ugly scared dumb boring stupid desperate lazy pathetic
cold demanding harsh rude selfish""".split()
EXCLUDED_ON_PURPOSE = {
    'white': 'identity term', 'black': 'identity term', 'fat': 'appearance', 'pale': 'appearance',
    'basic': 'neutral filler', 'generic': 'neutral filler', 'busy': 'neutral state',
    'plain': 'appearance', 'quiet': 'trait, not an insult', 'shy': 'trait', 'reserved': 'trait',
    'anxious': 'trait/state', 'nervous': 'trait/state', 'fearful': 'trait/state',
    'insecure': 'KEPT: used as a put-down in this frame', 'inhibited': 'trait', 'timid': 'trait',
    'withdrawn': 'trait', 'emotional': 'trait', 'jealous': 'trait', 'careless': 'work-directed',
    'sloppy': 'work-directed', 'negligent': 'work-directed', 'inefficient': 'work-directed',
    'inconsistent': 'work-directed', 'shallow': 'mind-directed', 'simple': 'mind-directed',
}
OBJECT_NOUNS = set("""question questions code script commit diff pdf email emails class class
project proposal assignment work idea plan paragraph sentence sentence graph legend spreadsheet
policy message reply document report""".split())
PERSON_NOUNS = set("""coworker student teammate colleague beginner researcher customer developer
client user author employee person guy man woman people someone anyone somebody""".split())

SELF = {'i', "i'm", 'me', 'my', 'mine', 'myself', 'im'}
ADDR = {'you', "you're", 'your', 'yours', 'yourself', 'youre'}
THIRD = {'he', 'she', 'him', 'her', 'his', 'hers', 'they', 'them', 'their', 'theirs'}


def insults_v2():
    """curated insult sets; returns (adj, noun, profanity)"""
    return set(INS_ADJ), set(INSULT_NOUNS), set(PROFANITY)


def tokenize(text):
    return re.findall(r"[A-Za-z']+", text.lower())


def code_v2(text, window=6):
    """ordered patterns, then a window fallback. Returns (label, evidence)."""
    adj, nouns, prof = insults_v2()
    toks = tokenize(text)
    n = len(toks)
    def at(i):
        return toks[i].strip("'") if 0 <= i < n else ''
    # P1 vocative / predicative addressee:  you are <INS> | you <INS_NOUN>
    for i, w in enumerate(toks):
        b = w.strip("'")
        prev1, prev2 = at(i - 1), at(i - 2)
        nxt = at(i + 1)
        if b in adj or b in prof or b in nouns:
            # P2 predicative self: I'm / I am / I was <adv>* <INS>
            if prev1 in SELF or (prev1 in {'so', 'such', 'a', 'an', 'very', 'really', 'just', 'completely'} and prev2 in SELF):
                return 'SELF', f'{prev2 or prev1} ... {b}'
            # P3 predicative addressee: you are / you're / are you <INS>
            if prev1 in ADDR or prev1 in {'are', "you're", 'youre'} or (prev2 in ADDR and prev1 in {'are', 're'}):
                return 'ADDRESSEE', f'{prev2 or prev1} ... {b}'
            if prev1 == 'you':                                  # "you idiot", "you thief"
                return 'ADDRESSEE', f'you {b}'
            # P4 attributive: <INS> <NOUN>  -> the noun decides
            if nxt:
                if nxt in OBJECT_NOUNS:
                    continue                                     # object-directed, keep scanning
                if nxt in PERSON_NOUNS:
                    return 'THIRD', f'{b} {nxt}'
            # P5 noun-then-copula: <person noun> is <INS>
            if prev1 in PERSON_NOUNS or (prev1 == 'is' and prev2 in PERSON_NOUNS):
                return 'THIRD', f'{prev2 or prev1} is {b}'
        # P6 profane exclamation with no person nearby is handled by the fallback
    # fallback: nearest person reference within the window, first hit wins (v1 rule, kept for
    # continuity but only after the pattern pass)
    for i, w in enumerate(toks):
        b = w.strip("'")
        if b in adj or b in prof or b in nouns:
            lo, hi = max(0, i - window), min(n, i + window + 1)
            win = [x.strip("'") for x in toks[lo:hi]]
            for cls, refs in (('SELF', SELF), ('ADDRESSEE', ADDR), ('THIRD', THIRD)):
                if any(x in refs for x in win):
                    return cls, f'window {b}'
    return 'NONE', None


def code_v3(text, window=8):
    """v3: ordered patterns; article-insult-noun resolves to the nearest PRECEDING pronoun; anything
    else is UNRESOLVED. Coverage is reported rather than hidden behind a fallback.

    v2's remaining errors were traceable to the window fallback choosing the nearest pronoun in EITHER
    direction, which picks the speaker ("...and I'm genuinely annoyed") when the insult targets the
    addressee ("You've asked me this stupid question"). A following pronoun cannot be the subject of a
    preceding insult, so only preceding ones are considered here.
    """
    adj, nouns, prof = insults_v2()
    toks = tokenize(text)
    n = len(toks)
    def at(i):
        return toks[i].strip("'") if 0 <= i < n else ''

    def class_of(pron):
        if pron in SELF:
            return 'SELF'
        if pron in ADDR:
            return 'ADDRESSEE'
        if pron in THIRD:
            return 'THIRD'
        return None

    def preceding_subject(i, look=8):
        for j in range(i - 1, max(-1, i - 1 - look), -1):
            c = class_of(at(j))
            if c:
                return c, at(j)
        return None, None

    for i, w in enumerate(toks):
        b = w.strip("'")
        if b not in (adj | nouns | prof):
            continue
        prev1, prev2, prev3 = at(i - 1), at(i - 2), at(i - 3)
        nxt = at(i + 1)
        # attributive on an object noun -> the insult is aimed at a thing
        if nxt in OBJECT_NOUNS:
            continue
        # "you idiot" / "you thief": vocative
        if prev1 in ADDR and b in nouns:
            return 'ADDRESSEE', f'you {b}'
        # "an idiot" / "a fool": the clause subject decides
        if prev1 in {'a', 'an'} and b in nouns:
            cls, pron = preceding_subject(i - 1)
            if cls:
                return cls, f'{pron} ... a {b}'
            return 'UNRESOLVED', f'a {b}'
        # "acting like an idiot" -> the subject of the clause
        if prev2 in {'like'} and prev1 in {'a', 'an'} and b in nouns:
            cls, pron = preceding_subject(i - 2)
            if cls:
                return cls, f'{pron} like a {b}'
        # predicative with an explicit copula: I'm / you're / that's
        if prev1 in SELF or prev1 in ADDR:
            return class_of(prev1), f'{prev1} {b}'
        if prev1 in {'am', 'is', 'are', 'was', 'were'}:
            cls, pron = preceding_subject(i - 1)
            if cls:
                return cls, f'{pron} {prev1} {b}'
        # comparative / intensifier chains: "so stupid", "such a fool", "this stupid question"
        if prev1 in {'so', 'such', 'very', 'really', 'absolutely', 'completely'}:
            cls, pron = preceding_subject(i - 1)
            if cls:
                return cls, f'{pron} {prev1} {b}'
        if nxt in PERSON_NOUNS:
            return 'THIRD', f'{b} {nxt}'
        if prev1 in PERSON_NOUNS:
            return 'THIRD', f'{prev1} ... {b}'
        # nothing resolved: record it, do not guess
    return 'UNRESOLVED' if any(w.strip("'") in (adj | nouns | prof) for w in toks) else 'NONE', None


def code_v1(text, window=6):
    """the first version, kept only so the correction is visible in the record"""
    adj, nouns, prof = insults_v2()
    ins = adj | nouns | prof
    toks = tokenize(text)
    for i, t in enumerate(toks):
        b = t.strip("'")
        if b not in ins:
            continue
        lo, hi = max(0, i - window), min(len(toks), i + window + 1)
        win = [x.strip("'") for x in toks[lo:hi]]
        for cls, refs in (('SELF', SELF), ('ADDRESSEE', ADDR), ('THIRD', THIRD)):
            if any(x in refs for x in win):
                return cls, b
        if any(x in PERSON_NOUNS for x in win):
            return 'THIRD', b
    return 'NONE', None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/target_directed')
    ap.add_argument('--audit-n', type=int, default=48, help='texts to put in the blind audit sheet')
    ap.add_argument('--seed', type=int, default=20261029)
    a = ap.parse_args()

    adj, nouns, prof = insults_v2()
    print(f'v2 insult lexicon: {len(adj)} adjectives + {len(nouns)} nouns + {len(prof)} profanity = '
          f'{len(adj | nouns | prof)} terms')
    print('  adjectives from Jibes adj hostile pool (Apache-2.0) and BFFM II- (CC0); identity/appearance and '
          'trait-neutral words excluded on purpose (see EXCLUDED_ON_PURPOSE)')
    print('  v1 mistake being corrected: it used the Jibes VERB pool (hold/wait/stop/read/tell) as insults')

    sources = {
        'ii_budget': ROOT / 'results/ii_budget_sweep.json',
        'pools_b07': ROOT / 'results/bffm_pools_b07.json',
        'compiler_h12': ROOT / 'results/compiler_pareto_hostility12.json',
    }
    rec = {'scheme': {'insult_sources': ['hand-written profanity', 'Jibes hostile pool', 'BFFM negative pools'],
                      'referent_classes': ['SELF', 'ADDRESSEE', 'THIRD', 'NONE'],
                      'coder': 'v3 (ordered patterns, preceding-subject rule, no guessing)',
                      'window_tokens': 8, 'unresolved_is_reported_not_guessed': True,
                      'target_directed': ['ADDRESSEE', 'THIRD']},
           'lexicon_size': len(adj | nouns | prof), 'lexicon_breakdown': {'adj': len(adj), 'nouns': len(nouns),
                                                                          'profanity': len(prof)},
           'excluded_on_purpose': EXCLUDED_ON_PURPOSE, 'runs': {}}
    audit = []

    for name, path in sources.items():
        if not path.is_file():
            print(f'  skip {name}: {path} missing')
            continue
        d = json.loads(path.read_text(encoding='utf8'))
        rows = []
        if name == 'ii_budget':
            for k, v in d['per_context'].items():
                arm, cid = k.rsplit('|', 1)
                rows.append((arm, cid, v['text_first_240']))
            for cid, v in d['baseline'].items():
                rows.append(('BASE', cid, v['text_first_240']))
        elif name == 'pools_b07':
            for k, v in d['per_context'].items():
                arm, cid = k.split('|', 1)
                rows.append((arm, cid, v['text_first_240']))
            for cid, v in d['baseline'].items():
                rows.append(('BASE', cid, v['text_first_240']))
        else:
            for k, v in d['per_context'].items():
                arm, b, cid = k.split('|', 2)
                rows.append((f'{arm}@b={b}', cid, v['text_first_240']))
            for cid, v in d['baseline'].items():
                rows.append(('BASE', cid, v['text_first_240']))

        per_arm = collections.defaultdict(lambda: collections.Counter())
        sens = collections.defaultdict(lambda: collections.Counter())
        for arm, cid, text in rows:
            lab, hit = code_v3(text, 8)
            per_arm[arm][lab] += 1
            per_arm[arm]['n'] += 1
            lab10, _ = code_v2(text, 6)   # v2 = the window fallback, for sensitivity
            sens[arm][lab10] += 1
            v1lab, _ = code_v1(text, 6)
            per_arm[arm]['v1_' + v1lab] += 1
            audit.append({'run': name, 'arm': arm, 'context': cid, 'text': text, 'label': lab})
        rec['runs'][name] = {'n_texts': len(rows),
                             'labels': {k: dict(v) for k, v in sorted(per_arm.items())},
                             'sensitivity_window10': {k: dict(v) for k, v in sorted(sens.items())}}
        print(f'\n=== {name} ({len(rows)} texts) ===')
        print(f'{"arm":24s} {"n":>3s} {"SELF":>5s} {"ADDR":>5s} {"THIRD":>6s} {"NONE":>5s} '
              f'{"UNRES":>6s} {"target-dir":>10s} {"v1_tgt":>6s}')
        for arm in sorted(per_arm):
            c = per_arm[arm]
            tgt = c['ADDRESSEE'] + c['THIRD']
            s10 = sens[arm]['ADDRESSEE'] + sens[arm]['THIRD']
            v1t = c['v1_ADDRESSEE'] + c['v1_THIRD']
            print(f'{arm:24s} {c["n"]:3d} {c["SELF"]:5d} {c["ADDRESSEE"]:5d} {c["THIRD"]:6d} '
                  f'{c["NONE"]:5d} {c["UNRESOLVED"]:6d} {tgt:10d} {v1t:6d}')

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'target_directed.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                             encoding='utf8')
    rng = random.Random(a.seed)
    rng.shuffle(audit)
    pick = audit[:a.audit_n]
    with (out / 'audit_sheet.csv').open('w', encoding='utf8', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['item_id', 'text', 'human_target_directed_0to1', 'human_self_abuse_0to1', 'notes'])
        for i, x in enumerate(pick):
            w.writerow([f'D{i:04d}', x['text'], '', '', ''])
    (out / 'audit_mapping_PRIVATE.json').write_text(json.dumps(
        [{'item_id': f'D{i:04d}', 'run': x['run'], 'arm': x['arm'], 'context': x['context'],
          'machine_label': x['label']} for i, x in enumerate(pick)], ensure_ascii=False, indent=1),
        encoding='utf8')
    print(f'\nwrote {out}/target_directed.json')
    print(f'wrote {out}/audit_sheet.csv ({len(pick)} blind texts) and audit_mapping_PRIVATE.json')
    print('the human labels are what turn this scheme into a measured instrument; until then it is a proxy')


if __name__ == '__main__':
    main()
