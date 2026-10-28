#!/usr/bin/env python3
"""Census: how many Goldberg-1992 marker adjectives survive the J2 slot constraint?

The cross-trait specificity test (plan 06 §8, still open) needs a SECOND trait measured
on the SAME instrument. `lexicons/bffm100_factors.json` gives ten pools of evaluative
trait adjectives from one public source (factor x key), so all five trait axes share one
register by construction. Before any GPU run, this script answers the only question that
can kill the design cheaply: after the frozen slot filter, are the pools still big enough?

Slot filter -- byte-identical to the frozen pipeline, `build_lexicon.py::ntoks`:

    len(Tokenizer.from_file(tokenizer.json).encode(' ' + w).tokens) == 1

i.e. leading space, single token. Nothing else is filtered here: no POS test (the source
is already adjectives), no frame test (no corpus), no hand-picking. Terms that fail are
reported with their piece count so the loss is auditable rather than silent.

CPU only; no torch, no model weights. Seconds, not minutes.

    [i/N] elapsed, eta ...   per term (package progress.Progress)
"""
from __future__ import annotations
import argparse, hashlib, json, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # qwentest/


def resolve_pkg(given):
    cands = [Path(given)] if given else []
    cands += [ROOT / '03_unpacked/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
              Path('/root/autodl-tmp/qwen-gradient-discovery-v6')]
    for c in cands:
        if (c / 'progress.py').is_file():
            return c
    raise SystemExit('package not found; pass --pkg <dir with progress.py>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/bffm100_factors.json'))
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--tokenizer', default='', help='tokenizer.json; default <model>/tokenizer.json')
    ap.add_argument('--out', default='out/slot_census_bffm')
    ap.add_argument('--min-pool', type=int, default=6,
                    help='a pool below this many survivors cannot be split into two disjoint '
                         'halves of >=3, which G1 needs; default 6')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lex = json.loads(Path(a.lexicon).read_text(encoding='utf8'))
    pools = lex['pools']
    terms = sorted(lex['terms'])
    tok_path = a.tokenizer or str(Path(a.model) / 'tokenizer.json')

    print(f'Lexicon : {Path(a.lexicon).name}  ({lex["derived"]}) '
          f'[{lex["provenance"]["factor_and_key"]["doi"]}]')
    print(f'Pools   : ' + '  '.join(f'{k}={len(v)}' for k, v in sorted(pools.items())))
    print(f'Terms   : {len(terms)} unique adjectives; tokenizer {tok_path}')
    print(f'Filter  : len(tok.encode(" " + w).tokens) == 1  (same call as build_lexicon.py)')
    print(f'Work    : {len(terms)} encode calls; CPU, no model load')
    if a.dry_run:
        print('\n[dry-run] pools as they would be tested:')
        for k in sorted(pools):
            print(f'  {k:4s} ({len(pools[k]):2d}): ' + ', '.join(pools[k]))
        print(f'\n[dry-run] would write {a.out}/census.json and {a.out}/census.md')
        return

    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from progress import Progress
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def ntoks(w):
        return len(tok.encode(' ' + w).tokens)

    rows, prog = {}, Progress(len(terms))
    for w in terms:
        enc = tok.encode(' ' + w)
        rows[w] = {'n_tokens': len(enc.tokens), 'pieces': enc.tokens,
                   'ids': enc.ids, 'single_token': len(enc.tokens) == 1,
                   'bare_n_tokens': len(tok.encode(w).tokens),
                   **{k: lex['terms'][w][k] for k in ('factor', 'key', 'familiarity_prop')}}
        prog.step(f'{w} n={len(enc.tokens)}')

    surv = {k: sorted(t for t in v if rows[t]['single_token']) for k, v in pools.items()}
    lost = sorted(t for t in terms if not rows[t]['single_token'])

    usable, thin = [], []
    for k in sorted(surv):
        n = len(surv[k])
        (usable if n >= a.min_pool else thin).append(k)

    rec = {'schema': 'qwentest-slot-census-v1', 'lexicon': Path(a.lexicon).name,
           'lexicon_sha256': hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest(),
           'tokenizer': tok_path, 'tokenizer_vocab_size': tok.get_vocab_size(),
           'filter': "len(Tokenizer.encode(' ' + w).tokens) == 1",
           'n_terms': len(terms), 'n_single_token': sum(r['single_token'] for r in rows.values()),
           'n_multi_token': len(lost), 'min_pool': a.min_pool,
           'pools': {k: {'source_n': len(pools[k]), 'survivors': surv[k], 'n': len(surv[k]),
                         'halves': len(surv[k]) // 2,
                         'usable_for_G1': len(surv[k]) >= a.min_pool} for k in sorted(pools)},
           'usable_pools': usable, 'thin_pools': thin,
           'multi_token_terms': [{'term': t, 'n_tokens': rows[t]['n_tokens'],
                                  'pieces': rows[t]['pieces']} for t in lost],
           'terms': rows}
    (out / 'census.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1) + '\n',
                                     encoding='utf8')

    fam = {t: rows[t]['familiarity_prop'] for t in terms if rows[t]['familiarity_prop'] is not None}
    lines = ['# BFFM-100 slot census', '',
             f'- lexicon `{rec["lexicon"]}` sha256 `{rec["lexicon_sha256"][:16]}…`',
             f'- tokenizer `{tok_path}` (vocab {rec["tokenizer_vocab_size"]})',
             f'- filter `{rec["filter"]}`',
             f'- **{rec["n_single_token"]}/{rec["n_terms"]} adjectives are single tokens**; '
             f'{rec["n_multi_token"]} dropped', '',
             '| pool | source | survivors | halves | usable for G1 |', '|---|---:|---:|---:|---|']
    for k in sorted(pools):
        p = rec['pools'][k]
        lines.append(f'| `{k}` | {p["source_n"]} | {p["n"]} | {p["halves"]} | '
                     f'{"yes" if p["usable_for_G1"] else "**no**"} |')
    lines += ['', f'usable pools: {", ".join(usable) or "none"}',
              f'thin pools (<{a.min_pool}): {", ".join(thin) or "none"}', '',
              '## dropped terms (multi-token)', '']
    for t in lost:
        lines.append(f'- `{t}` -> {rows[t]["n_tokens"]} pieces {rows[t]["pieces"]}')
    if fam:
        s = [v for v in fam.values()]
        lines += ['', '## familiarity of the surviving terms (TDA `prop`, CC0)', '',
                  f'- median {statistics.median(s):.2f}, min {min(s):.2f}, max {max(s):.2f} '
                  f'(n={len(s)} with a value)']
    (out / 'census.md').write_text('\n'.join(lines) + '\n', encoding='utf8')

    print(f'\nsingle-token {rec["n_single_token"]}/{rec["n_terms"]}; dropped {len(lost)}: '
          + ', '.join(lost))
    for k in sorted(pools):
        p = rec['pools'][k]
        flag = '' if p['usable_for_G1'] else '  <-- TOO THIN'
        print(f'  {k:4s} {p["source_n"]:2d} -> {p["n"]:2d} (halves {p["halves"]}){flag}')
    print(f'wrote {out}/census.json and {out}/census.md')


if __name__ == '__main__':
    main()
