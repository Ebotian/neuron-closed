#!/usr/bin/env python3
"""Stage 0 of plan 2.28.6: can the borrowed toxicity gauge even SEE the construct we measure?

Why this runs before any steering. Plan 2.28 found that at b=0.05 none of three compilers moved
either toxicity classifier, while our own readout rose +0.50, and that the community recipe moved
the classifier only on 1 of 8 contexts. Two explanations remain and they cost very different
amounts of GPU to distinguish:

  (a) the compilers really do not change behaviour at that budget, or
  (b) the gauge is measuring a different construct than our readout.

If (b) holds, then our own probe material - the "hostile" slot continuations that J2 scores - must
already look NON-toxic to the classifier. That is checkable with no language model at all: just
classify the probe strings and the lexicon's slot words.

What this measures, per classifier and per language:
  * P(toxic) for every probe's `positive` and `negative` continuation;
  * the separation between the two (mean difference, and AUC over all positive/negative pairs);
  * the same for the lexicon's hostile vs neutral slot words placed in their frame;
  * the two classifiers' agreement on these texts.

FROZEN reading:
  report the AUC and the mean P(toxic) per side, split by language, for both classifiers.
  If AUC (positive > negative) is near 0.5 or both sides sit near 0, the gauge does not see our
  construct, and running steered generation against it would measure nothing.
  No threshold is chosen after the fact; AUC is threshold-free.

    [i/N] elapsed, eta ...   per text batch (package progress.Progress)
"""
from __future__ import annotations
import argparse, json, platform, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gauges', default='/root/autodl-tmp/gauges')
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--trait', default='hostility')
    ap.add_argument('--out', default='out/gauge_validity')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    rows = [r for r in rows if r.get('trait') == a.trait]
    lex = json.loads(Path(a.lexicon).read_text(encoding='utf8'))
    def zh(s):
        return any('\u4e00' <= ch <= '\u9fff' for ch in s)
    n_zh = sum(1 for r in rows if zh(r['prompt']))
    slot_texts = []
    for fam, d in lex['families'].items():
        for side, words in (('pos', d['pos']), ('neg', d['neg'])):
            for w in words:
                slot_texts.append((side, d['template'].format(w=w), fam))

    print('Question : does the borrowed toxicity gauge see the construct our readout measures?')
    print(f'Probes   : {len(rows)} {a.trait} rows ({n_zh} Chinese prompts, {len(rows) - n_zh} English)')
    print(f'           -> {len(rows)} positive + {len(rows)} negative continuations')
    print(f'Slots    : {len(slot_texts)} framed slot words from {Path(a.lexicon).name}')
    print(f'Gauges   : toxic-bert (sigmoid[0]=toxic) + s-nlp roberta (softmax[1]=toxic)')
    print(f'Work     : {(2 * len(rows) + len(slot_texts)) * 2} short forward passes, no language model')
    print('Frozen   : report AUC(positive > negative) and mean P(toxic) per side, split by language.')
    print('           AUC near 0.5 or both sides near 0 => the gauge does not see our construct and')
    print('           steered generation against it would measure nothing.')
    if a.dry_run:
        print(f'[dry-run] would write {a.out}/gauge_validity.json and .md')
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    gauges = {}
    for name, sub, kind in (('tox_bert', 'unitary_toxic-bert', 'sigmoid0'),
                            ('tox_snlp', 's-nlp_roberta_toxicity_classifier', 'softmax1')):
        d = Path(a.gauges) / sub
        if not d.is_dir():
            raise SystemExit(f'gauge model missing: {d}')
        gauges[name] = (AutoTokenizer.from_pretrained(d, local_files_only=True),
                        AutoModelForSequenceClassification.from_pretrained(
                            d, local_files_only=True).to(dev).eval(), kind)
    print('gauges loaded:', {k: v[2] for k, v in gauges.items()})

    import torch.nn.functional as F

    def score(texts):
        """P(toxic) for every text, both classifiers, one padded batch."""
        out = {k: [] for k in gauges}
        for k, (tk, m, kind) in gauges.items():
            for i in range(0, len(texts), 16):
                batch = texts[i:i + 16]
                enc = tk(batch, return_tensors='pt', padding=True, truncation=True,
                         max_length=256).to(dev)
                with torch.inference_mode():
                    logits = m(**enc).logits.float()
                if kind == 'sigmoid0':
                    p = torch.sigmoid(logits[:, 0])
                else:
                    p = F.softmax(logits, -1)[:, 1]
                out[k].extend(float(x) for x in p)
        return out

    def auc(pos, neg):
        """P(pos > neg) with ties counted as 0.5."""
        if not pos or not neg:
            return None
        s = 0.0
        for p in pos:
            for n in neg:
                s += 1.0 if p > n else (0.5 if p == n else 0.0)
        return s / (len(pos) * len(neg))

    rec = {'schema': 'qwentest-gauge-validity-v1', 'machine': platform.node(),
           'trait': a.trait, 'n_rows': len(rows), 'n_zh_prompts': n_zh,
           'gauges': {k: v[2] for k, v in gauges.items()}, 'blocks': {}}

    for label, sel in (('all', rows), ('zh', [r for r in rows if zh(r['prompt'])]),
                       ('en', [r for r in rows if not zh(r['prompt'])])):
        pos = [r['positive'] for r in sel]
        neg = [r['negative'] for r in sel]
        sp = score(pos)
        sn = score(neg)
        block = {}
        for k in gauges:
            block[k] = {'mean_pos': statistics.fmean(sp[k]), 'mean_neg': statistics.fmean(sn[k]),
                        'auc_pos_gt_neg': auc(sp[k], sn[k]),
                        'n_pos_at_0.5+': sum(1 for x in sp[k] if x >= 0.5),
                        'n_neg_at_0.5+': sum(1 for x in sn[k] if x >= 0.5), 'n': len(sel)}
        rec['blocks'][f'probe_continuations_{label}'] = block

    for fam in sorted(lex['families']):
        pos = [t for s, t, f in slot_texts if f == fam and s == 'pos']
        neg = [t for s, t, f in slot_texts if f == fam and s == 'neg']
        sp, sn = score(pos), score(neg)
        block = {}
        for k in gauges:
            block[k] = {'mean_pos': statistics.fmean(sp[k]), 'mean_neg': statistics.fmean(sn[k]),
                        'auc_pos_gt_neg': auc(sp[k], sn[k]),
                        'n_pos_at_0.5+': sum(1 for x in sp[k] if x >= 0.5),
                        'n_neg_at_0.5+': sum(1 for x in sn[k] if x >= 0.5), 'n': len(pos)}
        rec['blocks'][f'slot_words_{fam}'] = block

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'gauge_validity.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                             encoding='utf8')
    L = ['# Stage 0: can the borrowed toxicity gauge see our construct?', '',
         f'- machine `{rec["machine"]}` · trait `{a.trait}` · {len(rows)} probe rows '
         f'({n_zh} Chinese) · gauges {rec["gauges"]}',
         '- P(toxic) per text; AUC is threshold-free and counts ties as 0.5', '',
         '| block | gauge | mean P(pos) | mean P(neg) | AUC(pos>neg) | #pos>=0.5 | #neg>=0.5 | n |',
         '|---|---|---:|---:|---:|---:|---:|---:|']
    for name, block in rec['blocks'].items():
        for k, v in block.items():
            L.append(f'| `{name}` | {k} | {v["mean_pos"]:.4f} | {v["mean_neg"]:.4f} | '
                     f'{v["auc_pos_gt_neg"]:.3f} | {v["n_pos_at_0.5+"]} | {v["n_neg_at_0.5+"]} | {v["n"]} |')
    (out / 'gauge_validity.md').write_text('\n'.join(L) + '\n', encoding='utf8')
    print('\n' + '\n'.join(L))
    print(f'\nwrote {out}/gauge_validity.json and .md')


if __name__ == '__main__':
    main()
