#!/usr/bin/env python3
"""Find a budget where the SPECIFIC hostility direction is visible AND affordable. (plan 2.30.4 items 1-3)

Why. Plan 2.30 established that the antagonism direction (BFFM II- vs II+) is behaviourally specific:
at b=0.7 it drives P(toxic) to 0.9875 while I-/III-/IV-/V- stay at 0.0006-0.1655, at comparable
capability cost. But b=0.7 costs 1.44 nat of reference logP, so there is still no USABLE operating
point. This run answers the only question that matters for the deliverable: does a budget exist
where the specific direction is visible to an independent gauge and still cheap?

It also answers, in the same run and for the same contexts, how far the famous coordinate sits from
this specific direction: cos(u_II-, u_known) and the two arms at the same budget.

Arms:
  u_II-     the specific direction rebuilt here (12 frames x 12 contexts of gradients)
  u_known   the community recipe as a dose list, L14N8949 +0.7 / L13N7956 -0.7, taken as a vector for
            the cosine and as-is for generation
  BASE      no steering

Budgets swept for u_II-: 0.05, 0.2, 0.3, 0.4, 0.5, 0.7   (0.05 = our certified clean band, 0.7 = plan 2.30)

FROZEN reading (before the run):
  * report, per budget: mean P(toxic), n of 12 above 0.5, dlogP(REFERENCE_TEXT), rep4, and target-directed
    profanity counts;
  * the USABLE POINT exists iff some budget has P(toxic) > 0.5 AND |dlogP| < 0.3
    (the 0.3 nat tolerance is stated here, before the numbers, and is not moved afterwards);
  * report cos(u_II-, u_known) and their top-64 overlap, plus a paired comparison of the two arms at
    b=0.7 where both are measurable;
  * no budget is chosen because it looks best: the rule above decides.

    [i/N] elapsed, eta ...   per measurement (package progress.Progress)
"""
from __future__ import annotations
import argparse, json, math, platform, re, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PROF = r"(fuck|shit|bitch|bastard|asshole|damn|crap|cunt|idiot|stupid|worthless|pathetic|loser)"
SELF_PAT = re.compile(r"\b(i'?m|i am|i was|i've|my|me)\b[^.!?]{0,40}" + PROF, re.I)
YOU_PAT = re.compile(r"\byou\b[^.!?]{0,40}" + PROF, re.I)


def resolve_pkg(given):
    cands = [Path(given)] if given else []
    cands += [ROOT / 'vendor/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
              Path('/root/autodl-tmp/scripts/j2space')]
    for c in cands:
        if (c / 'engine.py').is_file() and (c / 'semantic.py').is_file():
            return c
    raise SystemExit('package not found; pass --pkg')


def text_metrics(t):
    toks = re.findall(r"\w+|[^\w\s]", t.lower())
    n = len(toks)
    if n < 5:
        return {'rep4': 0.0, 'distinct2': 1.0}
    qg = [tuple(toks[i:i + 4]) for i in range(n - 3)]
    bg = [tuple(toks[i:i + 2]) for i in range(n - 1)]
    return {'rep4': 1 - len(set(qg)) / len(qg), 'distinct2': len(set(bg)) / len(bg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--bffm', default=str(ROOT / 'lexicons/bffm100_factors.json'))
    ap.add_argument('--census', default='')
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--gauges', default='/root/autodl-tmp/gauges')
    ap.add_argument('--layers', default='13-14')
    ap.add_argument('--frame', default='You are so {w}.')
    ap.add_argument('--budgets', default='0.05,0.2,0.3,0.4,0.5,0.7')
    ap.add_argument('--known', default='14:8949:+0.7,13:7956:-0.7')
    ap.add_argument('--tolerance', type=float, default=0.3,
                    help='capability tolerance in nat of reference logP for the "usable point" rule')
    ap.add_argument('--trait', default='hostility')
    ap.add_argument('--lang', default='en', choices=['en', 'zh', 'all'])
    ap.add_argument('--contexts', type=int, default=12)
    ap.add_argument('--max-new', type=int, default=64)
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--out', default='out/ii_budget')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    budgets = [float(x) for x in a.budgets.split(',')]
    known = []
    for spec in a.known.split(','):
        l, n, d = spec.split(':')
        known.append({'layer': int(l), 'neuron': int(n), 'dose': float(d)})

    bffm = json.loads(Path(a.bffm).read_text(encoding='utf8'))
    cands = [Path(a.census)] if a.census else []
    cands += [Path(__file__).resolve().parent / '07_results/slot_census_bffm100.json',
              ROOT / '07_results/slot_census_bffm100.json']
    cen = next((c for c in cands if c.is_file()), None)
    if cen is None:
        raise SystemExit('need the plan 2.19 census')
    single = {t for p in json.loads(cen.read_text(encoding='utf8'))['pools'].values()
              for t in p['survivors']}
    neg = [w for w in bffm['pools']['II-'] if w in single]
    pos = [w for w in bffm['pools']['II+'] if w in single]

    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    def _zh(s):
        return any('\u4e00' <= ch <= '\u9fff' for ch in s)
    sel = [r for r in rows if r.get('trait') == a.trait]
    if a.lang != 'all':
        sel = [r for r in sel if _zh(r['prompt']) == (a.lang == 'zh')]
    ctx = sel[:a.contexts]
    if not ctx:
        raise SystemExit('no contexts')

    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
    n_grad = len(ctx) * (len(neg) + len(pos))
    n_gen = len(ctx) * (len(budgets) + 1) + len(ctx)

    print('Question : is there a budget where the specific hostility direction is visible AND affordable?')
    print(f'Direction: II- ({len(neg)} words) vs II+ ({len(pos)} words) in frame {a.frame!r}')
    print(f'Budgets  : {budgets}   (plus the community recipe as an anchor)')
    print(f'Contexts : {len(ctx)} ({a.trait}/{a.lang}) -> {[c["id"] for c in ctx]}')
    print(f'Work     : {n_grad} forward/backward + {n_gen} generation/capability pairs')
    print(f'Frozen   : a USABLE POINT exists iff some budget has P(toxic) > 0.5 AND |dlogP| < {a.tolerance} nat;'
          ' cos(u_II-, u_known) and top-64 overlap are reported; no budget is picked for looking best.')
    if a.dry_run:
        print(f'[dry-run] would write {a.out}/ii_budget.json and .md')
        return

    import torch
    import torch.nn.functional as F
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import (REFERENCE_TEXT, apply_steering, candidate_ids, capture_mlp_inputs,
                        column_norms, continuation_logp, estimate_p50, model_lm, prompt_prefix,
                        score_for_dose)
    from progress import Progress

    dev = 'cuda:0'
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to(dev).eval()
    model.requires_grad_(False)
    lm = model_lm(model)
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device=dev)
    print('p50:', {l: round(p50[l], 6) for l in layers})
    What = {}
    for l in layers:
        Wt = lm.layers[l].mlp.down_proj.weight.detach().float().T
        What[l] = Wt / Wt.norm(dim=1, keepdim=True).clamp(min=1e-12)
    B = {l: What[l].T.double() @ What[l].double() for l in layers}
    npl = What[layers[0]].shape[0]

    prog = Progress(n_grad + n_gen)

    def grad_of(prefix_ids, text):
        ids = torch.tensor([prefix_ids + tok(text, add_special_tokens=False).input_ids],
                           dtype=torch.long, device=dev)
        pre = len(prefix_ids); suf = ids.shape[1] - pre
        with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
            out = model(input_ids=ids, use_cache=False)
            logits = out.logits[:, pre - 1:pre - 1 + suf]
            target = ids[:, pre:pre + suf]
            lp = -F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                  target.reshape(-1), reduction='mean')
            g = torch.autograd.grad(lp, [cap[l] for l in layers])
        return {l: score_for_dose(gg.detach().float().sum(dim=tuple(range(gg.ndim - 1))),
                                  p50[l], norms[l].to(gg.device)).double().cpu()
                for l, gg in zip(layers, g)}

    G = {l: torch.zeros(npl, dtype=torch.float64) for l in layers}
    for cx in ctx:
        pid = prompt_prefix(tok, cx['prompt'])
        acc = {l: torch.zeros(npl, dtype=torch.float64) for l in layers}
        for words, sgn in ((neg, +1.0), (pos, -1.0)):
            for w in words:
                g = grad_of(pid, a.frame.format(w=w))
                for l in layers:
                    acc[l] += sgn * g[l] / len(words)
                prog.step(f'{cx["id"]} {"II-" if sgn > 0 else "II+"} ({len(words)} words)')
        for l in layers:
            G[l] += acc[l] / len(ctx)

    e = {l: What[l].T.double().to(dev) @ G[l].to(dev) for l in layers}
    ds = {l: torch.linalg.solve(B[l], e[l]) for l in layers}
    raw = {l: What[l].double() @ torch.linalg.solve(B[l], ds[l]) for l in layers}
    bu = math.sqrt(sum(float((raw[l].float() @ What[l]).pow(2).sum()) for l in layers))
    u_ii = {l: (raw[l] / bu).cpu() for l in layers}      # keep the deliverable on CPU

    # the community recipe as a vector, M-normalised, for the cosine (its own doses are literal)
    u_known = {l: torch.zeros(npl, dtype=torch.float64) for l in layers}
    for it in known:
        u_known[it['layer']][it['neuron']] = it['dose']
    bku = math.sqrt(sum(float((u_known[l].float().to(dev) @ What[l]).pow(2).sum()) for l in layers))
    u_known_n = {l: (u_known[l] / bku).cpu() for l in layers}
    cos = float(torch.cat([u_ii[l] for l in layers]) @ torch.cat([u_known_n[l] for l in layers]))
    t1 = set(torch.topk(torch.cat([u_ii[l].abs() for l in layers]), 64).indices.tolist())
    t2 = set(torch.nonzero(torch.cat([u_known[l] for l in layers]).abs() > 0).flatten().tolist())
    print(f'cos(u_II-, u_known) = {cos:+.4f}   |known| has {len(t2)} nonzero coordinates, '
          f'and the II- top-64 contains {len(t1 & t2)} of them')

    gauges = {}
    from transformers import AutoModelForSequenceClassification
    for name, sub, kind in (('tox_bert', 'unitary_toxic-bert', 'sigmoid0'),
                            ('tox_snlp', 's-nlp_roberta_toxicity_classifier', 'softmax1')):
        d = Path(a.gauges) / sub
        gauges[name] = (AutoTokenizer.from_pretrained(d, local_files_only=True),
                        AutoModelForSequenceClassification.from_pretrained(
                            d, local_files_only=True).to(dev).eval(), kind)

    def p_toxic(texts):
        out = {k: [] for k in gauges}
        for k, (tk, m, kind) in gauges.items():
            for i in range(0, len(texts), 8):
                enc = tk(texts[i:i + 8], return_tensors='pt', padding=True, truncation=True,
                         max_length=256).to(dev)
                with torch.inference_mode():
                    lg = m(**enc).logits.float()
                p = torch.sigmoid(lg[:, 0]) if kind == 'sigmoid0' else F.softmax(lg, -1)[:, 1]
                out[k].extend(float(x) for x in p)
        return out

    def run(iv, prompt):
        rid, rpre, rsuf = candidate_ids(tok, prompt, REFERENCE_TEXT, dev, a.max_seq)
        with apply_steering(lm, iv, p50, norms), torch.inference_mode():
            lp = float(continuation_logp(model, rid, rpre, rsuf))
            ids = torch.tensor([prompt_prefix(tok, prompt)], dtype=torch.long, device=dev)
            o = model.generate(input_ids=ids, do_sample=False, max_new_tokens=a.max_new,
                               use_cache=True,
                               pad_token_id=tok.pad_token_id if tok.pad_token_id is not None
                               else tok.eos_token_id)
            text = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
        p = p_toxic([text])
        return {'ref_logp': lp, 'text_first_240': text[:240], **{k: v[0] for k, v in p.items()},
                **text_metrics(text),
                'self_directed': bool(SELF_PAT.search(text)),
                'you_directed': bool(YOU_PAT.search(text))}

    def iv_of(d, scale):
        return [{'layer': l, 'neuron': j, 'dose': float(d[l][j] * scale)}
                for l in layers for j in torch.nonzero(d[l].abs() > 1e-15).flatten().tolist()]

    base = {}
    for cx in ctx:
        base[cx['id']] = run([], cx['prompt'])
        prog.step(f'base {cx["id"]}')
    res = {}
    for b in budgets:
        iv = iv_of(u_ii, b)
        for cx in ctx:
            res[(f'II-@b={b}', cx['id'])] = run(iv, cx['prompt'])
            prog.step(f'II- b={b} {cx["id"]} p_tox={res[(f"II-@b={b}", cx["id"])]["tox_bert"]:.3f}')
    for cx in ctx:
        res[('known@0.7', cx['id'])] = run(known, cx['prompt'])
        prog.step(f'known {cx["id"]}')

    cids = [c['id'] for c in ctx]
    def block(label):
        rows_ = [res[(label, c)] for c in cids]
        d = {}
        for key in ('tox_bert', 'tox_snlp', 'ref_logp', 'rep4', 'distinct2'):
            mb = [base[c][key] for c in cids]
            st = [r[key] for r in rows_]
            diff = [x - y for x, y in zip(st, mb)]
            d[key] = {'mean_base': statistics.fmean(mb), 'mean_steer': statistics.fmean(st),
                      'mean_delta': statistics.fmean(diff), 'range': [min(diff), max(diff)]}
        d['n_pos'] = sum(1 for r in rows_ if r['tox_bert'] >= 0.5)
        d['n_you'] = sum(1 for r in rows_ if r['you_directed'])
        d['n_self'] = sum(1 for r in rows_ if r['self_directed'])
        return d

    an = {f'II-@b={b}': block(f'II-@b={b}') for b in budgets}
    an['known@0.7'] = block('known@0.7')

    usable = [(b, an[f'II-@b={b}']['tox_bert']['mean_steer'],
               an[f'II-@b={b}']['ref_logp']['mean_delta']) for b in budgets]
    point = [(b, tox, dl) for b, tox, dl in usable if tox > 0.5 and abs(dl) < a.tolerance]

    rec = {'schema': 'qwentest-ii-budget-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'layers': layers,
           'frame': a.frame, 'budgets': budgets, 'contexts': cids, 'trait': a.trait,
           'tolerance_nat': a.tolerance, 'ii_neg_words': neg, 'ii_pos_words': pos,
           'cos_ii_known': cos, 'known_nonzero': len(t2), 'ii_top64_hits_on_known': len(t1 & t2),
           'analysis': an, 'baseline': base,
           'usable_point_exists': bool(point), 'usable_points': point,
           'per_context': {f'{lab}|{c}': v for (lab, c), v in res.items()}}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'ii_budget.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                        encoding='utf8')
    L = ['# Budget sweep of the specific hostility direction (plan 2.30.4 items 1-3)', '',
         f'- machine `{rec["machine"]}` / {rec["gpu"]} · layers {layers} · frame {a.frame!r} · '
         f'{len(cids)} {a.trait}/{a.lang} contexts',
         f'- II- words: {neg}', f'- II+ words: {pos}',
         f'- baseline P(toxic) mean {statistics.fmean(base[c]["tox_bert"] for c in cids):.5f}', '',
         '| arm | P(tox) mean | n>=0.5 | dlogP ref | rep4 | you-dir | self-dir |',
         '|---|---:|---:|---:|---:|---:|---:|']
    for lab, d in an.items():
        L.append(f'| `{lab}` | {d["tox_bert"]["mean_steer"]:.4f} | {d["n_pos"]}/{len(cids)} | '
                 f'{d["ref_logp"]["mean_delta"]:+.4f} | {d["rep4"]["mean_steer"]:.3f} | '
                 f'{d["n_you"]}/{len(cids)} | {d["n_self"]}/{len(cids)} |')
    L += ['', '## the frozen rule', '',
          f'a usable point exists iff P(toxic) > 0.5 and |dlogP| < {a.tolerance} nat', '',
          f'- points that satisfy it: **{point if point else "NONE"}**', '',
          '## how far is the famous coordinate from this direction?', '',
          f'- cos(u_II-, u_known) = **{cos:+.4f}**',
          f'- u_known has {len(t2)} nonzero coordinates; the II- top-64 contains {len(t1 & t2)} of them',
          f'- both arms measured on the same contexts; compare the `known@0.7` row with `II-@b=0.7`']
    (out / 'ii_budget.md').write_text('\n'.join(L) + '\n', encoding='utf8')
    print('\n' + '\n'.join(L))
    print(f'\nwrote {out}/ii_budget.json and .md')


if __name__ == '__main__':
    main()
