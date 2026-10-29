#!/usr/bin/env python3
"""Same-register specificity at the only budget where the gauge has range: five BFFM pools.

Why this run. Plan 2.29 found that the borrowed toxicity classifiers are blind at our certified
budget (b=0.05) and have full range at b=0.7, where a dense antagonism-vs-warmth direction drives
P(toxic) from 0.0006 to 0.99. That makes the question this line has never been able to answer
testable: is the effect ANTAGONISM-SPECIFIC, or is it just strong negative evaluation?

Design. One instrument, one frame, one solver, one budget; only the WORD POOL changes. The pools
are Goldberg-1992 BFFM-100 factor/key groups (CC0 membership, paper Table 3 factors), each screened
to single tokens by the plan 2.19 census. For pool X the direction is

    u_X  ~  mean_contexts [ mean_{w in X-} grad logP(frame(w))  -  mean_{w in X+} grad logP(frame(w)) ]

then the SAME closed form (u = What B^-1 d*, M-normalised) and the SAME budget b=0.7.

Pools actually used (single-token survivors are quoted so the asymmetry is visible, not hidden):
    II-  cold demanding harsh rude selfish          (5)  vs II+  cooperative generous helpful kind pleasant sympathetic warm (7)
    I-   inhibited quiet reserved shy timid withdrawn (6)  vs I+   active bold daring energetic verbal vigorous (6)
    III- careless inconsistent inefficient negligent sloppy (5) vs III+ careful efficient neat organized practical prompt steady systematic thorough (9)
    IV-  anxious emotional fearful insecure jealous nervous (6) vs IV+ relaxed (1)   <- weak anchor, reported
    V-   shallow simple                              (2)  vs V+   artistic bright complex creative deep imaginative innovative intellectual philosophical (9)

FROZEN CRITERIA (registered here, before running; no threshold may move afterwards):
  CR1  E_tox(II-) exceeds E_tox(X-) for X in {I, III, IV, V} in at least 4 of the 5 comparisons
       (paired per context, exact two-sided sign test reported alongside).
  CR2  the fraction of replies containing SECOND-PERSON profanity (you-directed) is higher for II-
       than the mean of the other four pools.
  CR3  if IV- comes within 0.05 of II- on E_tox, the conclusion is "strong negative evaluation,
       not antagonism" and the specificity claim is not supported.
  Auxiliary, reported not judged: self-directed profanity counts (plan 2.29 showed the antagonism
  direction produced self-insults), rep4 and distinct-2 (degeneration), and R = dlogP of
  REFERENCE_TEXT (capability cost).

Gauge discipline (from plan 2.29 stage 0): English adjective-frame material only; the classifiers
are blind on Chinese and near-blind on the verb frame, so contexts are English-only.

    [i/N] elapsed, eta ...   per measurement (package progress.Progress)
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, re, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

POOLS = [('II', 'II-', 'II+'), ('I', 'I-', 'I+'), ('III', 'III-', 'III+'),
         ('IV', 'IV-', 'IV+'), ('V', 'V-', 'V+')]


def resolve_pkg(given):
    cands = [Path(given)] if given else []
    cands += [ROOT / '03_unpacked/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
              Path('/root/autodl-tmp/scripts/j2space')]
    for c in cands:
        if (c / 'engine.py').is_file() and (c / 'semantic.py').is_file():
            return c
    raise SystemExit('package not found; pass --pkg')


def sign_p(diffs):
    v = [d for d in diffs if abs(d) > 1e-12]
    n = len(v)
    if n == 0:
        return None, 0
    k = min(sum(1 for d in v if d > 0), sum(1 for d in v if d < 0))
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n), n


def text_metrics(t):
    toks = re.findall(r"\w+|[^\w\s]", t.lower())
    n = len(toks)
    if n < 5:
        return {'rep4': 0.0, 'distinct2': 1.0, 'n_tok': n}
    qg = [tuple(toks[i:i + 4]) for i in range(n - 3)]
    bg = [tuple(toks[i:i + 2]) for i in range(n - 1)]
    return {'rep4': 1 - len(set(qg)) / len(qg), 'distinct2': len(set(bg)) / len(bg), 'n_tok': n}


PROF = r"(fuck|shit|bitch|bastard|asshole|damn|crap|cunt|idiot|stupid|worthless|pathetic|loser)"
SELF_PAT = re.compile(r"\b(i'?m|i am|i was|i've|my|me)\b[^.!?]{0,40}" + PROF, re.I)
YOU_PAT = re.compile(r"\byou\b[^.!?]{0,40}" + PROF, re.I)


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
    ap.add_argument('--budget', type=float, default=0.7)
    ap.add_argument('--trait', default='hostility')
    ap.add_argument('--lang', default='en', choices=['en', 'zh', 'all'])
    ap.add_argument('--contexts', type=int, default=12)
    ap.add_argument('--max-new', type=int, default=64)
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--out', default='out/bffm_pools')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    bffm = json.loads(Path(a.bffm).read_text(encoding='utf8'))
    cands = [Path(a.census)] if a.census else []
    cands += [Path(__file__).resolve().parent / '07_results/slot_census_bffm100.json',
              ROOT / '07_results/slot_census_bffm100.json']
    cen = next((c for c in cands if c.is_file()), None)
    if cen is None:
        raise SystemExit('need the plan 2.19 census (07_results/slot_census_bffm100.json)')
    single = {t for p in json.loads(cen.read_text(encoding='utf8'))['pools'].values()
              for t in p['survivors']}

    pools = {}
    for name, neg, pos in POOLS:
        nw = [w for w in bffm['pools'][neg] if w in single]
        pw = [w for w in bffm['pools'][pos] if w in single]
        if not nw or not pw:
            raise SystemExit(f'pool {name}: empty after the single-token filter')
        pools[name] = {'neg_key': neg, 'pos_key': pos, 'neg': nw, 'pos': pw}

    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    def _zh(s):
        return any('\u4e00' <= ch <= '\u9fff' for ch in s)
    sel = [r for r in rows if r.get('trait') == a.trait]
    if a.lang != 'all':
        sel = [r for r in sel if _zh(r['prompt']) == (a.lang == 'zh')]
    ctx = sel[:a.contexts]
    if not ctx:
        raise SystemExit('no contexts')

    n_grad = len(ctx) * sum(len(p['neg']) + len(p['pos']) for p in pools.values())
    n_gen = len(pools) * len(ctx) + len(ctx)
    print('Question : is the effect antagonism-specific, or just strong negative evaluation?')
    print(f'Instrument: one frame {a.frame!r}, one solver, one budget {a.budget}, five BFFM pools')
    for name, p in pools.items():
        print(f'  {name:4s} {p["neg_key"]:4s} ({len(p["neg"])}) vs {p["pos_key"]:4s} ({len(p["pos"])})')
    print(f'Contexts : {len(ctx)} ({a.trait}, lang {a.lang}) -> {[c["id"] for c in ctx]}')
    print(f'Work     : {n_grad} forward/backward passes + {n_gen} generations + {n_gen} capability forwards')
    print('Frozen   : CR1 E_tox(II-) beats the other four in >=4/5; CR2 you-directed profanity higher for II-;'
          ' CR3 if IV- is within 0.05 of II- then the claim is "strong negative evaluation", not antagonism.')
    if a.dry_run:
        print(f'[dry-run] would write {a.out}/bffm_pools.json and .md')
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
    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
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

    G = {name: {l: torch.zeros(npl, dtype=torch.float64) for l in layers} for name in pools}
    for cx in ctx:
        pid = prompt_prefix(tok, cx['prompt'])
        per_pool = {name: {l: torch.zeros(npl, dtype=torch.float64) for l in layers} for name in pools}
        for name, p in pools.items():
            for side, words, sgn in (('neg', p['neg'], +1.0), ('pos', p['pos'], -1.0)):
                acc = {l: torch.zeros(npl, dtype=torch.float64) for l in layers}
                for w in words:
                    g = grad_of(pid, a.frame.format(w=w))
                    for l in layers:
                        acc[l] += g[l]
                for l in layers:
                    per_pool[name][l] += sgn * acc[l] / len(words)
                prog.step(f'{cx["id"]} {name} {side} ({len(words)} words)')
        for name in pools:
            for l in layers:
                G[name][l] += per_pool[name][l] / len(ctx)

    dirs = {}
    for name in pools:
        e = {l: What[l].T.double().to(dev) @ G[name][l].to(dev) for l in layers}
        ds = {l: torch.linalg.solve(B[l], e[l]) for l in layers}
        u = {l: What[l].double() @ torch.linalg.solve(B[l], ds[l]) for l in layers}
        bu = math.sqrt(sum(float((u[l].float() @ What[l]).pow(2).sum()) for l in layers))
        dirs[name] = {l: u[l] / bu for l in layers}

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
        return {'ref_logp': lp, 'text_first_240': text[:240],
                **{k: v[0] for k, v in p.items()}, **text_metrics(text),
                'self_directed': bool(SELF_PAT.search(text)), 'you_directed': bool(YOU_PAT.search(text))}

    base = {}
    for cx in ctx:
        base[cx['id']] = run([], cx['prompt'])
        prog.step(f'base {cx["id"]}')
    res = {}
    for name in pools:
        iv = [{'layer': l, 'neuron': j, 'dose': float(dirs[name][l][j] * a.budget)}
              for l in layers for j in torch.nonzero(dirs[name][l].abs() > 1e-15).flatten().tolist()]
        for cx in ctx:
            res[(name, cx['id'])] = run(iv, cx['prompt'])
            prog.step(f'{name} {cx["id"]} p_tox={res[(name, cx["id"])]["tox_bert"]:.3f}')

    cids = [c['id'] for c in ctx]
    an = {}
    for name in pools:
        d = {}
        for key in ('tox_bert', 'tox_snlp', 'own_hits' if False else 'rep4', 'ref_logp'):
            mb = [base[c][key] for c in cids]
            st = [res[(name, c)][key] for c in cids]
            diff = [x - y for x, y in zip(st, mb)]
            d[key] = {'mean_base': statistics.fmean(mb), 'mean_steer': statistics.fmean(st),
                      'mean_delta': statistics.fmean(diff), 'delta_range': [min(diff), max(diff)]}
        d['n_you_directed'] = sum(1 for c in cids if res[(name, c)]['you_directed'])
        d['n_self_directed'] = sum(1 for c in cids if res[(name, c)]['self_directed'])
        d['n_pos'] = sum(1 for c in cids if res[(name, c)]['tox_bert'] >= 0.5)
        an[name] = d

    others = [n for n in pools if n != 'II']
    cr1 = {n: {'diff': statistics.fmean(res[('II', c)]['tox_bert'] - res[(n, c)]['tox_bert']
                                        for c in cids)} for n in others}
    cr1['wins'] = sum(1 for n in others if cr1[n]['diff'] > 0)
    if_p = sign_p([res[('II', c)]['tox_bert'] - res[('IV', c)]['tox_bert'] for c in cids])
    cr2 = {'II_frac_you': an['II']['n_you_directed'] / len(cids),
           'others_mean_frac_you': statistics.fmean(an[n]['n_you_directed'] / len(cids) for n in others)}
    rec = {'schema': 'qwentest-bffm-pools-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'layers': layers,
           'frame': a.frame, 'budget': a.budget, 'contexts': cids, 'trait': a.trait,
           'lang': a.lang, 'census': str(cen), 'pools': pools,
           'p50': {str(l): p50[l] for l in layers},
           'coverage_method': {'II- vs II+': None},
           'analysis': an, 'baseline': base,
           'CR1_II_beats_others': cr1, 'CR1_pass': cr1['wins'] >= 4,
           'CR2_you_directed': cr2, 'CR2_pass': cr2['II_frac_you'] > cr2['others_mean_frac_you'],
           'CR3_IV_within_0.05': abs(cr1['IV']['diff']) <= 0.05,
           'II_vs_IV_sign_p': if_p[0],
           'per_context': {f'{n}|{c}': v for (n, c), v in res.items()}}
    rec.pop('coverage_method')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'bffm_pools.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                        encoding='utf8')
    L = ['# BFFM pool specificity at b=0.7 (the only budget where the gauge has range)', '',
         f'- machine `{rec["machine"]}` / {rec["gpu"]} · layers {layers} · frame {a.frame!r} · '
         f'budget {a.budget} · {len(cids)} {a.trait}/{a.lang} contexts',
         f'- baseline P(toxic): mean {statistics.fmean(base[c]["tox_bert"] for c in cids):.5f}', '',
         '| pool | P(tox) mean | Δ vs base | n>=0.5 | you-directed | self-directed | rep4 | R (ΔlogP) |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in pools:
        d = an[name]
        L.append(f'| `{name}-` vs `{name}+` | {d["tox_bert"]["mean_steer"]:.4f} | '
                 f'{d["tox_bert"]["mean_delta"]:+.4f} | {d["n_pos"]}/{len(cids)} | '
                 f'{d["n_you_directed"]}/{len(cids)} | {d["n_self_directed"]}/{len(cids)} | '
                 f'{d["rep4"]["mean_steer"]:.3f} | {d["ref_logp"]["mean_delta"]:+.4f} |')
    L += ['', '## frozen criteria', '',
          f'- **CR1**: II- beats the other four in **{cr1["wins"]}/4** -> '
          f'**{"PASS" if cr1["wins"] >= 4 else "FAIL"}**',
          '  per-pool Δ(II- minus X): ' + ', '.join(f'{n}={cr1[n]["diff"]:+.3f}' for n in others),
          f'- **CR2**: you-directed fraction II- {cr2["II_frac_you"]:.2f} vs others mean '
          f'{cr2["others_mean_frac_you"]:.2f} -> **{"PASS" if cr2["II_frac_you"] > cr2["others_mean_frac_you"] else "FAIL"}**',
          f'- **CR3**: |II- minus IV-| = {abs(cr1["IV"]["diff"]):.3f} '
          f'{"<= 0.05 -> specificity NOT supported" if abs(cr1["IV"]["diff"]) <= 0.05 else "> 0.05"}',
          f'- II- vs IV- paired exact sign test p = {if_p[0]} (n={if_p[1]})']
    (out / 'bffm_pools.md').write_text('\n'.join(L) + '\n', encoding='utf8')
    print('\n' + '\n'.join(L))
    print(f'\nwrote {out}/bffm_pools.json and .md')


if __name__ == '__main__':
    main()
