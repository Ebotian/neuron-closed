#!/usr/bin/env python3
"""Compare semantic compilers by an INDEPENDENT gauge. Do not pick one; report the Pareto.

STATUS (2026-10-28, end of day): DRY-RUN OK, NEVER EXECUTED. It needs the two gauge models, and the
download of those was interrupted when the remote box was shut down (280 of 438 MB of toxic-bert).
Nothing here is evidence yet. First run tomorrow needs: (1) finish the download and verify sha256,
(2) a 2-context smoke test, (3) the full 8-context run.

Registered rule (record 11 / record 13 section 7): no compiler may be scored by its own objective.
Every compiler here produces a full-dimensional gradient in dose units, the SAME solver turns it
into a direction at the SAME physical budget, and every direction is then scored by gauges that
belong to no compiler.

Compilers (all cheap, all borrowed rather than invented):
  caa_jibes   token-level contrastive difference on our existing 8 contexts:
              grad[ logP(positive continuation) - logP(negative continuation) ], mean over contexts.
              This is textbook CAA (Rimsky et al.) applied to the probes we already have.
  caa_bffm    same mechanism and same frame, but the contrast is Goldberg-1992 BFFM-100
              II- (antagonism: cold, demanding, harsh, rude, selfish) vs II+ (warmth: cooperative,
              generous, helpful, kind, pleasant, sympathetic, warm) - an INDEPENDENT word source
              (CC0, already in the repo). Same frame, different words: that isolates the word source.
  j2          the slot-language log-mass we have used all along (Jibes adj+verb pools, exact VJP),
              which is also route C but at the pool level rather than one string per side.

Gauges (none of them is a compiler):
  E_tox_bert  P(toxic) from unitary/toxic-bert (apache-2.0) on greedy generations, mean over contexts
  E_tox_snlp  P(toxic) from s-nlp/roberta_toxicity_classifier (openrail++, the ParaDetox line CANS
              cites), same texts
  E_own       count of Jibes-derived hostile words in the same generations - OUR instrument, kept only
              as a cross-check of the two borrowed gauges, never as the primary
  R           capability proxy: change in teacher-forced logP of engine.REFERENCE_TEXT
  C           compiler cost = forward/backward passes actually used

FROZEN reading, written before the run:
  * report the full Pareto table (E_tox_bert, E_tox_snlp, R, C) for the three compilers;
  * report pairwise cos(u_m, u_n) and top-64 overlap of the dose rankings, so "the compilers disagree"
    and "the effect disagrees" are separable;
  * report the agreement between the two borrowed gauges (Pearson on P(toxic), and Cohen kappa at
    0.5) and between each of them and E_own;
  * DO NOT select a compiler. Selection is a later, evidence-based step.
  PREDICTIONS TO FALSIFY (registered here, before running):
    P1  cos(u_caa_jibes, u_j2) < 0.8
    P2  the two borrowed gauges agree better with each other than either agrees with E_own
    P3  E_tox_bert(u_j2) > E_tox_bert(u_caa_jibes)
  SECOND-ROUND PREDICTIONS (this run: --trait hostility --lang en, because stage 0 of plan 2.28.6
  shows both classifiers are blind on Chinese and near-blind on the verb frame, while on English
  probe continuations they separate hostile from neutral at AUC 0.986-1.000):
    Q1  at b=0.05 the community control moves E_tox but no compiler does
        (i.e. the readout's own direction stays behaviourally invisible at the certified budget)
    Q2  at b=0.7 the j2 direction DOES move E_tox, i.e. the effect becomes visible only when the
        budget leaves the certified band
    Q3  E_tox ordering at b=0.7 satisfies control > j2 (the community recipe is still the strongest
        single mover, consistent with plan 2.21's finding that L14N8949 is not the readout's top-1)
  Error discipline (plan 2.18.4): a single dJ2-scale difference below 1e-02 is not claimed; the
  gauges here are probabilities on 8 contexts, so their spread is reported as a paired range.

    [i/N] elapsed, eta ...   per measurement (package progress.Progress)
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # qwentest/


def resolve_pkg(given):
    cands = [Path(given)] if given else []
    cands += [ROOT / '03_unpacked/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
              Path('/root/autodl-tmp/scripts/j2space'),
              Path('/root/autodl-tmp/qwen-gradient-discovery-v6')]
    for c in cands:
        if (c / 'engine.py').is_file() and (c / 'semantic.py').is_file():
            return c
    raise SystemExit('package not found; pass --pkg <dir with engine.py AND semantic.py>')


def load_pools(path, variant='full'):
    lex = json.loads(Path(path).read_text(encoding='utf8'))
    fams = {}
    for fam, d in lex['families'].items():
        key = 'pos' if variant == 'full' else f'pos_{variant}'
        tmpl = d.get('template') or lex.get('templates', {}).get(fam)
        if not tmpl:
            raise ValueError(f'family {fam} has no template')
        fams[fam] = {'pos': d[key], 'neg': d['neg'], 'template': tmpl}
    return lex, fams


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--bffm', default=str(ROOT / 'lexicons/bffm100_factors.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--gauges', default='/root/autodl-tmp/gauges',
                    help='dir with unitary_toxic-bert/ and s-nlp_roberta_toxicity_classifier/')
    ap.add_argument('--layers', default='13-14')
    ap.add_argument('--budgets', default='0.05',
                    help='comma list of physical budgets; 0.05 is our certified clean区, 0.7 is the '
                         'community operating point')
    ap.add_argument('--trait', default='hostility')
    ap.add_argument('--lang', default='en', choices=['en', 'zh', 'all'],
                    help='gauge_validity (plan 2.28.6 stage 0) shows both classifiers are blind on '
                         'Chinese and near-blind on the verb frame, so the default is English only')
    ap.add_argument('--contexts', type=int, default=8)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--max-new', type=int, default=96)
    ap.add_argument('--out', default='out/compiler_pareto')
    ap.add_argument('--control-known', action='store_true',
                    help='add a POSITIVE CONTROL arm: the community recipe L14N8949+0.7 / L13N7956-0.7, '
                         'which two independent reports show produces hostile text. Without it a flat '
                         'gauge cannot be distinguished from a broken gauge.')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    bffm = json.loads(Path(a.bffm).read_text(encoding='utf8'))
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    def _zh(s):
        return any('\u4e00' <= ch <= '\u9fff' for ch in s)
    sel = [r for r in rows if r.get('trait') == a.trait]
    if a.lang != 'all':
        sel = [r for r in sel if (_zh(r['prompt']) == (a.lang == 'zh'))]
    ctx = sel[:a.contexts]
    budgets = [float(x) for x in a.budgets.split(',')]
    if not ctx:
        raise SystemExit(f'no {a.split} contexts in {a.probes}')
    # BFFM: only single-token survivors are usable (plan 2.19 census: II+ 7, II- 5)
    # the single-token census (plan 2.19) may sit next to this script or under the repo root
    cands = [Path(__file__).resolve().parent / '07_results/slot_census_bffm100.json',
             ROOT / '07_results/slot_census_bffm100.json']
    census = next((c for c in cands if c.is_file()), None)
    single = set()
    if census is not None:
        cs = json.loads(census.read_text(encoding='utf8'))
        single = {t for pool in cs['pools'].values() for t in pool['survivors']}
        print(f'census : {census} -> {len(single)} single-token survivors known', flush=True)
    else:
        print('census : NOT FOUND - the BFFM words are NOT filtered to single tokens', flush=True)
    ant = [w for w in bffm['pools']['II-'] if not single or w in single]
    war = [w for w in bffm['pools']['II+'] if not single or w in single]
    if not ant or not war:
        raise SystemExit('BFFM pools are empty after the single-token filter')
    frame = fams[sorted(fams)[0]]['template']      # reuse the Jibes adj frame; only the WORDS differ

    compilers = ['caa_jibes', 'caa_bffm', 'j2']
    n_grad = len(ctx) * (2 + len(ant) + len(war) + len(fams))   # 2 CAA + BFFM words + one per family
    n_pass = n_grad
    print('Question : which semantic compiler survives an INDEPENDENT gauge? (compare, do not choose)')
    print(f'Compilers: {compilers}')
    print(f'  caa_bffm contrast: antagonism {len(ant)} words {ant}')
    print(f'                     vs warmth    {len(war)} words {war}')
    print(f'                     inside frame {frame!r} (same frame as caa_jibes; only the words differ)')
    print(f'Solver   : one shared closed form, budget(s) {budgets} via M = What What\'')
    print(f'Filter   : trait {a.trait}, language {a.lang} -> {len(ctx)} contexts')
    print(f'Gauges   : toxic-bert + s-nlp roberta on greedy <= {a.max_new} tokens; R = REFERENCE_TEXT logP;'
          f' C = passes')
    print(f'Contexts : {len(ctx)} -> {[c["id"] for c in ctx]}')
    n_arms = len(compilers) + (1 if a.control_known else 0)
    print(f'Work     : {n_grad} forward/backward passes for gradients + '
          f'{n_arms * len(ctx) * (len(budgets) * 2 + 0)} (generation+capability) pairs + {len(ctx)} baselines')
    print('Frozen   : report the Pareto; do not select. Predictions P1/P2/P3 are registered in the '
          'docstring and will be reported as HELD or FALSIFIED.')
    if a.dry_run:
        print(f'[dry-run] would write {a.out}/compiler_pareto.json and .md')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import (REFERENCE_TEXT, apply_steering, candidate_ids, capture_mlp_inputs,
                        column_norms, continuation_logp, estimate_p50, model_lm, prompt_prefix,
                        score_for_dose)
    from progress import Progress
    from semantic import batch_mean_logp, build_batch, language_log_mass, sem_gradient

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

    prog = Progress(n_grad + len(compilers) * len(ctx) * 2)

    def grad_of_string(prefix_ids, completion):
        """d/dh of logP(completion | prefix), summed over positions, in dose units."""
        ids = torch.tensor([prefix_ids + tok(completion, add_special_tokens=False).input_ids],
                           dtype=torch.long, device=dev)
        pre = len(prefix_ids)
        suf = ids.shape[1] - pre
        with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
            out = model(input_ids=ids, use_cache=False)
            logits = out.logits[:, pre - 1:pre - 1 + suf]
            target = ids[:, pre:pre + suf]
            import torch.nn.functional as F
            lp = -F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                  target.reshape(-1), reduction='mean')
            grads = torch.autograd.grad(lp, [cap[l] for l in layers])
        return {l: g.detach().float().sum(dim=tuple(range(g.ndim - 1))) for l, g in zip(layers, grads)}

    def to_dose(raw):
        return {l: score_for_dose(raw[l], p50[l], norms[l].to(raw[l].device)).double().cpu()
                for l in layers}

    G = {c: {l: torch.zeros(npl, dtype=torch.float64) for l in layers} for c in compilers}
    for c in ctx:
        cid = c['id']
        pid = prompt_prefix(tok, c['prompt'])
        gp = to_dose(grad_of_string(pid, c['positive']))
        gn = to_dose(grad_of_string(pid, c['negative']))
        for l in layers:
            G['caa_jibes'][l] += (gp[l] - gn[l]) / len(ctx)
        # same mechanism, BFFM words inside the same frame
        bf = {l: torch.zeros(npl, dtype=torch.float64) for l in layers}
        for words, sgn in ((ant, +1.0), (war, -1.0)):
            acc = {l: torch.zeros(npl, dtype=torch.float64) for l in layers}
            for w in words:
                g = to_dose(grad_of_string(pid, frame.format(w=w)))
                for l in layers:
                    acc[l] += g[l]
            for l in layers:
                bf[l] += sgn * acc[l] / len(words)
        for l in layers:
            G['caa_bffm'][l] += bf[l] / len(ctx)
        # j2: existing slot-language log-mass, both families averaged
        acc = None
        for fam, f in sorted(fams.items()):
            _j2, gg = sem_gradient(tok, model, lm, layers, c['prompt'], f['template'], f, dev, a.max_seq)
            d = to_dose(gg)
            acc = d if acc is None else {l: acc[l] + d[l] for l in layers}
        for l in layers:
            G['j2'][l] += acc[l] / (len(fams) * len(ctx))
        prog.step(f'{cid} gradients for {len(compilers)} compilers')

    # shared solver: u = What B^-1 d*,  d* = B^-1 e,  e = What' g ; then normalise to the M budget
    dirs = {}
    for c in compilers:
        g = G[c]
        e = {l: What[l].T.double().to(dev) @ g[l].to(dev) for l in layers}
        d_star = {l: torch.linalg.solve(B[l], e[l]) for l in layers}
        u = {l: (What[l].double() @ torch.linalg.solve(B[l], d_star[l])) for l in layers}
        bu = math.sqrt(sum(float((u[l].float() @ What[l]).pow(2).sum()) for l in layers))
        assert bu > 0
        dirs[c] = {l: u[l] / bu for l in layers}      # unit M-norm; scaled per budget below

    def cos(d1, d2):
        x = torch.cat([d1[l].flatten() for l in layers]).double()
        y = torch.cat([d2[l].flatten() for l in layers]).double()
        return float(x @ y / (x.norm() * y.norm()))

    def iv_of(d):
        out = []
        for l in layers:
            for j in torch.nonzero(d[l].abs() > 1e-15).flatten().tolist():
                out.append({'layer': l, 'neuron': int(j), 'dose': float(d[l][j])})
        return out

    # ---------------- gauges ----------------
    from transformers import AutoModelForSequenceClassification
    gauges = {}
    for name, sub in (('tox_bert', 'unitary_toxic-bert'),
                      ('tox_snlp', 's-nlp_roberta_toxicity_classifier')):
        d = Path(a.gauges) / sub
        if not d.is_dir():
            raise SystemExit(f'gauge model missing: {d}')
        m = AutoModelForSequenceClassification.from_pretrained(d, local_files_only=True).to(dev).eval()
        tk = AutoTokenizer.from_pretrained(d, local_files_only=True)
        gauges[name] = (tk, m)

    # label maps, read from each model's config.json:
    #   unitary/toxic-bert         6 outputs, MULTI-label (sigmoid); index 0 = 'toxic'
    #   s-nlp/roberta_toxicity_classifier  2 outputs (softmax); index 1 = 'toxic'
    # Reading toxic-bert's softmax[-1] would have reported 'identity_hate', not 'toxic'.
    def p_toxic(text):
        out = {}
        for name, (tk, m) in gauges.items():
            with torch.inference_mode():
                enc = tk(text or ' ', return_tensors='pt', truncation=True, max_length=512).to(dev)
                logits = m(**enc).logits.float()[0]
                if logits.numel() == 6:
                    out[name] = float(torch.sigmoid(logits[0]))          # toxic-bert, label 'toxic'
                elif logits.numel() == 2:
                    out[name] = float(torch.softmax(logits, -1)[1])      # s-nlp, label 'toxic'
                else:
                    raise SystemExit(f'{name}: unexpected {logits.numel()} outputs')
        return out

    own = [w.lower() for f in fams.values() for w in f['pos']]

    arms = list(compilers)
    if a.control_known:
        arms.append('known_recipe')
        dirs['known_recipe'] = None          # handled as a literal dose list, not a solved direction

    def iv_at(c, budget):
        if c == 'known_recipe':
            # the community recipe is a literal dose list, not a budget-normalised direction
            return [{'layer': 14, 'neuron': 8949, 'dose': 0.7},
                    {'layer': 13, 'neuron': 7956, 'dose': -0.7}]
        return iv_of({l: dirs[c][l] * budget for l in layers})

    # baseline first (independent of arm and budget), then every (arm, budget) steering pass
    base = {}
    prog_total = len(ctx) + len(arms) * len(budgets) * len(ctx)
    for cx in ctx:
        cid = cx['id']
        rid, rpre, rsuf = candidate_ids(tok, cx['prompt'], REFERENCE_TEXT, dev, a.max_seq)
        with torch.inference_mode():
            lp = float(continuation_logp(model, rid, rpre, rsuf))
            ids = torch.tensor([prompt_prefix(tok, cx['prompt'])], dtype=torch.long, device=dev)
            out = model.generate(input_ids=ids, do_sample=False, max_new_tokens=a.max_new,
                                 use_cache=True,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None
                                 else tok.eos_token_id)
            text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        p = p_toxic(text)
        base[cid] = {'ref_logp': lp, 'text_first_240': text[:240], **p,
                     'own_hits': sum(text.lower().count(w) for w in own)}
        prog.step(f'base {cid} p_tox={p["tox_bert"]:.4f}/{p["tox_snlp"]:.4f} ref={lp:+.4f}')

    res = {}
    for c in arms:
        for b in budgets:
            if c == 'known_recipe' and b != budgets[0]:
                # the community recipe is a fixed dose list; measuring it once is enough
                for cx in ctx:
                    res[(c, b, cx['id'])] = dict(res[(c, budgets[0], cx['id'])])
                continue
            iv = iv_at(c, b)
            for cx in ctx:
                cid = cx['id']
                rid, rpre, rsuf = candidate_ids(tok, cx['prompt'], REFERENCE_TEXT, dev, a.max_seq)
                with apply_steering(lm, iv, p50, norms), torch.inference_mode():
                    lp = float(continuation_logp(model, rid, rpre, rsuf))
                    ids = torch.tensor([prompt_prefix(tok, cx['prompt'])], dtype=torch.long, device=dev)
                    out = model.generate(input_ids=ids, do_sample=False, max_new_tokens=a.max_new,
                                         use_cache=True,
                                         pad_token_id=tok.pad_token_id if tok.pad_token_id is not None
                                         else tok.eos_token_id)
                    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                p = p_toxic(text)
                res[(c, b, cid)] = {'ref_logp': lp, 'text_first_240': text[:240], **p,
                                    'own_hits': sum(text.lower().count(w) for w in own)}
                prog.step(f'{c} b={b} {cid} p_tox={p["tox_bert"]:.4f}/{p["tox_snlp"]:.4f} '
                          f'ref={lp:+.4f}')

    # ---------------- frozen analysis ----------------
    cids = [c['id'] for c in ctx]
    an = {}
    for c in arms:
        for b in budgets:
            d = {}
            for key in ('tox_bert', 'tox_snlp', 'own_hits', 'ref_logp'):
                mb = [base[cid][key] for cid in cids]
                ste = [res[(c, b, cid)][key] for cid in cids]
                diffs = [s - x for s, x in zip(ste, mb)]
                d[key] = {'mean_base': statistics.fmean(mb), 'mean_steer': statistics.fmean(ste),
                          'mean_delta': statistics.fmean(diffs),
                          'delta_range': [min(diffs), max(diffs)],
                          'n_positive': sum(1 for x in diffs if x > 0), 'n': len(diffs)}
            an[f'{c}|b={b}'] = d
    pairs = {}
    solved = [c for c in arms if dirs.get(c) is not None]
    for i, c1 in enumerate(solved):
        for c2 in solved[i + 1:]:
            flat1 = torch.cat([dirs[c1][l].flatten() for l in layers]).double()
            flat2 = torch.cat([dirs[c2][l].flatten() for l in layers]).double()
            t1 = set(torch.topk(flat1.abs(), 64).indices.tolist())
            t2 = set(torch.topk(flat2.abs(), 64).indices.tolist())
            pairs[f'{c1}|{c2}'] = {'cos': cos(dirs[c1], dirs[c2]),
                                   'top64_overlap': len(t1 & t2) / 64}
    # gauge agreement on the same texts
    allrows = list(base.values()) + [v for v in res.values()]
    g1 = [v['tox_bert'] for v in allrows]
    g2 = [v['tox_snlp'] for v in allrows]
    g3 = [float(v['own_hits']) for v in allrows]
    def kappa(x, y, thr=0.5):
        a = [1 if v >= thr else 0 for v in x]
        b = [1 if v >= thr else 0 for v in y]
        n = len(a); po = sum(u == v for u, v in zip(a, b)) / n
        pa = sum(a) / n; pb = sum(b) / n
        pe = pa * pb + (1 - pa) * (1 - pb)
        return {'agreement': po, 'kappa': (po - pe) / (1 - pe) if pe < 1 else float('nan'), 'n': n}
    agree = {'pearson_bert_snlp': statistics.correlation(g1, g2) if len(set(g1)) > 1 else None,
             'kappa_bert_snlp': kappa(g1, g2),
             'pearson_bert_own': statistics.correlation(g1, g3) if len(set(g3)) > 1 else None,
             'pearson_snlp_own': statistics.correlation(g2, g3) if len(set(g3)) > 1 else None}

    rec = {'schema': 'qwentest-compiler-pareto-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'layers': layers,
           'budgets': budgets, 'contexts': cids, 'trait': a.trait, 'lang': a.lang,
           'compilers': compilers, 'arms': arms,
           'bffm_antagonism_words': ant, 'bffm_warmth_words': war, 'bffm_frame': frame,
           'p50': {str(l): p50[l] for l in layers},
           'lexicon_sha256': hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest(),
           'gauge_label_maps': {'tox_bert': 'multi-label sigmoid, index 0 = toxic',
                                'tox_snlp': 'binary softmax, index 1 = toxic'},
           'predictions': {'P1': 'cos(caa_jibes, j2) < 0.8',
                           'P2': 'the two borrowed gauges agree better with each other than with own',
                           'P3': 'E_tox_bert(j2) > E_tox_bert(caa_jibes)'},
           'analysis': an, 'direction_pairs': pairs, 'gauge_agreement': agree,
           'baseline': base,
           'per_context': {f'{c}|{b}|{cid}': v for (c, b, cid), v in res.items()}}
    if a.control_known:
        rec['positive_control'] = {
            'arm': 'known_recipe L14N8949+0.7 L13N7956-0.7',
            'reading': 'if this arm does not move the gauges, the gauge/prompt pair - not the compiler - is '
                       'the problem; if it does, the compilers are simply too weak at this budget'}
    b0 = budgets[0]
    rec['prediction_outcome'] = {
        'P1': pairs.get('caa_jibes|j2', {}).get('cos') is not None and
              pairs['caa_jibes|j2']['cos'] < 0.8,
        'P2': (agree['pearson_bert_snlp'] is not None and agree['pearson_bert_own'] is not None
               and agree['pearson_snlp_own'] is not None and
               agree['pearson_bert_snlp'] > max(agree['pearson_bert_own'], agree['pearson_snlp_own'])),
        'P3': an[f'j2|b={b0}']['tox_bert']['mean_steer'] > an[f'caa_jibes|b={b0}']['tox_bert']['mean_steer']}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'compiler_pareto.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                              encoding='utf8')
    L = ['# Semantic compilers under an independent gauge (compare, do not choose)', '',
         f'- machine `{rec["machine"]}` / {rec["gpu"]} · layers {layers} · budgets {budgets} · '
         f'{len(cids)} contexts · greedy <= {a.max_new} tokens',
         f'- compilers: {compilers}',
         f'- caa_bffm words: antagonism {len(ant)}, warmth {len(war)}, frame {frame!r}', '',
         '## Pareto table (all deltas are steer minus base, mean over contexts)', '',
         '| arm | budget | E_tox_bert | E_tox_snlp | E_own | R (dlogP ref) |',
         '|---|---|---:|---:|---:|---:|']
    L = L[:-1] if False else L
    for k_, d in an.items():
        L.append(f'| `{k_}` | {d["tox_bert"]["mean_delta"]:+.4f} | {d["tox_snlp"]["mean_delta"]:+.4f} | '
                 f'{d["own_hits"]["mean_delta"]:+.3f} | {d["ref_logp"]["mean_delta"]:+.5f} |')
    L += ['', '## direction agreement (does the compiler change the vector?)', '',
          '| pair | cos | top-64 overlap |', '|---|---:|---:|']
    for k, v in pairs.items():
        L.append(f'| `{k}` | {v["cos"]:.4f} | {v["top64_overlap"]:.3f} |')
    L += ['', '## gauge agreement on the same texts', '',
          f'- Pearson(tox_bert, tox_snlp) = {agree["pearson_bert_snlp"]}',
          f'- kappa(tox_bert, tox_snlp) at 0.5 = {agree["kappa_bert_snlp"]["kappa"]:.4f} '
          f'(agreement {agree["kappa_bert_snlp"]["agreement"]:.3f}, n={agree["kappa_bert_snlp"]["n"]})',
          f'- Pearson(tox_bert, own) = {agree["pearson_bert_own"]}',
          f'- Pearson(tox_snlp, own) = {agree["pearson_snlp_own"]}', '',
          '## registered predictions', '']
    for k, v in rec['prediction_outcome'].items():
        L.append(f'- {k}: **{"HELD" if v else "FALSIFIED"}**  ({rec["predictions"][k]})')
    (out / 'compiler_pareto.md').write_text('\n'.join(L) + '\n', encoding='utf8')
    print('\n' + '\n'.join(L))
    print(f'\nwrote {out}/compiler_pareto.json and .md')


if __name__ == '__main__':
    main()
