#!/usr/bin/env python3
"""Is the R7 gain real, or is it degeneration? Judge-free behavioural check.

Everything in plan 2.20-2.24 is a READOUT measurement: dJ2, a logit-space log-mass over
hostile vs neutral slot words. At b=0.5 the dense arm reaches dJ2 = +1.88 on a baseline of
about -0.2 to -0.45, which is far beyond any dose this repo used before. The external
Qwen3-8B atlas (08_reference §4) measured what happens in that regime: 85.60% of its
"persona" replies carried the `repetition` tag, 70.20% scored coherence <= 3 from its own
judge, and only 8.52% ended naturally. So the live risk is that u_M simply breaks the model
more than u_g does, and a broken model puts more mass on hostile slot words too.

This measures that WITHOUT an LLM judge, because 08_reference §4.4 showed one judge can
return 4.30% / 7.74% / 48.16% for "degeneration" depending on whether you read its category
or its tags. Objective, reproducible quantities only:

  generation (greedy, <= max_new): stop reason (eos / max), tokens produced,
      distinct-1, distinct-2, rep4 = 1 - distinct_4grams/total_4grams,
      longest immediately-repeated span in tokens
  capability proxy: change in teacher-forced mean logP of engine.REFERENCE_TEXT
      (the frozen neutral prose+code+opinion text already used for p50)

Arms: none | u_g (~g, full support) | u_M (~M^+g, full support), built per context exactly
as in plan 2.22, at budgets 0.05 and 0.5 (the two ends of the range already measured).

FROZEN criteria (written before the run)
  PRIMARY  u_M is NOT worse than u_g, paired over the 8 contexts at each budget:
             rep4(u_M) <= rep4(u_g)              in >= 7/8 contexts, AND
             dlogP_ref(u_M) >= dlogP_ref(u_g)    in >= 7/8 contexts
           (worse = more repetition / more negative capability proxy). Exact one-sided sign
           test reported. If either fails, the dJ2 gain is contaminated and R7's practical
           value is NOT established.
  CONTEXT  absolute levels are reported for the unsteered baseline too, so "both arms are
           degenerate at b=0.5" is a possible and publishable outcome.

Steering: full support means 18432 coordinates, and the frozen `engine.apply_steering` hook
loops over them in Python at EVERY decode step, which is unusable for generation. This script
therefore uses a vectorised hook (one scatter per layer) -- and VALIDATES it against the
frozen implementation first, requiring bitwise equality on a 32-coordinate case before any
measurement is taken.

    [i/N] elapsed, eta ...   per measurement (package progress.Progress)
"""
from __future__ import annotations
import argparse, contextlib, hashlib, json, math, platform, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # qwentest/


def resolve_pkg(given):
    cands = [Path(given)] if given else []
    cands += [ROOT / 'vendor/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
              Path('/root/autodl-tmp/scripts/j2space'),
              Path('/root/autodl-tmp/qwen-gradient-discovery-v6')]
    for c in cands:
        if (c / 'engine.py').is_file() and (c / 'semantic.py').is_file():
            return c
    raise SystemExit('package not found; pass --pkg <dir with engine.py AND semantic.py>')


def load_pools(path, variant='full'):
    """Same rule as scan_sem.py::load_pools."""
    lex = json.loads(Path(path).read_text(encoding='utf8'))
    fams = {}
    for fam, d in lex['families'].items():
        key = 'pos' if variant == 'full' else f'pos_{variant}'
        tmpl = d.get('template') or lex.get('templates', {}).get(fam)
        if not tmpl:
            raise ValueError(f'family {fam} has no template')
        fams[fam] = {'pos': d[key], 'neg': d['neg'], 'template': tmpl}
    return lex, fams


def exact_sign_p(diffs, one_sided_positive=True):
    """Exact binomial sign test. Two-sided by default; zeros excluded."""
    v = [d for d in diffs if abs(d) > 1e-12]
    n = len(v)
    if n == 0:
        return None, 0
    k = sum(x > 0 for x in v)
    if one_sided_positive:
        # P(X >= k) under p=0.5, doubled only when it exceeds 0.5 (kept two-sided-safe)
        tail = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
        return min(1.0, 2 * min(tail, 1 - tail + math.comb(n, k) / 2 ** n)), n
    k2 = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k2 + 1)) / 2 ** n), n


@contextlib.contextmanager
def fast_steering(lm, iv_by_layer, p50, norms, torch):
    """Vectorised equivalent of engine.apply_steering.

    iv_by_layer: {layer: (LongTensor idx, FloatTensor dose)}. One scatter per layer per
    forward instead of a Python loop over every coordinate -- required for generation, where
    the hook fires at every decode step.
    """
    handles = []
    try:
        for l, (idx, dose) in iv_by_layer.items():
            if idx.numel() == 0:
                continue
            # Match engine.apply_steering BITWISE. The frozen hook does
            #     alpha = dose * p50[l] / max(float(norms[l][n]), 1e-12)   # Python float64
            #     act[..., n] += alpha                                     # scalar add
            # and for a bf16 activation PyTorch runs that scalar add in opmath float32:
            # the Python float becomes float32, NOT bf16, and the sum is rounded to bf16 once.
            # Measured on the GPU (46,080 elements): pre-rounding alpha to bf16 and using
            # `act[..., idx] += alpha_bf16` differs from the frozen path in 4 elements,
            # max 1.953e-03 -- which propagated to 3.922e-02 of drift in the readout.
            # Gathering to float32, adding float32 alpha and rounding back once is
            # bitwise identical (0 / 46,080 differing), so that is what this hook does.
            alpha = ((dose.double() * p50[l])
                     / norms[l].to(dose.device).double()[idx].clamp(min=1e-12)).to(torch.float32)

            def make(layer_id, ix, al):
                def hook(_module, inp):
                    act = inp[0].clone()
                    act[..., ix] = (act[..., ix].float() + al).to(act.dtype)
                    return (act,) + inp[1:]
                return hook
            handles.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(make(l, idx, alpha)))
        yield
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--layers', default='13-14')
    ap.add_argument('--budgets', default='0.05,0.5')
    ap.add_argument('--contexts', type=int, default=8)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--max-new', type=int, default=96)
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--out', default='out/behavior_check')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
    budgets = [float(x) for x in a.budgets.split(',')]
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    ctx = [r for r in rows if r.get('split') == a.split][:a.contexts]
    if not ctx:
        raise SystemExit(f'no {a.split} contexts in {a.probes}')
    arms = ['none', 'u_g', 'u_M']
    # 'none' does not depend on the budget, so it is measured once per context; the ETA is
    # wrong if the total counts it per budget.
    per_ctx = len(arms) + (len(arms) - 1) * (len(budgets) - 1)
    n_meas = len(ctx) * per_ctx

    print('Question : is the R7 dJ2 gain real, or is it degeneration? (judge-free)')
    print(f'Arms     : {arms}  (u_g ~ g, u_M ~ M^+g, both full support, built per context as in §2.22)')
    print(f'Budgets  : {budgets}   contexts {len(ctx)}   max_new_tokens {a.max_new}')
    print(f'Metrics  : stop reason, tokens, distinct-1/2, rep4, longest immediate repeat;')
    print(f'           capability proxy = change in teacher-forced mean logP of engine.REFERENCE_TEXT')
    print(f'Work     : {n_meas} generations + {n_meas} capability forwards + grad passes')
    print('Frozen   : PRIMARY = rep4(u_M) <= rep4(u_g) in >=7/8 AND dlogP_ref(u_M) >= dlogP_ref(u_g)')
    print('           in >=7/8, per budget, paired over contexts. Absolute levels reported for the')
    print('           unsteered baseline too, so "both arms degenerate" is an allowed outcome.')
    if a.dry_run:
        print(f'[dry-run] would write {a.out}/behavior_check.json and .md')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import (REFERENCE_TEXT, apply_steering, candidate_ids, column_norms,
                        continuation_logp, estimate_p50, model_lm, prompt_prefix, score_for_dose)
    from progress import Progress
    from semantic import batch_mean_logp, build_batch, language_log_mass, sem_gradient

    dev = 'cuda:0'
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to(dev).eval()
    model.requires_grad_(False)
    lm = model_lm(model)
    if len(lm.layers) != 32:
        raise RuntimeError(f'expected 32 layers, got {len(lm.layers)}')
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device=dev)
    print('p50:', {l: round(p50[l], 6) for l in layers})

    What, B, n_dim = {}, {}, None
    for l in layers:
        Wt = lm.layers[l].mlp.down_proj.weight.detach().double().T
        Wt = Wt / Wt.norm(dim=1, keepdim=True).clamp(min=1e-12)
        What[l] = Wt
        B[l] = Wt.T @ Wt
        n_dim = Wt.shape[0]

    def budget_of(u_by_layer):
        return math.sqrt(sum(float((u @ What[l]).pow(2).sum()) for l, u in u_by_layer.items()))

    # ---- validate the vectorised hook against the frozen one, bitwise ----
    probe_prompt = ctx[0]['prompt']
    test_iv = [{'layer': layers[0], 'neuron': i, 'dose': 0.01 * (i % 7 - 3)}
               for i in (8949, 7956, 1, 2, 3, 100, 5000)] + \
              [{'layer': layers[-1], 'neuron': i, 'dose': -0.02} for i in (11, 222, 3333)]
    fam0 = sorted(fams)[0]
    ids0, pre0, suf0, kinds0 = build_batch(tok, probe_prompt, fams[fam0]['template'],
                                           fams[fam0]['pos'], fams[fam0]['neg'], dev, a.max_seq)
    with apply_steering(lm, test_iv, p50, norms), torch.inference_mode():
        S_ref = batch_mean_logp(model, ids0, pre0, suf0).float().cpu()
    by_layer = {}
    for it in test_iv:
        by_layer.setdefault(int(it['layer']), []).append(it)
    ivl = {l: (torch.tensor([x['neuron'] for x in v], device=dev),
               torch.tensor([x['dose'] for x in v], device=dev, dtype=torch.float64))
           for l, v in by_layer.items()}
    with fast_steering(lm, ivl, p50, norms, torch), torch.inference_mode():
        S_fast = batch_mean_logp(model, ids0, pre0, suf0).float().cpu()
    maxdiff = float((S_ref - S_fast).abs().max())
    print(f'vectorised hook vs frozen apply_steering: max |dS| = {maxdiff:.3e} '
          f'({"BITWISE EQUAL" if maxdiff == 0.0 else "DIFFERS"})')
    if maxdiff != 0.0:
        raise SystemExit('vectorised steering does not reproduce the frozen implementation '
                         'bitwise; refusing to measure with it')

    def to_iv(u_by_layer, scale):
        out = {}
        for l, u in u_by_layer.items():
            v = u * scale
            nz = torch.nonzero(v.abs() > 1e-15).flatten()
            out[l] = (nz.to(dev), v[nz].to(dev))
        return out

    def gen_metrics(text, n_tokens, eos_hit):
        t = tok(text, add_special_tokens=False).input_ids
        n = len(t)
        d1 = len(set(t)) / n if n else 0.0
        bg = [tuple(t[i:i + 2]) for i in range(n - 1)]
        qg = [tuple(t[i:i + 4]) for i in range(n - 3)]
        rep4 = 1 - (len(set(qg)) / len(qg)) if qg else 0.0
        longest = 0
        for L in range(8, n // 2 + 1):
            found = False
            for i in range(0, n - 2 * L + 1):
                if t[i:i + L] == t[i + L:i + 2 * L]:
                    found = True
                    break
            if found:
                longest = L
            else:
                break
        return {'stop': 'eos' if eos_hit else 'max', 'n_tokens': int(n_tokens),
                'n_decoded': n, 'distinct1': d1,
                'distinct2': len(set(bg)) / len(bg) if bg else 0.0,
                'rep4': rep4, 'longest_immediate_repeat': longest,
                'text_first_240': text[:240]}

    prog = Progress(len(ctx) * len(fams) + n_meas)
    rec = {'schema': 'qwentest-behavior-check-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'layers': layers,
           'budgets': budgets, 'arms': arms, 'max_new': a.max_new,
           'pool_variant': a.pool_variant, 'p50': {str(l): p50[l] for l in layers},
           'lexicon_sha256': hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest(),
           'hook_equivalence_max_abs_diff': maxdiff,
           'reference_text_sha256': hashlib.sha256(REFERENCE_TEXT.encode()).hexdigest(),
           'contexts': [c['id'] for c in ctx],
           'frozen_criteria': {
               'primary': 'rep4(u_M) <= rep4(u_g) in >=7/8 AND dlogP_ref(u_M) >= dlogP_ref(u_g) '
                          'in >=7/8, per budget, paired over contexts',
               'allowed_outcome': 'both arms degenerate at a budget is a publishable result'},
           'per_context': {}}

    ref_ids = {}
    for c in ctx:
        ref_ids[c['id']] = candidate_ids(tok, c['prompt'], REFERENCE_TEXT, dev, a.max_seq)

    for c in ctx:
        cid = c['id']
        gsum = None
        for fam, f in sorted(fams.items()):
            _j2, gg = sem_gradient(tok, model, lm, layers, c['prompt'], f['template'], f, dev,
                                   a.max_seq)
            sc = {l: score_for_dose(gg[l].float(), p50[l], norms[l].to(gg[l].device)) for l in layers}
            gsum = sc if gsum is None else {l: gsum[l] + sc[l] for l in layers}
            prog.step(f'{cid} g {fam}')
        g = {l: (gsum[l] / len(fams)).double().to(dev) for l in layers}
        e = {l: What[l].T @ g[l] for l in layers}
        ds = {l: torch.linalg.solve(B[l], e[l]) for l in layers}
        dirs = {'none': None, 'u_g': g,
                'u_M': {l: What[l] @ torch.linalg.solve(B[l], ds[l]) for l in layers}}

        rid, rpre, rsuf = ref_ids[cid]
        entry = {}
        for b in budgets:
            for arm in arms:
                if arm == 'none' and b != budgets[0]:
                    continue                      # baseline does not depend on the budget
                v = dirs[arm]
                if v is None:
                    ivl_arm = {l: (torch.zeros(0, dtype=torch.long, device=dev),
                                   torch.zeros(0, dtype=torch.float64, device=dev))
                               for l in layers}
                    iv_frozen = []
                else:
                    bu = budget_of(v)
                    if bu < 1e-12:
                        raise SystemExit(f'degenerate direction for {arm}')
                    scale = b / bu
                    ivl_arm = to_iv(v, scale)
                    iv_frozen = None
                with torch.inference_mode():
                    with fast_steering(lm, ivl_arm, p50, norms, torch):
                        lp = float(continuation_logp(model, rid, rpre, rsuf))
                        prefix = prompt_prefix(tok, c['prompt'])
                        ids = torch.tensor([prefix], dtype=torch.long, device=dev)
                        out = model.generate(input_ids=ids, do_sample=False,
                                             max_new_tokens=a.max_new, use_cache=True,
                                             pad_token_id=tok.pad_token_id
                                             if tok.pad_token_id is not None else tok.eos_token_id)
                new = out[0, ids.shape[1]:]
                eos = int(tok.eos_token_id) if tok.eos_token_id is not None else -1
                eos_hit = bool((new == eos).any())
                text = tok.decode(new, skip_special_tokens=True)
                m = gen_metrics(text, int(new.numel()), eos_hit)
                m['ref_logp'] = lp
                m['budget'] = b
                m['n_nonzero'] = int(sum(ix.numel() for ix, _ in ivl_arm.values()))
                entry[f'{arm}|{b}'] = m
                prog.step(f'{cid} b={b} {arm} stop={m["stop"]} rep4={m["rep4"]:.3f} '
                          f'reflogp={lp:+.4f}')
        base_lp = entry[f'none|{budgets[0]}']['ref_logp']
        for k, m in entry.items():
            m['dlogp_ref'] = m['ref_logp'] - base_lp
        rec['per_context'][cid] = entry

    # ---------------- frozen analysis ----------------
    cids = rec['contexts']
    an = {}
    for b in budgets:
        for metric, better_when_lower in (('rep4', True), ('longest_immediate_repeat', True),
                                          ('distinct2', False), ('dlogp_ref', False)):
            diffs = []
            for cid in cids:
                x = rec['per_context'][cid][f'u_M|{b}'][metric]
                y = rec['per_context'][cid][f'u_g|{b}'][metric]
                diffs.append((y - x) if better_when_lower else (x - y))   # >0 means u_M better
            p, n = exact_sign_p(diffs)
            # "not worse" includes ties: exact_sign_p drops zeros, so n can be 0 when both
            # arms are clean. A tie is NOT a failure of the criterion, and reporting it as
            # one (as the first version did) turns a clean result into a fake FAIL.
            an[f'b={b}|{metric}'] = {'u_M_better_in': sum(d > 1e-12 for d in diffs),
                                     'u_M_not_worse_in': sum(d >= -1e-12 for d in diffs),
                                     'u_M_worse_in': sum(d < -1e-12 for d in diffs),
                                     'n_nontied': n, 'n_ties': sum(abs(d) <= 1e-12 for d in diffs),
                                     'exact_sign_p': p, 'median_diff': statistics.median(diffs),
                                     'mean_u_M': statistics.fmean(
                                         rec['per_context'][c][f'u_M|{b}'][metric] for c in cids),
                                     'mean_u_g': statistics.fmean(
                                         rec['per_context'][c][f'u_g|{b}'][metric] for c in cids),
                                     'mean_none': statistics.fmean(
                                         rec['per_context'][c][f'none|{budgets[0]}'][metric]
                                         for c in cids)}
        an[f'b={b}|stop_counts'] = {
            arm: {s: sum(1 for c in cids
                         if rec['per_context'][c][f'{arm}|{b if arm != "none" else budgets[0]}']['stop'] == s)
                  for s in ('eos', 'max')} for arm in arms}
    rec['analysis'] = an
    prim = {}
    need = max(1, math.ceil(7 / 8 * len(cids)))
    for b in budgets:
        rep = an[f'b={b}|rep4']; cap = an[f'b={b}|dlogp_ref']
        ok_rep = rep['u_M_not_worse_in'] >= need
        ok_cap = cap['u_M_not_worse_in'] >= need
        prim[str(b)] = {'rep4_u_M_not_worse_in': rep['u_M_not_worse_in'],
                        'rep4_u_M_worse_in': rep['u_M_worse_in'],
                        'rep4_ties': rep['n_ties'],
                        'dlogp_u_M_not_worse_in': cap['u_M_not_worse_in'],
                        'dlogp_u_M_worse_in': cap['u_M_worse_in'],
                        'dlogp_ties': cap['n_ties'],
                        'n': len(cids), 'need': need,
                        'primary_pass': bool(ok_rep and ok_cap)}
    rec['primary'] = prim

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'behavior_check.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                             encoding='utf8')
    lines = ['# Is the R7 gain degeneration? Judge-free behavioural check', '',
             f'- machine `{rec["machine"]}` / {rec["gpu"]} · layers {layers} · '
             f'max_new {a.max_new} · contexts {len(cids)}',
             f'- vectorised hook vs frozen `apply_steering`: max |dS| = {maxdiff:.1e} '
             f'({"bitwise equal" if maxdiff == 0.0 else "DIFFERS"})',
             f'- capability proxy: teacher-forced mean logP of `engine.REFERENCE_TEXT` '
             f'(sha256 {rec["reference_text_sha256"][:16]}…)', '',
             '## absolute levels (mean over contexts)', '',
             '| budget | arm | stop=eos | rep4 | longest imm. repeat | distinct-2 | dlogP ref |',
             '|---:|---|---:|---:|---:|---:|---:|']
    for b in budgets:
        for arm in arms:
            key = f'{arm}|{b if arm != "none" else budgets[0]}'
            eos = sum(1 for c in cids if rec['per_context'][c][key]['stop'] == 'eos')
            f4 = lambda m: statistics.fmean(rec['per_context'][c][key][m] for c in cids)
            lines.append(f'| {b} | `{arm}` | {eos}/{len(cids)} | {f4("rep4"):.4f} | '
                         f'{f4("longest_immediate_repeat"):.1f} | {f4("distinct2"):.4f} | '
                         f'{f4("dlogp_ref"):+.5f} |')
    lines += ['', '## frozen paired tests: is u_M worse than u_g? (positive = u_M better)', '',
              '| budget | metric | u_M better | ties | u_M worse | n non-tied | exact sign p | '
              'mean u_g | mean u_M | mean none |',
              '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for b in budgets:
        for metric in ('rep4', 'longest_immediate_repeat', 'distinct2', 'dlogp_ref'):
            x = an[f'b={b}|{metric}']
            sp = 'n/a (all ties)' if x['exact_sign_p'] is None else f'{x["exact_sign_p"]:.4f}'
            lines.append(f'| {b} | `{metric}` | {x["u_M_better_in"]} | {x["n_ties"]} | '
                         f'{x["u_M_worse_in"]} | {x["n_nontied"]} | {sp} | '
                         f'{x["mean_u_g"]:+.5f} | {x["mean_u_M"]:+.5f} | {x["mean_none"]:+.5f} |')
    lines += ['', '## PRIMARY verdict', '']
    for b in budgets:
        v = prim[str(b)]
        lines.append(f'- b={b}: rep4 not-worse in {v["rep4_u_M_not_worse_in"]}/{v["n"]} '
                     f'(worse in {v["rep4_u_M_worse_in"]}, ties {v["rep4_ties"]}), '
                     f'dlogP_ref not-worse in {v["dlogp_u_M_not_worse_in"]}/{v["n"]} '
                     f'(worse in {v["dlogp_u_M_worse_in"]}, ties {v["dlogp_ties"]}); '
                     f'need >={v["need"]}/{v["n"]} -> '
                     f'**{"PASS" if v["primary_pass"] else "FAIL"}**')
    (out / 'behavior_check.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/behavior_check.json and .md')


if __name__ == '__main__':
    main()
