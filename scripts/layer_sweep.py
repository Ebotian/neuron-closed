#!/usr/bin/env python3
"""Which layer should the dense budget-optimal direction act on?

STATUS (2026-10-29): RUN. Dry-run, a 2-layer smoke test (which reproduced cond(B) 495.5/530.1 and the M/g
ratio 1.37-1.40 from plans 2.15/2.21, so the harness is consistent with the earlier numbers), then all 32
layers in 7 min 54 s. Results and the four conclusions are in plan 2.32 and 07_results/layer_sweep_all32.
This file was originally committed by accident inside 37487411 while still unrun; that is now historical.

Every result in this line so far was measured on layers 13-14, inherited from the original
persona-matrix scan -- it was never chosen by measurement. Two registered questions need
this sweep:

1. `08_reference` §6.3 records a tension: the external Qwen3-8B atlas (36 layers) has its
   persona peak at relative depth 0.17-0.31 raw and **0.25-0.33** after stripping
   degeneration, while our L13-L14 of 32 layers is relative depth **0.41-0.44**. Different
   model, prompt and outcome measure, so it is a tension and not a refutation -- but only a
   sweep on OUR model can say where our readout's sensitive band actually is.
2. §2.21.3 found budget/||g|| = 1.000 at K=1 but 2.757 at full support over layers 13-14,
   i.e. the dose-budget metric M departs from identity only in the dense regime. cond(B) was
   495.5 / 530.1 on those two layers. Both the size of the M^+ advantage and the departure
   of M from I are properties of the layer's column geometry, so they may vary a lot by
   layer.

One backward gives the gradient for EVERY hooked layer, so pass 1 costs 2 grad-enabled
batches per context regardless of the layer count. Pass 2 builds What_l and B_l for one
layer at a time (9216 x 2560 float64 is 189 MB per layer, so holding all 32 would be 6 GB)
and measures every context x arm x budget inside it.

Arms per layer: u_g (~g, that layer only) and u_M (~M^+g, that layer only), each normalised
to the SAME physical budget b, so layers are compared at equal cost.

FROZEN reading (before the run):
  report the whole profile, the argmax layer, and the profile of the M/g ratio. No layer is
  excluded and no budget is cherry-picked. Compare the argmax's relative depth against
  08_reference's 0.25-0.33 band and against our inherited 0.41-0.44.
  §2.18.4 error discipline: layer-to-layer differences below 1e-02 are not claimed.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--layers', default='all', help='"all" or e.g. "8-18"')
    ap.add_argument('--budgets', default='0.05')
    ap.add_argument('--contexts', type=int, default=8)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--out', default='out/layer_sweep')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lex, fams = load_pools(a.lexicon, a.pool_variant)
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    ctx = [r for r in rows if r.get('split') == a.split][:a.contexts]
    if not ctx:
        raise SystemExit(f'no {a.split} contexts in {a.probes}')
    budgets = [float(x) for x in a.budgets.split(',')]
    n_layers_model = 32

    if a.layers == 'all':
        layers = list(range(n_layers_model))
    else:
        lo, hi = (int(x) for x in a.layers.split('-'))
        layers = list(range(lo, hi + 1))
    arms = ['u_g', 'u_M']
    n_meas = len(layers) * len(ctx) * len(arms) * len(budgets)

    print('Question : which layer should the dense budget-optimal direction act on?')
    print(f'Readout  : J2 = mean over families {sorted(fams)}, pool variant {a.pool_variant}')
    print(f'Layers   : {layers[0]}..{layers[-1]} ({len(layers)} of {n_layers_model})')
    print(f'Arms     : u_g (~g) and u_M (~M^+g), each within ONE layer, each at the SAME budget')
    print(f'Budgets  : {budgets}   contexts {len(ctx)}')
    print(f'Work     : pass1 = {len(ctx)}x{len(fams)} grad passes (one backward gives all layers) '
          f'+ {len(ctx)} baselines; pass2 = {n_meas} J2 evaluations = {n_meas * len(fams)} forwards')
    print('Frozen   : report the whole profile, the argmax layer and the M/g ratio profile; no layer')
    print('           excluded, no budget cherry-picked. Compare relative depth against')
    print('           08_reference 0.25-0.33 (their degeneration-stripped peak) and our inherited')
    print(f'           L13-14 = {13 / n_layers_model:.3f}-{14 / n_layers_model:.3f}.')
    print('           Layer-to-layer differences below 1e-02 are not claimed (§2.18.4).')
    if a.dry_run:
        print(f'[dry-run] would write {a.out}/layer_sweep.json and .md')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import (apply_steering, column_norms, estimate_p50, model_lm, score_for_dose)
    from progress import Progress
    from semantic import batch_mean_logp, build_batch, language_log_mass, sem_gradient

    dev = 'cuda:0'
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).to(dev).eval()
    model.requires_grad_(False)
    lm = model_lm(model)
    if len(lm.layers) != n_layers_model:
        raise RuntimeError(f'expected {n_layers_model} layers, got {len(lm.layers)}')
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device=dev)
    print('p50 range over swept layers: '
          f'{min(p50.values()):.4f} .. {max(p50.values()):.4f}')

    # ---------------- pass 1: gradients for all layers, and baselines ----------------
    prog = Progress(len(ctx) * (len(fams) + 1) + n_meas)
    G_ctx, batches, base = {}, {}, {}
    for c in ctx:
        cid = c['id']
        gsum, jfam = None, {}
        for fam, f in sorted(fams.items()):
            j2, gg = sem_gradient(tok, model, lm, layers, c['prompt'], f['template'], f, dev,
                                  a.max_seq)
            jfam[fam] = j2
            sc = {l: score_for_dose(gg[l].float(), p50[l], norms[l].to(gg[l].device)) for l in layers}
            gsum = sc if gsum is None else {l: gsum[l] + sc[l] for l in layers}
            prog.step(f'{cid} g {fam} (all {len(layers)} layers) J2={j2:+.4f}')
        G_ctx[cid] = {l: (gsum[l] / len(fams)).double().to(dev) for l in layers}
        batches[cid] = {fam: build_batch(tok, c['prompt'], f['template'], f['pos'], f['neg'],
                                        dev, a.max_seq) for fam, f in sorted(fams.items())}
        with torch.inference_mode():
            vals = []
            for fam, (ids, pre, suf, kinds) in batches[cid].items():
                S = batch_mean_logp(model, ids, pre, suf)
                jp, _ = language_log_mass(S[kinds > 0].detach().float())
                jn, _ = language_log_mass(S[kinds < 0].detach().float())
                vals.append(jp - jn)
        base[cid] = sum(vals) / len(vals)
        prog.step(f'{cid} base J2={base[cid]:+.6f}')

    def j2_of(cid, iv):
        with apply_steering(lm, iv, p50, norms), torch.inference_mode():
            vals = []
            for fam, (ids, pre, suf, kinds) in batches[cid].items():
                S = batch_mean_logp(model, ids, pre, suf)
                jp, _ = language_log_mass(S[kinds > 0].detach().float())
                jn, _ = language_log_mass(S[kinds < 0].detach().float())
                vals.append(jp - jn)
        return sum(vals) / len(vals)

    # ---------------- pass 2: one layer at a time ----------------
    rec = {'schema': 'qwentest-layer-sweep-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model,
           'n_layers_model': n_layers_model, 'layers': layers, 'budgets': budgets, 'arms': arms,
           'contexts': [c['id'] for c in ctx], 'pool_variant': a.pool_variant,
           'lexicon_sha256': hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest(),
           'p50': {str(l): p50[l] for l in layers}, 'baseline_J2': base,
           'per_layer': {}}

    for l in layers:
        Wt = lm.layers[l].mlp.down_proj.weight.detach().double().T          # 9216 x 2560
        Wt = Wt / Wt.norm(dim=1, keepdim=True).clamp(min=1e-12)
        Bl = Wt.T @ Wt
        entry = {'cond_B': float(torch.linalg.cond(Bl)), 'per_context': {}}
        for c in ctx:
            cid = c['id']
            g = G_ctx[cid][l]
            e = Wt.T @ g
            ds = torch.linalg.solve(Bl, e)
            uM = Wt @ torch.linalg.solve(Bl, ds)
            gn = float(g.norm())
            dirs = {'u_g': g, 'u_M': uM}
            per = {'norm_g': gn, 'baseline_J2': base[cid]}
            for arm, v in dirs.items():
                bud = float((v @ Wt).norm())
                per[f'budget_over_norm_g_{arm}'] = bud / gn if gn else None
                for b in budgets:
                    if bud < 1e-12:
                        raise SystemExit(f'degenerate direction L{l} {cid} {arm}')
                    u = v * (b / bud)
                    nz = torch.nonzero(u.abs() > 1e-15).flatten()
                    iv = [{'layer': l, 'neuron': int(i), 'dose': float(u[i])} for i in nz.tolist()]
                    val = j2_of(cid, iv)
                    per[f'{arm}|{b}'] = {'dJ2': val - base[cid],
                                         'predicted': float(g @ u),
                                         'n_nonzero': len(iv)}
                    prog.step(f'L{l} {cid} b={b} {arm} dJ2={val - base[cid]:+.5f}')
            per['ratio_M_over_g_predicted'] = (
                math.sqrt(float(e @ torch.linalg.solve(Bl, ds))) / (gn ** 2 / float(e.norm()))
                if gn and float(e.norm()) > 0 else None)
            entry['per_context'][cid] = per
        rec['per_layer'][str(l)] = entry
        del Wt, Bl
        torch.cuda.empty_cache()

    # ---------------- analysis ----------------
    cids = rec['contexts']
    prof = {}
    for l in layers:
        e = rec['per_layer'][str(l)]
        row = {'relative_depth': l / n_layers_model, 'cond_B': e['cond_B']}
        for arm in arms:
            for b in budgets:
                row[f'{arm}|{b}'] = statistics.fmean(
                    e['per_context'][c][f'{arm}|{b}']['dJ2'] for c in cids)
        for b in budgets:
            row[f'ratio|{b}'] = row[f'u_M|{b}'] / row[f'u_g|{b}'] if row[f'u_g|{b}'] else None
            wins = sum(1 for c in cids
                       if e['per_context'][c][f'u_M|{b}']['dJ2'] > e['per_context'][c][f'u_g|{b}']['dJ2'])
            row[f'wins_M|{b}'] = f'{wins}/{len(cids)}'
        row['budget_over_norm_g'] = statistics.fmean(
            e['per_context'][c]['budget_over_norm_g_u_g'] for c in cids)
        prof[str(l)] = row
    rec['profile'] = prof
    b0 = str(budgets[0])
    best = max(layers, key=lambda l: prof[str(l)][f'u_M|{b0}'])
    worst = min(layers, key=lambda l: prof[str(l)][f'u_M|{b0}'])
    rec['argmax_layer'] = {'layer': best, 'relative_depth': best / n_layers_model,
                           'dJ2_u_M': prof[str(best)][f'u_M|{b0}'],
                           'worst_layer': worst, 'worst_dJ2_u_M': prof[str(worst)][f'u_M|{b0}'],
                           'spread_best_over_worst': prof[str(best)][f'u_M|{b0}']
                           / prof[str(worst)][f'u_M|{b0}'] if prof[str(worst)][f'u_M|{b0}'] else None,
                           'inherited_layers_13_14_rank': sorted(
                               (prof[str(l)][f'u_M|{b0}'] for l in layers), reverse=True
                           ).index(prof['13'][f'u_M|{b0}']) + 1 if '13' in prof else None}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'layer_sweep.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                          encoding='utf8')
    lines = ['# Which layer should the dense budget-optimal direction act on?', '',
             f'- machine `{rec["machine"]}` / {rec["gpu"]} · {len(layers)} of {n_layers_model} '
             f'layers · contexts {len(cids)} · budgets {budgets}',
             f'- each arm acts within ONE layer and is normalised to the SAME physical budget',
             f'- **argmax layer = L{best}** (relative depth {best / n_layers_model:.3f}), '
             f'dJ2 {prof[str(best)][f"u_M|{b0}"]:+.5f}; worst = L{worst} '
             f'({prof[str(worst)][f"u_M|{b0}"]:+.5f})',
             f'- inherited L13-14 = relative depth {13 / n_layers_model:.3f}-'
             f'{14 / n_layers_model:.3f}; 08_reference degeneration-stripped peak = 0.25-0.33', '',
             f'| layer | rel depth | dJ2 u_g (b={b0}) | dJ2 u_M (b={b0}) | ratio M/g | wins | '
             'budget/||g|| | cond(B) |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    for l in layers:
        r = prof[str(l)]
        lines.append(f'| {l} | {r["relative_depth"]:.3f} | {r[f"u_g|{b0}"]:+.5f} | '
                     f'{r[f"u_M|{b0}"]:+.5f} | {r[f"ratio|{b0}"]:.4f} | {r[f"wins_M|{b0}"]} | '
                     f'{r["budget_over_norm_g"]:.3f} | {r["cond_B"]:.4g} |')
    for b in budgets[1:]:
        lines += ['', f'## budget {b}', '', f'| layer | dJ2 u_g | dJ2 u_M | ratio | wins |',
                  '|---:|---:|---:|---:|---:|']
        for l in layers:
            r = prof[str(l)]
            lines.append(f'| {l} | {r[f"u_g|{str(b)}"]:+.5f} | {r[f"u_M|{str(b)}"]:+.5f} | '
                         f'{r[f"ratio|{str(b)}"]:.4f} | {r[f"wins_M|{str(b)}"]} |')
    (out / 'layer_sweep.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/layer_sweep.json and .md')


if __name__ == '__main__':
    main()
