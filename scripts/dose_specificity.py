#!/usr/bin/env python3
"""Plan 06: is L14N8949 a hostility-specific direction, or a generic likelihood lever?

Four readouts, pre-registered decisions S1-S4 (see 06_plan-specificity-dose.md):

  R1  J2  language-mass readout (adj + verb slot families from the scan manifest) on the
          held-out prompts, target across doses
  R2  J1  cross-trait string margins on SHARED contexts (probes/specificity.jsonl) - used
          only for the relative ranking, not as a reproducible readout
  R3  placebo: neutral_2 vs neutral_1 on the same contexts
  R4  generic: mean log-probability of engine.REFERENCE_TEXT (catches "everything gets better")

  --mirror additionally measures the flipped-sign arm of the same coordinate. It is
  DESCRIPTIVE ONLY: the frozen rules S1-S4 are unchanged, and asymmetry is expected
  (a first-order sensitivity is not a symmetric lever).

Every measurement prints `[i/N] elapsed, eta ...` via the package's progress.Progress.
--dry-run needs no GPU and no torch: it prints the plan and the measurement count.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def resolve_pkg(given):
    here = Path(__file__).resolve().parent
    cands = [Path(given)] if given else []
    cands += [here.parent / 'vendor/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
              Path('/root/autodl-tmp/qwen-gradient-discovery-v6')]
    for c in cands:
        if (c / 'engine.py').is_file():
            return c
    raise SystemExit('package not found; pass --pkg <dir with engine.py>')


def parse_coord(s):
    layer, neuron, sign = s.split(':')
    return {'layer': int(layer), 'neuron': int(neuron), 'sign': int(sign)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='', help='frozen package dir (engine.py, semantic.py, progress.py)')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--scan-dir', required=True, help='J2 scan dir (for pools/templates/p50/controls)')
    ap.add_argument('--probes', required=True, help='probes jsonl with held-out hostility rows')
    ap.add_argument('--candidate', default='14:8949:+1', help='layer:neuron:sign')
    ap.add_argument('--doses', default='0.005,0.01,0.02,0.05')
    ap.add_argument('--primary', type=float, default=0.02)
    ap.add_argument('--controls', type=int, default=6, help='same-layer random neurons (own predicted sign)')
    ap.add_argument('--mirror', action='store_true',
                    help='also measure the flipped-sign arm (descriptive, not part of S1-S4)')
    ap.add_argument('--seed', type=int, default=1729)
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--out', default='out/dose')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    doses = [float(x) for x in a.doses.split(',')]
    if a.primary not in doses:
        raise SystemExit(f'--primary {a.primary} must be one of --doses {doses}')
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from progress import Progress                       # torch-free

    scan = Path(a.scan_dir)
    meta = json.loads((scan / 'manifest.json').read_text(encoding='utf8'))
    cal = json.loads((scan / 'calibration.json').read_text(encoding='utf8'))
    fams = {k: {'pos': v['pos'], 'neg': v['neg'], 'template': meta['templates'][k]}
            for k, v in meta['pools'].items()}
    heldout = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    heldout = [r for r in heldout if r['trait'] == 'hostility' and r['split'] == 'heldout']
    spec = [json.loads(l) for l in (pkg / 'probes/specificity.jsonl').read_text(encoding='utf8').splitlines() if l.strip()]
    cand = parse_coord(a.candidate)
    control_doses = sorted({a.primary, max(doses)})

    n_j2_base = len(heldout)
    n_target = len(heldout) * len(doses)
    n_ctrl = len(heldout) * len(control_doses) * a.controls
    n_trait = len(spec) * len(spec[0]['continuations'])
    n_placebo = len(spec)
    n_generic = len(doses) + 1
    n_mirror = len(heldout) * len(doses) if a.mirror else 0
    total = n_j2_base + n_target + n_ctrl + n_trait * 2 + n_placebo * 2 + n_generic + n_mirror
    print(f'Plan: candidate L{cand["layer"]}N{cand["neuron"]}{cand["sign"]:+d}; doses {doses} '
          f'(primary {a.primary}); {a.controls} same-layer random controls; '
          f'{len(heldout)} held-out probes; {len(spec)} shared contexts x '
          f'{len(spec[0]["continuations"])} traits')
    print(f'      measurements: J2 baseline {n_j2_base} + target {n_target} + controls {n_ctrl} '
          f'+ cross-trait {n_trait} x2 + placebo {n_placebo} x2 + generic {n_generic}'
          + (f' + mirror {n_mirror}' if a.mirror else '') + f' = {total}')
    print(f'      package {pkg} | scan {scan.name} | estimate ~{total * 0.25 + 40:.0f} s '
          f'(0.25 s/measurement + 40 s hash/load; first run after a reboot pays Triton compile)')
    if a.dry_run:
        return

    import torch
    from engine import (REFERENCE_TEXT, apply_steering, column_norms, load_model, pair_objective)
    from semantic import j2_value

    tok, model, lm = load_model(a.model)
    layers = meta['layers']
    p50 = {int(l): float(v) for l, v in cal['p50'].items()}
    norms = column_norms(lm, layers)
    device = 'cuda:0'

    def j2_mean(prompt, iv):
        vals = [j2_value(tok, model, lm, p50, norms, prompt, f['template'], f, iv, device, a.max_seq)
                for f in fams.values()]
        return statistics.mean(vals)

    @torch.inference_mode()
    def ref_logp(iv=None):
        ids = tok(REFERENCE_TEXT, return_tensors='pt').input_ids.to(device)
        with apply_steering(lm, iv or [], p50, norms):
            logits = model(input_ids=ids, use_cache=False).logits
        lp = -torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1), reduction='mean')
        return float(lp)

    def trait_margin(ctx, text, iv):
        probe = {'prompt': ctx['prompt'], 'positive': text, 'negative': ctx['neutral_1']}
        return pair_objective(tok, model, lm, probe, device, p50, norms, iv, a.max_seq)[0]

    # controls: same-layer random neurons with their own predicted sign, from the scan's gradients
    grads = torch.load(scan / 'hostility_all_gradients.pt', map_location='cpu', weights_only=True)
    row = grads[int(cand['layer'])]['mean']
    import random
    rng = random.Random(a.seed)
    used = {(cand['layer'], cand['neuron'])}
    controls = []
    while len(controls) < a.controls:
        n = rng.randrange(int(row.numel()))
        if (cand['layer'], n) in used:
            continue
        used.add((cand['layer'], n))
        g = float(row[n])
        controls.append({'layer': int(cand['layer']), 'neuron': n, 'sign': 1 if g >= 0 else -1,
                         'predicted_gradient': g})

    prog = Progress(total)
    out = {'candidate': cand, 'doses': doses, 'primary': a.primary, 'controls': controls,
           'heldout': [r['id'] for r in heldout], 'families': list(fams),
           'runtime': meta.get('runtime'), 'scan_identity': meta.get('model_fingerprint', {}).get('content_sha256')}

    def iv_of(coord, dose):
        return [{'layer': coord['layer'], 'neuron': coord['neuron'], 'dose': dose * coord['sign']}]

    base_j2 = {}
    for r in heldout:
        base_j2[r['id']] = j2_mean(r['prompt'], [])
        prog.step(f'J2 baseline {r["id"]}')
    curve = {}
    for d in doses:
        vals = []
        for r in heldout:
            v = j2_mean(r['prompt'], iv_of(cand, d))
            vals.append(v - base_j2[r['id']])
            prog.step(f'J2 target d={d} {r["id"]} dJ2={vals[-1]:+.5f}')
        curve[d] = {'mean': statistics.mean(vals), 'sd': statistics.stdev(vals), 'per_probe': vals}
    curve_mirror = {}
    if a.mirror:
        mirrored = {'layer': cand['layer'], 'neuron': cand['neuron'], 'sign': -cand['sign']}
        for d in doses:
            vals = []
            for r in heldout:
                v = j2_mean(r['prompt'], iv_of(mirrored, d))
                vals.append(v - base_j2[r['id']])
                prog.step(f'J2 mirror (sign {-cand["sign"]:+d}) d={d} {r["id"]} dJ2={vals[-1]:+.5f}')
            curve_mirror[d] = {'mean': statistics.mean(vals), 'sd': statistics.stdev(vals),
                               'per_probe': vals}
    ctrl = {d: [] for d in control_doses}
    for cd in control_doses:
        for c in controls:
            vals = [j2_mean(r['prompt'], iv_of(c, cd)) - base_j2[r['id']] for r in heldout]
            for r, v in zip(heldout, vals):
                prog.step(f'J2 control L{c["layer"]}N{c["neuron"]} d={cd} dJ2={v:+.5f}')
            ctrl[cd].append({'coord': f'L{c["layer"]}N{c["neuron"]}{c["sign"]:+d}',
                             'mean': statistics.mean(vals)})
    traits = sorted(spec[0]['continuations'])
    matrix, placebo = {}, {'base': [], 'steered': [], 'delta': []}
    for ctx in spec:
        for t in traits:
            b = trait_margin(ctx, ctx['continuations'][t], [])
            s = trait_margin(ctx, ctx['continuations'][t], iv_of(cand, a.primary))
            prog.step(f'J1 trait {ctx["id"]} {t} d={s - b:+.5f}')
            matrix.setdefault(t, []).append(s - b)
        pb = trait_margin(ctx, ctx['neutral_2'], [])
        ps = trait_margin(ctx, ctx['neutral_2'], iv_of(cand, a.primary))
        prog.step(f'J1 placebo {ctx["id"]} d={ps - pb:+.5f}')
        placebo['delta'].append(ps - pb)
    gb = ref_logp()
    prog.step('generic reference logP baseline')
    generic = {}
    for d in doses:
        generic[d] = ref_logp(iv_of(cand, d)) - gb
        prog.step(f'generic reference d={d} dlogP={generic[d]:+.5f}')

    # ---- decisions S1-S4 (frozen in the plan) ----
    seq = [curve[d]['mean'] for d in doses]
    ups = sum(1 for x, y in zip(seq, seq[1:]) if y >= x)
    # frozen rule: >=3 ascending steps out of the 4 planned doses. Short runs cannot
    # judge monotonicity, so they report None (inconclusive) instead of a verdict.
    s1 = (ups >= 3) if len(doses) >= 4 else None
    tm = {t: statistics.mean(v) for t, v in matrix.items()}
    ordered = sorted(tm.items(), key=lambda kv: -kv[1])
    s2 = ordered[0][0] == 'hostility' and (ordered[0][1] >= 2 * max(ordered[1][1], 1e-9))
    s3 = abs(generic[a.primary]) < 0.5 * abs(tm.get('hostility', 0.0))
    s4 = abs(statistics.mean(placebo['delta'])) < abs(tm.get('hostility', 0.0)) / 3
    ctrl_primary = [c['mean'] for c in ctrl[a.primary]]
    out.update({'curve': curve, 'curve_mirror': curve_mirror,
                'asymmetry': {d: curve[d]['mean'] - curve_mirror[d]['mean']
                              for d in doses} if a.mirror else {}, 'controls_by_dose': ctrl, 'trait_matrix': matrix,
                'trait_means': tm, 'placebo': placebo, 'generic': generic,
                'verdict': {'S1_monotone': s1, 'S1_ups': ups, 'S2_specific': s2,
                            'S2_ranking': ordered, 'S3_not_generic': s3,
                            'S4_placebo_flat': s4,
                            'target_primary': curve[a.primary]['mean'],
                            'controls_primary_mean': statistics.mean(ctrl_primary) if ctrl_primary else None,
                            'controls_below_target': (sum(1 for c in ctrl_primary
                                                          if c < curve[a.primary]['mean'])
                                                      if ctrl_primary else None),
                            'controls_n': len(ctrl_primary)}})

    outp = Path(a.out); outp.mkdir(parents=True, exist_ok=True)
    (outp / 'dose.json').write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf8')
    lines = ['# Plan 06 — specificity & dose-response', '',
             f'candidate `L{cand["layer"]}N{cand["neuron"]}{cand["sign"]:+d}` · primary dose {a.primary} · '
             f'{len(heldout)} held-out probes · {len(controls)} same-layer random controls', '',
             '## Dose-response (J2 language mass, held-out mean)', '',
             '| dose | mean ΔJ2 | sd | vs controls |', '|---|---:|---:|---:|']
    for d in doses:
        cmp = ''
        if d in ctrl:
            cmp = f'{sum(1 for c in ctrl[d] if c["mean"] < curve[d]["mean"])}/{len(ctrl[d])} controls below'
        lines.append(f'| {d} | {curve[d]["mean"]:+.5f} | {curve[d]["sd"]:.5f} | {cmp} |')
    if curve_mirror:
        lines += ['', '## Mirror arm (descriptive only — frozen rules unchanged)', '',
                  '| dose | ΔJ2 (+sign) | ΔJ2 (−sign) | asymmetry |', '|---|---:|---:|---:|']
        for d in doses:
            lines.append(f'| {d} | {curve[d]["mean"]:+.5f} | {curve_mirror[d]["mean"]:+.5f} | '
                         f'{out["asymmetry"][d]:+.5f} |')
        lines.append('')
        lines.append('Asymmetry is expected: a first-order sensitivity is not a symmetric lever. '
                     'A mirror arm that is *equally* strong would instead suggest a generic '
                     'perturbation effect rather than a direction.')
    lines += ['', '## Cross-trait (J1 string margin, shared contexts, primary dose)', '',
              '| trait | mean Δmargin |', '|---|---:|']
    for t, v in ordered:
        lines.append(f'| {t} | {v:+.5f} |')
    lines += ['', f'placebo (neutral vs neutral) Δ = {statistics.mean(placebo["delta"]):+.5f}',
              f'generic reference-text ΔlogP at primary = {generic[a.primary]:+.5f}', '',
              '## Verdict (rules frozen before the run)', '']
    for k in ('S1_monotone', 'S2_specific', 'S3_not_generic', 'S4_placebo_flat'):
        v = out['verdict'][k]
        lines.append(f'- {k}: **{"PASS" if v is True else ("FAIL" if v is False else "n/a (needs 4 doses)")}**')
    lines.append(f'- S1 detail: {ups}/{len(doses)-1} ascending steps')
    lines.append('- S2 detail: ' + ', '.join(f'{t} {v:+.4f}' for t, v in ordered))
    (outp / 'dose.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n'.join(lines))
    judged = [out['verdict'][k] for k in ('S1_monotone', 'S2_specific', 'S3_not_generic', 'S4_placebo_flat')
              if out['verdict'][k] is not None]
    ok = bool(judged) and all(judged)
    verdict = 'H1 (hostility-specific)' if ok else ('INCONCLUSIVE (S1 needs 4 doses)' if judged == [] or
                                                    out['verdict']['S1_monotone'] is None and all(
                                                        out['verdict'][k] for k in ('S2_specific', 'S3_not_generic',
                                                                                    'S4_placebo_flat'))
                                                    else 'NOT H1 — see S1-S4 above')
    print(f'\nVERDICT: {verdict}')
    print('wrote', outp / 'dose.json', 'and', outp / 'dose.md')


if __name__ == '__main__':
    main()
