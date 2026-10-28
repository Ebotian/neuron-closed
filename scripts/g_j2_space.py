#!/usr/bin/env python3
"""Rebuild the control-cost matrix G on the output space that J2 ACTUALLY reads.

Why this run exists. `g_feasibility.py` built G = J'J with J = d h_final / d u at the
LAST PROMPT POSITION of a SINGLE sequence. But the active readout is J2
(`semantic.py`), which is multi-sequence and multi-position: `build_batch` makes one
sequence per slot word, `batch_mean_logp` averages log P over every suffix position, and
`scan_sem.py` averages the two template families (adj 19+19, verb 14+14). So that G was
built on a MISMATCHED output space -- and the total-vs-collateral identity shows a
mismatch is exactly the case where the choice of cost stops being vacuous.

This script builds G on the matched space and answers one question with numbers:

    does the mismatch change the control law?

Reported per context, all at the same candidate set S and the same dose convention:
  * cos(u*_matched, u*_mismatched)      <- the decision number
  * cos(u*_matched, u*_(G=I))           <- how far either is from current practice
  * off-diagonal / diagonal agreement between the two G's
  * best PAIR under each G, and whether it differs
  * split-half reliability of the matched G (so the comparison can be read against noise)
  * the matched G's anisotropy vs the isotropic null (median |rho|, condition number)

The mismatched G is NOT recomputed: it is read from a previous `g_feasibility.py` result
(--compare-to), same machine, same S, same p50, so the two matrices are directly
comparable and the run stays small. Coordinates are asserted to match before use.

Metric choice, stated explicitly: cost = squared Euclidean movement of the hidden states
the readout reads (NOT logit-space movement, which would weight directions by
lm_head'lm_head). Same metric as the previous probe; only the index set differs.

GPU: per context, 2 forwards (one per family) + (m + 1) backwards on the retained graphs.

    [i/N] elapsed, eta ...   per backward (package progress.Progress)
"""
from __future__ import annotations
import argparse, hashlib, json, platform, random, statistics, sys
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
    """Same rule as scan_sem.py::load_pools (inlined so we do not import its audit deps)."""
    lex = json.loads(Path(path).read_text(encoding='utf8'))
    fams = {}
    for fam, d in lex['families'].items():
        key = 'pos' if variant == 'full' else f'pos_{variant}'
        tmpl = d.get('template') or lex.get('templates', {}).get(fam)
        if not tmpl:
            raise ValueError(f'family {fam} has no template')
        fams[fam] = {'pos': d[key], 'neg': d['neg'], 'template': tmpl}
    return lex, fams


def build_S(seed, k, layers, per_layer):
    """Identical construction to g_feasibility.py, so the two G's share coordinates."""
    from engine import KNOWN_EVIL
    rng = random.Random(seed)
    coords = [(int(l), int(n)) for l, n, _ in KNOWN_EVIL if l in layers]
    seen = set(coords)
    while len(coords) < k:
        c = (rng.choice(layers), rng.randrange(per_layer))
        if c not in seen:
            seen.add(c); coords.append(c)
    return coords[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--compare-to', default='', help='g_feasibility.json from the mismatched probe')
    ap.add_argument('--layers', default='13-14')
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--projections', type=int, default=96)
    ap.add_argument('--contexts', type=int, default=3)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--out', default='out/g_j2_space')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
    if a.projections <= a.k:
        raise SystemExit(f'--projections ({a.projections}) must exceed --k ({a.k})')
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    ctx = [r for r in rows if r.get('split') == a.split][:a.contexts]
    if not ctx:
        raise SystemExit(f'no {a.split} contexts in {a.probes}')

    prev, prev_coords, prev_p50 = {}, [], {}
    if a.compare_to:
        old = json.loads(Path(a.compare_to).read_text(encoding='utf8'))
        prev = {c: r['G'] for c, r in old['per_context'].items()}
        prev_coords = [(x['layer'], x['neuron']) for x in old['coordinates']]
        prev_p50 = {int(k): v for k, v in old['p50'].items()}
    n_seq = sum(len(f['pos']) + len(f['neg']) for f in fams.values())
    n_back = len(ctx) * (a.projections + 1)

    print('Question : does building G on J2\'s OWN output space change the control law?')
    print(f'Readout  : J2 = mean over families {sorted(fams)} of '
          f'(log mean e^S+ - log mean e^S-), pool variant {a.pool_variant}')
    for fam, f in sorted(fams.items()):
        print(f'           {fam}: template {f["template"]!r}  pos={len(f["pos"])} neg={len(f["neg"])}')
    print(f'Output   : matched space = all read positions of all {n_seq} slot sequences; '
          f'mismatched = last prompt position of 1 sequence')
    print(f'Metric   : squared Euclidean movement of hidden states (not logit space)')
    print(f'Layers   : {layers}   |S| = {a.k}   projections m = {a.projections}')
    print(f'Contexts : {[c["id"] for c in ctx]}')
    print(f'Compare  : {a.compare_to or "(none)"}'
          + (f'  -> {sorted(set(prev) & {c["id"] for c in ctx})} available' if prev else ''))
    print(f'Work     : {len(ctx)} x (2 forwards + {a.projections + 1} backwards) = {n_back} backwards')
    if a.dry_run:
        print('\n[dry-run] would report per context:')
        print('  cos(u*_matched, u*_mismatched), cos(u*_matched, u*_I), cos(u*_mismatched, u*_I)')
        print('  Pearson of off-diagonal entries and Spearman of diagonals between the two G')
        print('  best pair under each G (same g) and whether it differs; Q gain of matched over I')
        print('  split-half reliability of the matched G; median |rho| and condition number')
        print(f'[dry-run] would write {a.out}/g_j2_space.json and .md')
        return

    import torch
    import torch.nn.functional as F
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import column_norms, estimate_p50, load_model, prompt_prefix
    from progress import Progress
    from semantic import build_batch, language_log_mass

    tok, model, lm = load_model(a.model)
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device='cuda:0')
    print('p50 (this machine):', {l: round(p50[l], 6) for l in layers})
    if prev_p50 and max(abs(prev_p50[l] - p50[l]) / p50[l] for l in layers) > 0.01:
        raise SystemExit('p50 differs from the comparison run by >1%; the two G are not comparable')

    per_layer = len(norms[layers[0]])
    S = build_S(a.seed, a.k, layers, per_layer)
    if prev_coords and prev_coords[:len(S)] != S:
        raise SystemExit('coordinate set differs from the comparison run; refusing to compare')
    idx_by_layer = {l: [i for i, (ll, _) in enumerate(S) if ll == l] for l in layers}
    K = len(S)
    print(f'S = {K} coordinates (first 4: {S[:4]}), '
          + ', '.join(f'L{l}:{len(idx_by_layer[l])}' for l in layers))

    dev = 'cuda:0'
    prog = Progress(n_back)
    done = 0
    rec = {'schema': 'qwentest-g-j2-space-v1', 'machine': platform.node(),
           'model': a.model, 'layers': layers, 'k': K, 'projections': a.projections,
           'pool_variant': a.pool_variant, 'seed': a.seed,
           'lexicon_sha256': hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest(),
           'templates': {f: fams[f]['template'] for f in fams},
           'n_slot_sequences': n_seq, 'p50': {str(l): p50[l] for l in layers},
           'compare_to': a.compare_to or None,
           'coordinates': [{'layer': l, 'neuron': n} for l, n in S],
           'metric': 'squared Euclidean movement of the hidden states the readout reads',
           'per_context': {}}

    def select(grads):
        """dose units, summed over batch+positions (broadcast injection), restricted to S."""
        v = torch.zeros(K, dtype=torch.float64)
        for l, g in zip(layers, grads):
            s = g.detach().double().sum(dim=tuple(range(g.ndim - 1)))
            s = s * (p50[l] / norms[l].to(s.device).double().clamp(min=1e-12))
            for i in idx_by_layer[l]:
                v[i] = s[S[i][1]]
        return v.cpu()

    for c in ctx:
        # ---- one grad-enabled forward per family; keep both graphs alive ----
        from engine import capture_mlp_inputs
        graphs, losses, reads = {}, {}, {}
        for fam, f in sorted(fams.items()):
            ids, pre, suf, kinds = build_batch(tok, c['prompt'], f['template'],
                                              f['pos'], f['neg'], dev, a.max_seq)
            with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
                out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
                logits = out.logits[:, pre - 1: pre - 1 + suf]
                target = ids[:, pre: pre + suf]
                if logits.shape[1] != suf:
                    raise AssertionError(f'token alignment failed for family {fam}')
                lp = -F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                      target.reshape(-1), reduction='none')
                Sv = lp.reshape(ids.shape[0], suf).mean(dim=1)      # == batch_mean_logp
                with torch.no_grad():                               # exact 1st derivative
                    Sd = Sv.detach()
                    _, w_pos = language_log_mass(Sd[kinds > 0])
                    _, w_neg = language_log_mass(Sd[kinds < 0])
                    coeff = torch.zeros_like(Sd)
                    coeff[kinds > 0] = w_pos
                    coeff[kinds < 0] = -w_neg
                H = out.hidden_states[-1]
                reads[fam] = H[:, pre - 1: pre - 1 + suf, :].float()   # what J2 reads
                losses[fam] = (coeff * Sv).sum()
                graphs[fam] = {l: cap[l] for l in layers}
            rec.setdefault('shapes', {})[fam] = {'n_seq': int(ids.shape[0]), 'suffix_len': int(suf),
                                                 'prefix_len': int(pre), 'read_dim': int(reads[fam].numel())}
        # readout = mean over families, exactly as scan_sem.py aggregates
        loss = sum(losses.values()) / len(losses)

        inputs = [graphs[f][l] for f in sorted(fams) for l in layers]
        g = None
        for fam in sorted(fams):
            grads = torch.autograd.grad(loss, [graphs[fam][l] for l in layers],
                                        retain_graph=True)
            v = select(grads)
            g = v if g is None else g + v
        g = g / len(fams)                                     # matches scan_sem's mean
        done += 1
        prog.step(f'{c["id"]} g  |g|={float(g.norm()):.4g}')

        wrng = torch.Generator().manual_seed(
            a.seed + int(hashlib.sha256(c['id'].encode()).hexdigest()[:8], 16))
        V = []
        for _ in range(a.projections):
            obj = 0
            for fam in sorted(fams):
                R = reads[fam]
                w = torch.randn(R.shape, generator=wrng, dtype=torch.float32).to(R.device)
                obj = obj + (R * w).sum()
            grads = torch.autograd.grad(obj, [graphs[f][l] for f in sorted(fams) for l in layers],
                                        retain_graph=True)
            per_fam = {f: grads[i * len(layers):(i + 1) * len(layers)]
                       for i, f in enumerate(sorted(fams))}
            v = None
            for fam in sorted(fams):
                vv = select(per_fam[fam])
                v = vv if v is None else v + vv
            V.append(v)
            done += 1
            prog.step(f'{c["id"]} proj {len(V)}/{a.projections}')
        V = torch.stack(V)
        G = (V.T @ V) / V.shape[0]
        del graphs, reads, losses, out, H, logits
        torch.cuda.empty_cache()

        # ---------------- analysis ----------------
        Gm = torch.tensor(prev[c['id']], dtype=torch.float64) if c['id'] in prev else None
        if Gm is not None and Gm.shape[0] != K:
            raise SystemExit(f'comparison G is {Gm.shape[0]}x{Gm.shape[0]} but |S| = {K}; '
                             f'--k must match the run in --compare-to')
        off = ~torch.eye(K, dtype=torch.bool)

        def rho_of(M):
            dg = torch.diagonal(M).clamp(min=1e-300)
            return M / torch.sqrt(dg[:, None] * dg[None, :])

        def ustar(M, gv, delta=0.05):
            Gi = torch.linalg.pinv(M)
            return delta * (Gi @ gv) / float(gv @ Gi @ gv)

        cos = lambda x, y: float(x @ y / (x.norm() * y.norm() + 1e-300))
        u_mat = ustar(G, g)
        u_I = ustar(torch.eye(K, dtype=torch.float64), g)
        r = {'J2_gradient_abs_norm': float(g.norm()),
             'cos_u_matched_vs_I': cos(u_mat, u_I)}
        if Gm is not None:
            u_mis = ustar(Gm, g)
            r['cos_u_matched_vs_mismatched'] = cos(u_mat, u_mis)
            r['cos_u_mismatched_vs_I'] = cos(u_mis, u_I)
            r['rel_diff_u_matched_mismatched'] = float((u_mat - u_mis).norm() / u_mat.norm())
            r['pearson_offdiag_G'] = statistics.correlation(
                [float(x) for x in G[off]], [float(x) for x in Gm[off]])
            dgm = [float(G[i, i]) for i in range(K)]
            dgp = [float(Gm[i, i]) for i in range(K)]
            rk = lambda xs: [sorted(range(len(xs)), key=lambda i: xs[i]).index(j) for j in range(len(xs))]
            r['pearson_diag_G'] = statistics.correlation(dgm, dgp)
            r['spearman_diag_G'] = statistics.correlation(rk(dgm), rk(dgp))
            r['cos_vec_G'] = cos(G.flatten(), Gm.flatten())
            # best pair under each G, for this real g
            def best_pair(M):
                ii, jj = torch.triu_indices(K, K, offset=1)
                A_, B_, C_ = M[ii, ii], M[jj, jj], M[ii, jj]
                Q = (B_ * g[ii] ** 2 - 2 * C_ * g[ii] * g[jj] + A_ * g[jj] ** 2) \
                    / (A_ * B_ - C_ ** 2).clamp(min=1e-30)
                b = int(Q.argmax())
                return sorted((int(ii[b]), int(jj[b]))), float(Q[b])
            bp_m, q_m = best_pair(G)
            bp_x, q_x = best_pair(Gm)
            top2 = sorted(torch.topk(g.abs(), 2).indices.tolist())

            def Q_joint(M, pair):
                """Q(S) = g_S' M_SS^-1 g_S for a 2-coordinate JOINT control.

                Not the sum of two single-coordinate Q's: with a non-diagonal M the
                cross term c is exactly what makes a joint control differ from two
                independent ones, so summing singles would compare different objects.
                """
                i, j = pair
                a, b, c = M[i, i], M[j, j], M[i, j]
                return float((b * g[i] ** 2 - 2 * c * g[i] * g[j] + a * g[j] ** 2)
                             / max(float(a * b - c ** 2), 1e-30))

            r['best_pair_matched'] = [S[i] for i in bp_m]
            r['best_pair_mismatched'] = [S[i] for i in bp_x]
            r['best_pair_G_is_I'] = [S[i] for i in top2]
            r['best_pair_agrees'] = bool(bp_m == bp_x)
            # same metric (matched G), two selection rules: maximise Q vs take two largest |g|
            r['Q_matched_at_matched_pair'] = q_m
            r['Q_matched_at_G_is_I_pair'] = Q_joint(G, top2)
            r['Q_gain_matched_rule_over_I_rule'] = (
                q_m / r['Q_matched_at_G_is_I_pair'] - 1 if r['Q_matched_at_G_is_I_pair'] else None)
            r['Q_mismatched_at_mismatched_pair'] = q_x
            r['Q_mismatched_at_matched_pair'] = Q_joint(Gm, bp_m)
        r['g'] = [float(x) for x in g]
        # split-half reliability of the matched G
        h = V.shape[0] // 2
        GA = (V[:h].T @ V[:h]) / h
        GB = (V[h:].T @ V[h:]) / (V.shape[0] - h)
        rA, rB = rho_of(GA)[off], rho_of(GB)[off]
        r['split_half_pearson_offdiag_rho'] = statistics.correlation(
            [float(x) for x in rA], [float(x) for x in rB])
        rr = rho_of(G)[off].abs()
        ev = torch.linalg.eigvalsh(G)
        r['median_abs_rho_offdiag'] = float(rr.median())
        r['condition_number'] = float(ev.max() / ev.clamp(min=1e-300).min())
        r['G'] = G.tolist()
        rec['per_context'][c['id']] = r

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / 'g_j2_space.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf8')

    lines = ['# G on J2\'s own output space vs the mismatched probe', '',
             f'- machine `{rec["machine"]}` · model `{rec["model"]}` · layers {layers} · |S| = {K} · m = {a.projections}',
             f'- readout: J2 = mean over families {sorted(fams)}; pool variant `{a.pool_variant}`; '
             f'lexicon sha256 `{rec["lexicon_sha256"][:16]}…`',
             f'- matched output space: ' + ', '.join(
                 f'{f} = {rec["shapes"][f]["n_seq"]} seqs × {rec["shapes"][f]["suffix_len"]} pos × 2560'
                 for f in sorted(fams)),
             f'- mismatched: last prompt position of 1 sequence (from `{a.compare_to or "n/a"}`)', '',
             '| context | cos(u*_match, u*_mism) | cos(u*_match, u*_I) | cos(u*_mism, u*_I) | '
             'r(off-diag G) | Spearman(diag G) | best pair agrees | split-half r | median |rho| | cond |',
             '|---|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for cid, r in rec['per_context'].items():
        f3 = lambda v: 'n/a' if v is None else f'{v:.3f}'
        # 'best_pair_agrees' only exists when a mismatched G was supplied for this context;
        # rendering an absent field as "no" would fabricate a disagreement.
        agree = r.get('best_pair_agrees')
        agree_s = 'n/a' if agree is None else ('yes' if agree else '**no**')
        lines.append(f'| `{cid}` | {f3(r.get("cos_u_matched_vs_mismatched"))} | '
                     f'{r["cos_u_matched_vs_I"]:.3f} | {f3(r.get("cos_u_mismatched_vs_I"))} | '
                     f'{f3(r.get("pearson_offdiag_G"))} | {f3(r.get("spearman_diag_G"))} | '
                     f'{agree_s} | '
                     f'{r["split_half_pearson_offdiag_rho"]:+.3f} | '
                     f'{r["median_abs_rho_offdiag"]:.3f} | {r["condition_number"]:.3g} |')
    lines += ['', '## best pair per context (same real g)', '',
              '| context | under matched G | under mismatched G | under G=I (two largest abs g) |',
              '|---|---|---|---|']
    for cid, r in rec['per_context'].items():
        if 'best_pair_matched' not in r:
            continue
        fmt = lambda p: ', '.join(f'L{l}N{n}' for l, n in p)
        lines.append(f'| `{cid}` | {fmt(r["best_pair_matched"])} | {fmt(r["best_pair_mismatched"])} | '
                     f'{fmt(r["best_pair_G_is_I"])} |')
    lines += ['', '## joint-pair Q under the matched metric, by selection rule', '',
              'Q(S) = g_S\' G_SS^-1 g_S for a 2-coordinate JOINT control (not the sum of two singles).', '',
              '| context | Q at matched-G pair | Q at G=I pair | gain of the matched rule |',
              '|---|---:|---:|---:|']
    for cid, r in rec['per_context'].items():
        if 'Q_matched_at_matched_pair' not in r:
            continue
        gain = r['Q_gain_matched_rule_over_I_rule']
        lines.append(f'| `{cid}` | {r["Q_matched_at_matched_pair"]:.4g} | '
                     f'{r["Q_matched_at_G_is_I_pair"]:.4g} | '
                     f'{"n/a" if gain is None else f"{gain:+.1%}"} |')
    (out / 'g_j2_space.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/g_j2_space.json and .md')


if __name__ == '__main__':
    main()
