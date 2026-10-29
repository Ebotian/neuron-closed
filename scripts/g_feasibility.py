#!/usr/bin/env python3
"""Feasibility probe: is there a NON-DIAGONAL control-cost matrix G, computable from the
frozen model, that makes the analytic min-energy control differ from "follow the gradient"?

Why this question. The closed form in the SIIM discussion is

    u* = delta * G^-1 g / (g' G^-1 g),      Q(S) = g_S' G_SS^-1 g_S

and every package so far runs it with G = I. With G = I the two shipped "solvers"
(`inverse.fixed_budget_control` -> b*g/||g||, `inverse.minimum_energy_control` ->
delta*g/(g'g)) are the SAME direction at different scale, and for a pair
Q({i,j}) = g_i^2 + g_j^2, so the cross term c vanishes and "analytic combination"
adds nothing beyond picking the two largest |g|. For a single coordinate
Q({i}) = g_i^2 / G_ii, so a DIAGONAL G changes only the RANKING; only a NON-DIAGONAL G
can change which coordinates should be combined. This script measures whether that
non-diagonal part exists, how big it is, and whether it survives a change of context.

Estimator. G is taken to be the quadratic cost "how much does the final hidden state at
the last position move", i.e. G = J'J with J = d h_final / d u in DOSE units. For
w ~ N(0, I_d), E[(J'w)(J'w)'] = J'J, so m random projections give an unbiased estimate
without ever forming J (which is 2560 x N). Gradients are summed over token positions
because an injection is broadcast to every position -- the same convention the frozen
`engine.capture_mlp_inputs` + `readout_gradients` path uses -- and scaled by
p50/||W_down[:,n]|| so a unit of u means one unit of dose, exactly as in steering.

CPU-side cost: none. GPU: C contexts x (1 forward + m backward) with one retained graph.

    [i/N] elapsed, eta ...   per backward (package progress.Progress)
"""
from __future__ import annotations
import argparse, hashlib, json, platform, random, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                     # qwentest/


def resolve_pkg(given):
    cands = [Path(given)] if given else []
    cands += [ROOT / 'vendor/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
              Path('/root/autodl-tmp/qwen-gradient-discovery-v6'),
              Path('/root/autodl-tmp/scripts/gfeas')]
    for c in cands:
        if (c / 'engine.py').is_file():
            return c
    raise SystemExit('package not found; pass --pkg <dir with engine.py>')


def contexts(path, limit, split):
    rows = [json.loads(l) for l in Path(path).read_text(encoding='utf8').splitlines() if l.strip()]
    rows = [r for r in rows if r.get('split') == split] if split else rows
    if not rows:
        raise SystemExit(f'no contexts in {path} (split={split})')
    return rows[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--layers', default='13-14')
    ap.add_argument('--k', type=int, default=32, help='coordinates in the candidate set S')
    ap.add_argument('--projections', type=int, default=192,
                    help='random w per context; must exceed k or G_hat is singular')
    ap.add_argument('--contexts', type=int, default=3)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--seed', type=int, default=20261028)
    ap.add_argument('--out', default='out/g_feasibility')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
    if a.projections <= a.k:
        raise SystemExit(f'--projections ({a.projections}) must exceed --k ({a.k}): '
                         f'G_hat would be rank-deficient and non-invertible')
    ctx = contexts(a.probes, a.contexts, a.split)
    rng = random.Random(a.seed)
    n_back = len(ctx) * a.projections

    print('Question : does a non-diagonal G = J\'J exist, and does it change the analytic control?')
    print(f'Layers   : {layers}   candidate set |S| = {a.k} (incl. KNOWN_EVIL coordinates)')
    print(f'Estimator: G_hat = mean over {a.projections} random w of (J\'w)(J\'w)\', dose units, '
          f'summed over token positions')
    print(f'Contexts : {len(ctx)} ({a.split}) -> {[c["id"] for c in ctx]}')
    print(f'Work     : {n_back} backward passes on one retained graph per context '
          f'(~0.4 s each) + model load')
    print(f'Need     : projections ({a.projections}) > k ({a.k})  [OK]')
    if a.dry_run:
        print('\n[dry-run] would report, per context:')
        print('  rho_ij = G_ij/sqrt(G_ii G_jj): median / p90 / max off-diagonal, '
              'same-layer vs cross-layer')
        print('  off-diagonal Frobenius share; condition number of G_hat')
        print('  verdict A: median cos(G_hat^-1 g, g) for realistic g (the J\'w samples)')
        print('  verdict B: for how many g does the best PAIR under G_hat differ from the '
              'two largest |g_i| (the G=I answer), and the Q gain')
        print('  verdict C: cross-context cos of vec(G_hat) -- is G context-stable?')
        print(f'[dry-run] would write {a.out}/g_feasibility.json and .md')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import (KNOWN_EVIL, capture_mlp_inputs, column_norms, estimate_p50,
                        load_model, prompt_prefix)
    from progress import Progress

    tok, model, lm = load_model(a.model)
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device='cuda:0')
    print('p50 (this machine):', {l: round(p50[l], 6) for l in layers})

    # candidate set S: the two known coordinates first, then seeded random ones
    per_layer = len(norms[layers[0]])
    coords = [(int(l), int(n)) for l, n, _ in KNOWN_EVIL if l in layers]
    seen = set(coords)
    while len(coords) < a.k:
        c = (rng.choice(layers), rng.randrange(per_layer))
        if c not in seen:
            seen.add(c); coords.append(c)
    coords = coords[:a.k]
    idx_by_layer = {l: [i for i, (ll, _) in enumerate(coords) if ll == l] for l in layers}
    print(f'S = {len(coords)} coordinates: '
          f'{sum(len(v) for v in idx_by_layer.values())} placed, '
          + ', '.join(f'L{l}({len(idx_by_layer[l])})' for l in layers))

    prog = Progress(n_back)
    per_ctx, done = {}, 0
    for c in ctx:
        ids = prompt_prefix(tok, c['prompt'])
        if len(ids) > a.max_seq:
            raise SystemExit(f'prompt too long: {c["id"]} ({len(ids)} > {a.max_seq})')
        inp = torch.tensor([ids], device='cuda:0', dtype=torch.long)
        with torch.enable_grad(), capture_mlp_inputs(lm, layers) as cap:
            out = model(input_ids=inp, use_cache=False, output_hidden_states=True)
            h = out.hidden_states[-1][0, -1, :].float()          # final state, last position
            d = h.numel()
            wrng = torch.Generator(device='cpu').manual_seed(
                a.seed + int(hashlib.sha256(c['id'].encode()).hexdigest()[:8], 16))
            V = []
            for _ in range(a.projections):
                w = torch.randn(d, generator=wrng, dtype=torch.float32).to(h.device)
                grads = torch.autograd.grad(torch.dot(h, w), [cap[l] for l in layers],
                                            retain_graph=True)
                v = torch.zeros(len(coords), dtype=torch.float64)
                for l, g in zip(layers, grads):
                    s = g.detach().double().sum(dim=tuple(range(g.ndim - 1)))   # broadcast inject
                    s = s * (p50[l] / norms[l].to(s.device).double().clamp(min=1e-12))
                    for i in idx_by_layer[l]:
                        v[i] = s[coords[i][1]]
                V.append(v.cpu())
                done += 1
                prog.step(f'{c["id"]} proj {len(V)}/{a.projections}')
        per_ctx[c['id']] = torch.stack(V)                        # m x K
        del out, h, grads
        torch.cuda.empty_cache()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    Gs = {k: (V.T @ V) / V.shape[0] for k, V in per_ctx.items()}   # unbiased for J'J

    def split_half(V):
        """Is the off-diagonal structure real, or estimator noise?

        G_hat from m projections in K dims is noisy (ratio m:K), and noise INFLATES
        off-diagonal rho while DEFLATING cos(G^-1 g, g). Estimating G from two disjoint
        halves and correlating their off-diagonal entries is the same control the G1/G2
        criteria use: noise does not replicate, structure does.
        """
        h = V.shape[0] // 2
        GA = (V[:h].T @ V[:h]) / h
        GB = (V[h:].T @ V[h:]) / (V.shape[0] - h)
        K = GA.shape[0]
        off = ~torch.eye(K, dtype=torch.bool)
        dA = torch.diagonal(GA).clamp(min=1e-30); dB = torch.diagonal(GB).clamp(min=1e-30)
        rA = (GA / torch.sqrt(dA[:, None] * dA[None, :]))[off]
        rB = (GB / torch.sqrt(dB[:, None] * dB[None, :]))[off]
        x, y = [float(t) for t in GA[off]], [float(t) for t in GB[off]]
        rx, ry = [float(t) for t in rA], [float(t) for t in rB]
        return {'pearson_offdiag_G': statistics.correlation(x, y),
                'pearson_offdiag_rho': statistics.correlation(rx, ry),
                'cos_vec_G': float((GA[off] @ GB[off]) / (GA[off].norm() * GB[off].norm())),
                'n_pairs': len(x)}

    def report(G, V):
        K = G.shape[0]
        dg = torch.diagonal(G).clamp(min=1e-30)
        rho = G / torch.sqrt(dg[:, None] * dg[None, :])
        off = ~torch.eye(K, dtype=torch.bool)
        same = torch.zeros(K, K, dtype=torch.bool)
        for i in range(K):
            for j in range(K):
                same[i, j] = coords[i][0] == coords[j][0]
        sm = off & same; cr = off & ~same
        def q(m):
            x = rho[m].abs().flatten()
            x = x[x > 0]
            if x.numel() == 0:
                return {'median': None, 'p90': None, 'max': None, 'n': 0}
            return {'median': float(x.median()), 'p90': float(x.quantile(.90)),
                    'max': float(x.max()), 'n': int(x.numel())}
        fro_off = float(G[off].pow(2).sum()); fro_all = float(G.pow(2).sum())
        ev = torch.linalg.eigvalsh(G)
        # verdict A: does G^-1 g point somewhere else than g?
        Gi = torch.linalg.pinv(G)
        cosA = []
        for g in V:
            u = Gi @ g
            cosA.append(float(g @ u / (g.norm() * u.norm() + 1e-30)))
        rg = torch.randn(V.shape[0], K, generator=torch.Generator().manual_seed(7),
                         dtype=torch.float64)
        cosR = []
        for g in rg:
            u = Gi @ g
            cosR.append(float(g @ u / (g.norm() * u.norm() + 1e-30)))
        # verdict B: best pair under G vs the G=I answer (two largest |g_i|).
        # Q({i,j}) = (b gi^2 - 2 c gi gj + a gj^2)/(ab - c^2) with [[a,c],[c,b]] = G_SS,
        # vectorised over all pairs.
        ii, jj = torch.triu_indices(K, K, offset=1)
        A_ = G[ii, ii]; B_ = G[jj, jj]; C_ = G[ii, jj]
        det = (A_ * B_ - C_ ** 2).clamp(min=1e-30)
        differ, gain = 0, []
        for g in V:
            gi, gj = g[ii], g[jj]
            Q = (B_ * gi ** 2 - 2 * C_ * gi * gj + A_ * gj ** 2) / det
            b = int(Q.argmax())
            best = sorted((int(ii[b]), int(jj[b])))
            top2 = sorted(torch.topk(g.abs(), 2).indices.tolist())
            qi = float(g[top2[0]] ** 2 / G[top2[0], top2[0]]) \
                + float(g[top2[1]] ** 2 / G[top2[1], top2[1]])
            differ += int(best != top2)
            gain.append((float(Q[b]) - qi) / abs(qi) if qi else float('nan'))
        return {'rho_off_diagonal': q(off), 'rho_same_layer': q(sm), 'rho_cross_layer': q(cr),
                'offdiag_frobenius_share': fro_off / fro_all,
                'condition_number': float(ev.max() / ev.clamp(min=1e-30).min()),
                'diag_min': float(dg.min()), 'diag_max': float(dg.max()),
                'verdictA_cos_realistic_median': statistics.median(cosA),
                'verdictA_cos_realistic_min': min(cosA),
                'verdictA_cos_random_median': statistics.median(cosR),
                'verdictB_pair_differs_fraction': differ / V.shape[0],
                'verdictB_Q_gain_median': statistics.median(gain),
                'verdictB_Q_gain_max': max(gain)}

    rec = {'schema': 'qwentest-g-feasibility-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
           'model': a.model, 'layers': layers, 'k': a.k, 'projections': a.projections,
           'p50': {str(l): p50[l] for l in layers}, 'seed': a.seed,
           'coordinates': [{'layer': l, 'neuron': n} for l, n in coords],
           'contexts': [c['id'] for c in ctx], 'per_context': {}}
    for cid, G in Gs.items():
        rec['per_context'][cid] = report(G, per_ctx[cid])
        rec['per_context'][cid]['split_half'] = split_half(per_ctx[cid])
        rec['per_context'][cid]['G'] = G.tolist()
    ids = list(Gs)
    if len(ids) > 1:
        cc = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                x, y = Gs[ids[i]].flatten(), Gs[ids[j]].flatten()
                cc[f'{ids[i]}|{ids[j]}'] = float(x @ y / (x.norm() * y.norm()))
        rec['verdictC_cross_context_cos'] = cc
    (out / 'g_feasibility.json').write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                            encoding='utf8')

    lines = ['# G = J\'J feasibility probe', '',
             f'- machine `{rec["machine"]}` / {rec["gpu"]} / torch {rec["torch"]}',
             f'- layers {layers}, |S| = {a.k}, {a.projections} projections, '
             f'{len(ctx)} contexts ({a.split})',
             f'- p50 ' + ', '.join(f'L{l}={p50[l]:.4f}' for l in layers), '',
             '| context | rho off-diag med | p90 | max | same-layer med | cross-layer med | '
             'off-diag Fro share | cond | A: cos(G^-1 g, g) med | B: pair differs | B: Q gain med | '
             'D: split-half r(off-diag) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cid, r in rec['per_context'].items():
        sh = r['split_half']
        f = lambda v: 'n/a' if v is None else f'{v:.3f}'
        lines.append(f'| `{cid}` | {f(r["rho_off_diagonal"]["median"])} | '
                     f'{f(r["rho_off_diagonal"]["p90"])} | {f(r["rho_off_diagonal"]["max"])} | '
                     f'{f(r["rho_same_layer"]["median"])} | {f(r["rho_cross_layer"]["median"])} | '
                     f'{r["offdiag_frobenius_share"]:.3f} | {r["condition_number"]:.3g} | '
                     f'{r["verdictA_cos_realistic_median"]:.3f} | '
                     f'{r["verdictB_pair_differs_fraction"]:.2f} | '
                     f'{r["verdictB_Q_gain_median"]:+.1%} | '
                     f'{sh["pearson_offdiag_rho"]:+.3f} |')
    if 'verdictC_cross_context_cos' in rec:
        lines += ['', 'cross-context cos(vec G_hat): '
                  + ', '.join(f'{k} = {v:.3f}' for k, v in rec['verdictC_cross_context_cos'].items())]
    (out / 'g_feasibility.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n' + '\n'.join(lines))
    print(f'\nwrote {out}/g_feasibility.json and .md')


if __name__ == '__main__':
    main()
