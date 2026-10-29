#!/usr/bin/env python3
"""The closed form over the WHOLE model: 32 layers x 9,216 = 294,912 coordinates, computed
(not measured), then intersected with the community's recipe library.

Scope, stated up front because it bounds every claim below. This script performs NO model
forward pass for the readout: it computes gradients (needed for the closed form), the budget
metric M, the closed-form solution, and the Q-greedy position order. It does NOT measure J2,
so nothing here says "this list is better" -- that requires the matched-budget readout
(plan 2.36), which is a different run. What it does say is geometric and checkable:
which positions the closed form ranks highest across the full coordinate space, what budget
share they take, and whether the community's documented coordinates are among them.

Why 32 layers was cheap after all. The solver never depended on the number of layers:
  * g for all 32 layers comes from ONE backward per context per family
    (capture_mlp_inputs hooks every requested layer), so the gradient cost is unchanged;
  * the budget metric is block-diagonal by layer BY DEFINITION (budget^2 = sum_l ||W_hat_l^T u_l||^2),
    so B_l = W_hat_l^T W_hat_l is 2560^2 = 52 MB and u_M_l = W_hat_l B_l^-1 d*_l is two matvecs;
  * Q(S) = g_S' M_SS^-1 g_S is therefore ADDITIVE across layers, which is what makes the
    global greedy tractable: at each step the marginal gain only needs the already-selected
    columns of the same layer, so the state is 32 tiny (P_l, y_l) pairs.

Frozen readings (written before the run; no effect claim is made in this script):
  A. position lists at K = 64, 256, 1024 over the 294,912-coordinate space, from the pooled g
     (one printable list) and per context (stability);
  B. layer histogram of our top-K positions, next to the community's layer histogram
     (community rows: 251 across 165 coordinates, from the public neuronary API snapshot
     2026-10-29, which lives OUTSIDE git);
  C. overlap with the community coordinates at each K, with the random baseline
     K x 165 / 294912 as the null, and the dose ratio u_ours/dose_theirs per shared coordinate.
  D. STABILITY: pairwise Jaccard of the per-context K=64 / 256 lists (same metric as plan 2.36).
  No p-values: there is no measurement here to put them on.

Dose units are the repo's: dose u_n displaces the residual stream by u_n * p50_l / ||W[:,n]||,
which is the SAME formula the community uses (10_reference section 2), so their documented
dose and our u_n are directly comparable (modulo the 6-12% p50 spread across quantisation).

GPU: 8 contexts x 2 families backwards + 32 Gram builds (~1-2 min), then CPU-side greedy.
"""
from __future__ import annotations
import argparse, hashlib, json, math, platform, statistics, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def resolve_pkg(given):
    cands = [Path(given)] if given else []
    cands += [ROOT / '03_unpacked/qwen-gradient-discovery-v6/qwen-gradient-discovery-v6',
              Path('/root/autodl-tmp/scripts/j2space'),
              Path('/root/autodl-tmp/qwen-gradient-discovery-v6')]
    for c in cands:
        if (c / 'engine.py').is_file() and (c / 'semantic.py').is_file():
            return c
    raise SystemExit('package not found; pass --pkg')


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


def greedy_global(Mblocks, g, dims, layers, K, torch, log_every=0):
    """Greedy max of Q(S) = g_S' M_SS^-1 g_S over the union, exploiting block structure.

    Mblocks[l] is the layer's 9,216 x 9,216 Gram (float32 on CPU); g[l] is float64.
    State per layer: S (selected local indices), P = M_SS^-1 (float64), y = P g_S.
    Returns (order [(layer, local_index)], Q prefix list, marginal gain per step).
    """
    S = {l: [] for l in layers}
    P = {l: torch.zeros(0, 0, dtype=torch.float64) for l in layers}
    y = {l: torch.zeros(0, dtype=torch.float64) for l in layers}
    diag = {l: torch.diagonal(Mblocks[l]).double() for l in layers}   # M_ii (=1 here, not assumed)
    ql = {l: 0.0 for l in layers}
    order, Qpre, marg = [], [], []
    for step in range(K):
        best = (-float('inf'), None)
        for l in layers:
            M, k = Mblocks[l], len(S[l])
            if k >= dims[l]:
                continue
            if k:
                v = M[:, S[l]].double()          # 9216 x k
                a = v @ y[l]
                b = ((v @ P[l]) * v).sum(1)
            else:
                v = None
                a = torch.zeros(dims[l], dtype=torch.float64)
                b = torch.zeros(dims[l], dtype=torch.float64)
            d = diag[l] - b
            num = g[l] - a
            gain = num * num / d.clamp(min=1e-12)
            gain[d <= 1e-12] = -float('inf')
            if k:
                gain[torch.tensor(S[l])] = -float('inf')
            j = int(torch.argmax(gain))
            gi = float(gain[j])
            if gi > best[0]:
                best = (gi, (l, j, v, num[j], float(d[j])))
        if best[1] is None:
            break
        gi, (l, j, v, res, dj) = best
        k = len(S[l])
        if k:
            # the bordering vector is M[S_l, j] (a k-vector), NOT M[:, S_l[j]]: the Schur
            # complement uses the new coordinate's inner products with the SELECTED set.
            vj = Mblocks[l][S[l], j].double()
            Pv = P[l] @ vj
            newP = torch.zeros(k + 1, k + 1, dtype=torch.float64)
            newP[:k, :k] = P[l] + torch.outer(Pv, Pv) / dj
            newP[:k, k] = -Pv / dj
            newP[k, :k] = -Pv / dj
            newP[k, k] = 1.0 / dj
            P[l] = newP
        else:
            P[l] = (1.0 / diag[l][j]).reshape(1, 1)
        S[l].append(j)
        y[l] = P[l] @ g[l][S[l]]
        ql[l] = float(g[l][S[l]] @ y[l])
        order.append((l, j))
        Qpre.append(sum(ql.values()))     # GLOBAL Q: additive over layer blocks
        marg.append(gi)
        if log_every and (step + 1) % log_every == 0:
            print(f'    greedy {step + 1}/{K}: Q={Qpre[-1]:.4f} layer spread '
                  f'{sorted({o[0] for o in order})[:6]}...', flush=True)
    return order, Qpre, marg, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkg', default='')
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--lexicon', default=str(ROOT / 'lexicons/jibes_adj_verb_pools.json'))
    ap.add_argument('--probes', default=str(ROOT / 'probes/contrasts_v3.jsonl'))
    ap.add_argument('--community', default='/home/ebt/neuronary-external/community_coords_all.json',
                    help='community coordinate snapshot (NOT in git): {coords:[{layer,neuron,doses}]}')
    ap.add_argument('--layers', default='0-31')
    ap.add_argument('--contexts', type=int, default=8)
    ap.add_argument('--split', default='heldout')
    ap.add_argument('--ks', default='64,256,1024')
    ap.add_argument('--ctx-kmax', type=int, default=256)
    ap.add_argument('--budget', type=float, default=0.05)
    ap.add_argument('--pool-variant', default='full', choices=['full', 'A', 'B'])
    ap.add_argument('--max-seq', type=int, default=384)
    ap.add_argument('--out', default='out/closed_form_all32')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    lo, hi = (int(x) for x in a.layers.split('-'))
    layers = list(range(lo, hi + 1))
    ks = [int(x) for x in a.ks.split(',')]
    kmax = max(ks)
    lex, fams = load_pools(a.lexicon, a.pool_variant)
    rows = [json.loads(l) for l in Path(a.probes).read_text(encoding='utf8').splitlines() if l.strip()]
    ctx = [r for r in rows if r.get('split') == a.split][:a.contexts]
    cids = [r['id'] for r in ctx]
    comm = json.loads(Path(a.community).read_text(encoding='utf8')) if Path(a.community).is_file() else None
    N = None

    print('Question : where does the closed form put its top positions in the WHOLE model,')
    print('           and does that intersect the community recipe library? (no readout measured)')
    print(f'Layers   : {layers}  ({len(layers)} x 9,216 coordinates)')
    print(f'Contexts : {len(cids)} -> {cids}')
    print(f'Greedy   : pooled K={kmax}; per context K={a.ctx_kmax}; ks reported {ks}')
    print(f'Community: {comm["n_coords"] if comm else "absent"} coordinates / '
          f'{comm["n_rows"] if comm else 0} rows' if comm else 'Community: absent')
    if a.dry_run:
        print('[dry-run] would write closed_form_all32.{json,md} and a top-K CSV')
        return

    import torch
    pkg = resolve_pkg(a.pkg)
    sys.path.insert(0, str(pkg))
    from engine import column_norms, estimate_p50, load_model, score_for_dose
    from semantic import sem_gradient

    tok, model, lm = load_model(a.model)
    norms = column_norms(lm, layers)
    p50 = estimate_p50(tok, model, lm, layers, device='cuda:0')
    dev = 'cuda:0'
    dims = {l: int(norms[l].numel()) for l in layers}
    N = sum(dims.values())
    print(f'p50: min {min(p50.values()):.3f} max {max(p50.values()):.3f}')

    # ---- g for every layer, one backward per context per family --------------------------
    G, gnorm = {}, {}
    for c in ctx:
        gsum = None
        for fam, f in sorted(fams.items()):
            _j2, gg = sem_gradient(tok, model, lm, layers, c['prompt'], f['template'], f, dev,
                                   a.max_seq)
            sc = {l: score_for_dose(gg[l].float(), p50[l], norms[l].to(gg[l].device))
                  for l in layers}
            gsum = sc if gsum is None else {l: gsum[l] + sc[l] for l in layers}
        G[c['id']] = {l: (gsum[l].double() / len(fams)).cpu() for l in layers}
        gnorm[c['id']] = {l: float(G[c['id']][l].norm()) for l in layers}
    g_pool = {l: torch.stack([G[c][l] for c in cids]).mean(0) for l in layers}
    print('g done; top-3 layers by pooled ||g||: '
          + str(sorted(((round(float(g_pool[l].norm()), 3), l) for l in layers), reverse=True)[:3]))

    # ---- Gram blocks: build on GPU one layer at a time, keep float32 on CPU ---------------
    Mblocks = {}
    for l in layers:
        Wt = lm.layers[l].mlp.down_proj.weight.detach().float().T.to(dev)
        Wt = Wt / Wt.norm(dim=1, keepdim=True).clamp(min=1e-12)
        Mblocks[l] = (Wt @ Wt.T).cpu()
        del Wt
        torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    print(f'M: {len(layers)} blocks of {dims[layers[0]]}^2 float32 '
          f'({sum(m.numel() * 4 for m in Mblocks.values()) / 1e9:.1f} GB, CPU)')

    # ---- greedy: pooled (printable list) and per context (stability) ---------------------
    # CPU greedy does many TINY matmuls (9,216 x k) @ (k x k); with the default thread pool the
    # per-call sync overhead dominates (~2 s/step measured). One thread is ~40x faster here and
    # the GPU work above is unaffected.
    torch.set_num_threads(int(__import__('os').environ.get('GREEDY_THREADS', '1')))
    print(f'CPU greedy threads: {torch.get_num_threads()}')
    ord_pool, Q_pool, mg_pool, S_pool = greedy_global(Mblocks, g_pool, dims, layers, kmax, torch,
                                                      log_every=64)
    ords_ctx = {}
    for c in cids:
        ords_ctx[c], Qc, _mg, _S = greedy_global(Mblocks, G[c], dims, layers, a.ctx_kmax, torch)
    print(f'greedy done: pooled {len(ord_pool)} positions over layers '
          f'{len({l for l, _ in ord_pool})}; per-context {a.ctx_kmax} each')

    def doses(order, g, K, budget):
        """Closed-form dose on the selected support, scaled to `budget` (exact: sqrt(g_S'u_S))."""
        by_layer = {}
        for l, j in order[:K]:
            by_layer.setdefault(l, []).append(j)
        u = {}
        for l, js in by_layer.items():
            idx = torch.tensor(js)
            Mss = Mblocks[l][idx][:, idx].double()
            u[l] = torch.linalg.solve(Mss, g[l][idx])
        b0 = math.sqrt(sum(float(g[l][torch.tensor(js)] @ u[l]) for l, js in by_layer.items()))
        s = budget / b0 if b0 > 0 else 0.0
        return {l: u[l] * s for l in u}, s, by_layer

    rec = {'schema': 'qwentest-closed-form-all32-v1', 'machine': platform.node(),
           'gpu': torch.cuda.get_device_name(0), 'model': a.model, 'layers': layers,
           'coordinate_space': N, 'dim_per_layer': dims, 'contexts': cids, 'ks': ks,
           'budget_for_doses': a.budget, 'pool_variant': a.pool_variant,
           'lexicon_sha256': hashlib.sha256(Path(a.lexicon).read_bytes()).hexdigest(),
           'p50': {str(l): p50[l] for l in layers},
           'g_norm_by_layer': {c: gnorm[c] for c in cids},
           'scope': 'no readout measured in this run: geometry and external comparison only',
           'frozen_readings': {
               'positions': 'top-K of the pooled-greedy order over 32 layers',
               'community_overlap': 'count and ranks of community coordinates inside top-K; '
                                    'null = K x n_community_coords / N',
               'dose_ratio': 'u_ours(budget) / their documented dose, per shared coordinate',
               'stability': 'pairwise Jaccard of the per-context lists at each K',
               'no_p_values': 'no measurement is in this run, so no effect claim is made'},
           'top': {}, 'community': None, 'stability': {}}

    for K in ks:
        u, s, by_layer = doses(ord_pool, g_pool, K, a.budget)
        sel = ord_pool[:K]
        shares = []
        for i, (l, j) in enumerate(sel):
            prev = Q_pool[i - 1] if i else 0.0
            shares.append((Q_pool[i] - prev) / Q_pool[K - 1] if Q_pool[K - 1] else 0.0)
        hist = {str(l): sum(1 for (ll, _) in sel if ll == l) for l in layers}
        rec['top'][str(K)] = {
            'Q_total': Q_pool[K - 1], 'scale': s,
            'layer_histogram': {k: v for k, v in hist.items() if v},
            'positions': [{'rank': i + 1, 'layer': l, 'neuron': j,
                           'sign': '+' if float(u[l][by_layer[l].index(j)]) > 0 else '-',
                           'dose': float(u[l][by_layer[l].index(j)]),
                           'q_share': shares[i]} for i, (l, j) in enumerate(sel)]}

    if comm:
        ccoords = [(int(x['layer']), int(x['neuron'])) for x in comm['coords']]
        cdose = {(int(x['layer']), int(x['neuron'])): statistics.median(x['doses'])
                 for x in comm['coords']}
        cmap = {c: i for i, c in enumerate(ccoords)}
        out = {'n_community_coords': len(ccoords), 'n_community_rows': comm['n_rows'],
               'null_expected': {}, 'hits': {}}
        for K in ks:
            sel = ord_pool[:K]
            rank = {}
            for i, c in enumerate(sel):
                if c in cmap:
                    rank[c] = i + 1
            u, s, by_layer = doses(ord_pool, g_pool, K, a.budget)
            hits = []
            for c, r in sorted(rank.items(), key=lambda kv: kv[1]):
                l, j = c
                uu = float(u[l][by_layer[l].index(j)])
                hits.append({'layer': l, 'neuron': j, 'rank': r,
                             'q_share': rec['top'][str(K)]['positions'][r - 1]['q_share'],
                             'dose_ours': uu, 'dose_theirs': cdose[c],
                             'ratio': uu / cdose[c] if cdose[c] else None})
            out['hits'][str(K)] = hits
            out['null_expected'][str(K)] = K * len(ccoords) / N
        c_layer = {}
        for c in ccoords:
            c_layer[c[0]] = c_layer.get(c[0], 0) + 1
        out['community_layer_histogram'] = {str(k): v for k, v in sorted(c_layer.items())}
        out['shared_coordinates_in_any_K'] = sorted({(h['layer'], h['neuron'])
                                                     for K in out['hits'].values() for h in K})
        rec['community'] = out

    for K in ks:
        vals = []
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                A = set(ords_ctx[cids[i]][:K]); B = set(ords_ctx[cids[j]][:K])
                vals.append(len(A & B) / max(1, len(A | B)))
        rec['stability'][str(K)] = {'median_pairwise_jaccard': statistics.median(vals),
                                    'min': min(vals), 'max': max(vals)}

    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'closed_form_all32.json').write_text(
        json.dumps(rec, ensure_ascii=False, indent=1), encoding='utf8')
    with open(out_dir / f'positions_all32_K{kmax}.csv', 'w') as fh:
        fh.write('rank,layer,neuron,sign,dose_at_b%g,q_share\n' % a.budget)
        for p in rec['top'][str(kmax)]['positions']:
            fh.write(f'{p["rank"]},{p["layer"]},{p["neuron"]},{p["sign"]},{p["dose"]:.8f},'
                     f'{p["q_share"]:.6f}\n')

    md = ['# The closed form over the whole model (32 layers, 294,912 coordinates)', '',
          f'- machine `{rec["machine"]}` / {rec["gpu"]} · model `{a.model}` · **no readout measured**',
          f'- contexts {len(cids)}: {", ".join(cids)}',
          f'- selection: greedy max of `Q(S)=g_SᵀM_SS⁻¹g_S`, M block-diagonal by layer '
          f'(exact budget metric); M_ii = 1 ⇒ Q is additive across layers',
          f'- doses normalised to `b={a.budget}` on the reported support', '',
          '## layer histogram of the top-K positions', '',
          '| K | layers used | top layers (count) | median Q share |',
          '|---:|---:|---|---:|']
    for K in ks:
        t = rec['top'][str(K)]
        h = sorted(t['layer_histogram'].items(), key=lambda kv: -kv[1])
        qs = statistics.median(p['q_share'] for p in t['positions'])
        md.append(f'| {K} | {len(t["layer_histogram"])} | ' +
                  ', '.join(f'L{k}:{v}' for k, v in h[:6]) + f' | {qs:.4f} |')
    md += ['', '## stability of the per-context lists', '',
           '| K | median pairwise Jaccard | min | max |', '|---:|---:|---:|---:|']
    for K in ks:
        s = rec['stability'][str(K)]
        md.append(f'| {K} | {s["median_pairwise_jaccard"]:.3f} | {s["min"]:.3f} | {s["max"]:.3f} |')
    if rec['community']:
        c = rec['community']
        md += ['', '## the community library (public API snapshot, outside git)', '',
               f'- {c["n_community_coords"]} unique coordinates / {c["n_community_rows"]} neuron-dose rows',
               f'- their layer histogram (top 6): ' + ', '.join(
                   f'L{k}:{v}' for k, v in sorted(c['community_layer_histogram'].items(),
                                                  key=lambda kv: -kv[1])[:6]), '',
               '| K | community coordinates found | random null | dose ratio (ours/theirs) |',
               '|---:|---:|---:|---|']
        for K in ks:
            hits = c['hits'][str(K)]
            ratios = [h['ratio'] for h in hits if h['ratio']]
            md.append(f'| {K} | **{len(hits)}** | {c["null_expected"][str(K)]:.2f} | ' +
                      (', '.join(f'{r:.4f}' for r in ratios) if ratios else '—') + ' |')
        md += ['', '| K | layer | neuron | our rank | our Q share | our dose | their dose | ratio |',
               '|---:|---:|---:|---:|---:|---:|---:|---:|']
        for K in ks:
            for h in c['hits'][str(K)]:
                md.append(f'| {K} | {h["layer"]} | {h["neuron"]} | {h["rank"]} | '
                          f'{h["q_share"]:.4f} | {h["dose_ours"]:+.5f} | {h["dose_theirs"]:+.2f} | '
                          f'{h["ratio"]:.4f} |')
    (out_dir / 'closed_form_all32.md').write_text('\n'.join(md) + '\n', encoding='utf8')
    print('\n'.join(md))
    print(f'\nwrote {out_dir}/closed_form_all32.json/.md and positions_all32_K{kmax}.csv')


if __name__ == '__main__':
    main()
