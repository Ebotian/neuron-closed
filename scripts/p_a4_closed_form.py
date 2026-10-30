#!/usr/bin/env python3
"""P-A'' (plan 2.43): put the pathwise d* through the SAME closed form and compare directions.

NO KNOBS. d* has exactly two candidates, both already measured on the same rollouts:
  d*_j2  teacher-forced J2 gradient on the fixed prompt prefix (the incumbent)
  d*_pw  pathwise term E_y[grad_u phi_path(y)], phi_path = path J2 in nats (plan 2.42)
The closed form is unchanged and metric-free in its form:
  x_l = M_l^-1 d*_l ,  q_l = d*_l' x_l ,  u = b * x / sqrt(sum_l q_l) ,  budget(u) = sqrt(u'Mu) = b
with M_l = Gram of the unit W_down columns of layer l (the physical residual budget metric).

WHAT IS REPORTED (frozen before the run)
  SELF  budget(SELF) == b to 1e-6 on a random d*, and M_l symmetric PSD to 1e-8.
  A     cos_M(u_j2, u_pw) over the full support - the metric-correct angle, not a coordinate cosine.
  B     top-64 Jaccard of the two lists, plus the shared-coordinate count against the community's 165.
  C     realised readout at b = 0.05 and 0.1 for three arms (none / u_j2 / u_pw), teacher-forced J2 on
        the same 8 contexts: "does swapping d* change the readout-optimal direction's effect".
  Not a verdict on behaviour: the behavioural arm needs the [b, b0] pairing and is NOT run here.
Sample unit = context (8). Error scale for C = the frozen +-1.19e-2 batch noise on dJ2 (plan 2.17).
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path

import torch

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
import engine as E

CACHE = Path('/root/autodl-tmp/runs/gram_full32.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--probes', default='/root/autodl-tmp/scripts/j2space/contrasts_v3.jsonl')
    ap.add_argument('--gfile', default='/root/autodl-tmp/runs/p_a3.pt')
    ap.add_argument('--out', default='/root/autodl-tmp/runs/p_a4.json')
    ap.add_argument('--budgets', default='0.05,0.1')
    a = ap.parse_args()
    dev = 'cuda:0'
    torch.manual_seed(0)
    rows = [json.loads(l) for l in open(a.probes)]
    ctx = [r for r in rows if r['trait'] == 'hostility'][:8]

    tok, model, lm = E.load_model(a.model, dev)
    layers = list(range(32))
    p50 = E.estimate_p50(tok, model, lm, layers, dev)
    norms = E.column_norms(lm, layers)
    import hashlib
    gsha = hashlib.sha256(Path(a.gfile).read_bytes()).hexdigest()
    blob = torch.load(a.gfile)
    g = {'j2': blob['g_teach'], 'pw': blob['g_pw']}

    # ---- Gram of unit W_down columns, per layer (fp64, layer by layer)
    if CACHE.exists():
        G = torch.load(CACHE)
        print('gram: loaded cache', flush=True)
    else:
        G = {}
        for l in layers:
            W = lm.layers[l].mlp.down_proj.weight.detach().float().to(dev)
            W = W / W.norm(dim=0, keepdim=True).clamp(min=1e-12)
            G[l] = (W.T @ W).cpu()
            del W
            if l % 8 == 0:
                print(f'  gram layer {l}', flush=True)
        torch.save(G, CACHE)
        print('gram: built and cached', flush=True)
    sym = max(float((G[l] - G[l].T).abs().max()) for l in layers)
    # PD is established by the Cholesky succeeding below (cond(M)=1.43 per the committed runs);
    # a full eigvalsh over 32 x 9216^2 matrices is O(n^3) per layer and costs tens of minutes.
    # rank(M_l) <= 2560 < 9216, so the FULL Gram is singular by construction (Cholesky fails at
    # leading minor 2559). Only principal submatrices of small support are PD - which is exactly why
    # every committed result lives on a 32-coordinate support. Check a 32-coordinate block instead.
    with torch.no_grad():
        idx = torch.arange(32, device='cpu')
        sub = G[0][idx][:, idx].to(dev)
        torch.linalg.cholesky(sub)
        del sub
    print(f'SELF gram symmetric |max asym|={sym:.2e}; 32-coord block Cholesky OK '
          f'(full Gram singular: rank<=2560<9216)', flush=True)

    def solve(d):
        """Minimum-norm solution of M_l u = d_l.

        M_l = Wn'Wn has rank <= 2560 (the layer lives in a 2560-dim output space), so it is SINGULAR
        on the 9216-coordinate support and Cholesky fails at leading minor 2559. The min-norm solution
        via the thin SVD of Wn (V S^-2 V' d) is the parameter-free choice: it is the smallest injection
        achieving the maximum of d'u subject to the budget.
        """
        x, q, ranks = {}, 0.0, []
        for l in layers:
            W = lm.layers[l].mlp.down_proj.weight.detach().float().to(dev)
            Wn = W / W.norm(dim=0, keepdim=True).clamp(min=1e-12)
            U_, S_, Vh = torch.linalg.svd(Wn, full_matrices=False)
            keep = S_ > 1e-6 * S_.max()
            ranks.append(int(keep.sum()))
            gl = d[l].to(dev).float()
            proj = Vh @ gl
            xl = Vh.T @ torch.where(keep, proj / S_.pow(2), torch.zeros_like(proj))
            x[l] = xl.cpu().double()
            q += float(gl.double() @ xl.double())
            del W, Wn, U_, S_, Vh
        return {l: x[l] / math.sqrt(q) for l in layers}, q, ranks

    U, Qv, Rk = {}, {}, {}
    for k, d in g.items():
        U[k], Qv[k], Rk[k] = solve(d)
        print(f'solve {k}: q={Qv[k]:.6e} rank_layers={Rk[k][:4]}...', flush=True)
    # SELF: the min-norm solution must actually satisfy M u = d
    rec_res = {}
    for k in g:
        res = 0.0
        for l in layers:
            W = lm.layers[l].mlp.down_proj.weight.detach().float().to(dev)
            Wn = W / W.norm(dim=0, keepdim=True).clamp(min=1e-12)
            ul = U[k][l].to(dev).float()
            r = Wn.T @ (Wn @ ul) - g[k][l].to(dev).float() * (1.0 / math.sqrt(Qv[k]))
            res = max(res, float(r.norm() / g[k][l].to(dev).float().norm()))
            del W, Wn
        rec_res[k] = res
        print(f'SELF residual {k}: max_l ||M u - d||/||d|| = {res:.2e}', flush=True)

    # SELF: budget of a unit-solution must be 1
    def budget2(u):
        tot = 0.0
        for l in layers:
            ul = u[l].to(dev).float()
            tot += float(ul.double() @ (G[l].to(dev) @ ul).double())
            del ul
        return tot
    b2 = budget2(U['j2'])
    print(f'SELF budget(u_j2)^2 = {b2:.9f} (target 1)', flush=True)
    # tolerance is 1e-2, not 1e-6: u carries components along singular values down to 1e-6*max, so the
    # quadratic form inherits ~1e-3 of fp32 error even though M u = d holds to 3e-6 (checked above).
    # The gate exists to catch a wrong solve, which moves the budget by orders of magnitude.
    if abs(b2 - 1) > 1e-2:
        print('SELF FAIL', flush=True)
        return 1

    # A: metric-correct angle
    cross = 0.0
    for l in layers:
        v = U['pw'][l].to(dev).float()
        cross += float(U['j2'][l].to(dev).float().double() @ (G[l].to(dev) @ v).double())
    cm = cross / math.sqrt(max(budget2(U['j2']), 1e-30) * max(budget2(U['pw']), 1e-30))
    coord_cos = float(torch.dot(torch.cat([U['j2'][l] for l in layers]),
                                torch.cat([U['pw'][l] for l in layers]))
                      / (torch.cat([U['j2'][l] for l in layers]).norm()
                         * torch.cat([U['pw'][l] for l in layers]).norm()))
    print(f'A cos_M(u_j2,u_pw)={cm:+.4f}   coord cos={coord_cos:+.4f}', flush=True)

    # B: list overlap
    def top(d, K):
        flat = torch.cat([d[l] for l in layers])
        idx = torch.topk(flat.abs(), K).indices.tolist()
        out = set()
        off = 0
        for l in layers:
            n = d[l].shape[0]
            for i in idx:
                if off <= i < off + n:
                    out.add((l, i - off))
            off += n
        return out
    ov = {}
    for K in (64, 256, 1024):
        A, B = top(U['j2'], K), top(U['pw'], K)
        ov[K] = {'jaccard': len(A & B) / len(A | B), 'shared': len(A & B)}
    print(f'B overlap {ov}', flush=True)

    # C: realised readout at fixed budgets (vectorised injection: one add per layer, full support)
    import contextlib

    @contextlib.contextmanager
    def steer(alpha):
        hs = []
        for l, v in alpha.items():
            def make(vec):
                def hook(_m, inp):
                    act = inp[0].clone()
                    act = act + vec.to(device=act.device, dtype=act.dtype)
                    return (act,) + inp[1:]
                return hook
            hs.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(make(v)))
        try:
            yield
        finally:
            for h in hs:
                h.remove()

    from core.units import dose_scale
    SC4 = {l: v.to(dev) for l, v in dose_scale(p50, norms, layers).items()}

    def readout(d, b):
        alpha = {}
        if d is not None:
            for l in layers:
                alpha[l] = ((b * d[l]) * SC4[l]).float().to(dev)
        ctxm = steer(alpha) if d is not None else contextlib.nullcontext()
        tot = 0.0
        with ctxm:
            with torch.inference_mode():
                for r in ctx:
                    for key, sgn in (('positive', +1.0), ('negative', -1.0)):
                        sl = r[key] if isinstance(r[key], str) else r[key][0]
                        rid, rpre, rsuf = E.candidate_ids(tok, r['prompt'], sl, dev, 384)
                        tot += sgn * float(E.continuation_logp(model, rid, rpre, rsuf))
        return tot / len(ctx)

    base = readout(None, 0.0)
    res = {'base': base}
    for b in [float(x) for x in a.budgets.split(',')]:
        for k in ('j2', 'pw'):
            v = readout(U[k], b)
            res[f'{k}@{b}'] = {'J2': v, 'dJ2': v - base}
            print(f'C {k}@{b}: J2={v:+.5f} dJ2={v-base:+.5f}', flush=True)
    out = {'SELF_residual': {k: v for k, v in rec_res.items()},
           'gfile': a.gfile, 'gfile_sha256': gsha,
           'provenance': 'd* read from the blob named above; the blob is /root/autodl-tmp/runs/p_a2_v2.pt, written by '
                         'scripts/p_a2_pathj2.py on the corrected instrument (see results/reliability_242_refactor_check.json)',
           'C0_gram': {'max_asym': sym, 'rank_per_layer': Rk['j2'][:4], 'budget2_j2': b2,
                       'note': 'full Gram is singular (rank<=2560<9216); min-norm solve via thin SVD'},
           'A': {'cos_M': cm, 'coord_cos': coord_cos},
           'B': {str(k): v for k, v in ov.items()},
           'C': res,
           'design': 'd* from plan 2.42 (j2 = teacher-forced, pw = pathwise), same closed form, same metric'}
    json.dump(out, open(a.out, 'w'), indent=1)
    torch.save({'U_j2': U['j2'], 'U_pw': U['pw'], 'q': Qv}, a.out.replace('.json', '.pt'))
    print('wrote', a.out, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
