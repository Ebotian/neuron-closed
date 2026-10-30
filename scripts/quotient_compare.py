#!/usr/bin/env python3
"""Plan 2.45: compare position lists in the QUOTIENT space, not on coordinates.

WHY. M = Gram(unit W_down columns) has rank 2560 per layer, so ker(W) is 6656-dimensional per layer:
position lists that differ by a kernel element are the SAME physical intervention. The committed
comparison said "directions reproducible 93.4%, exact overlap 1.7%". If that gap is representative
choice rather than disagreement, then mapping both lists through v = Wn u (R^2560 per layer) should
collapse it.

FROZEN CRITERION (written before the run)
  Q1  the quotient-space cosine between our K=64 list and the community 165-coordinate list must be
      HIGH (>= 0.5) for the hypothesis "same direction, different representative" to survive; the
      coordinate-space cosine is reported alongside and is expected to be low.
  Q2  the same comparison against a size-matched RANDOM list of 165 coordinates is the null: the
      quotient cosine must beat that null by a clear margin and the random null must be ~0.
  Q3  per-layer breakdown, because the two lists live in different layers (community L7/L22, ours L31/L5).
      If a layer is touched by only one side, the per-layer cosine is undefined and is reported as such.
"""
from __future__ import annotations
import csv, json, math, sys
import torch

sys.path.insert(0, '/root/autodl-tmp/scripts/j2space')
import engine as E


def read_list(path, dose_col):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            d = float(row[dose_col])
            if d != 0.0:
                out.append((int(row['layer']), int(row['neuron']), d))
    return out


def main():
    dev = 'cuda:0'
    tok, model, lm = E.load_model('/root/autodl-tmp/Qwen3.5-4B', dev)
    layers = list(range(32))
    ours = read_list('/root/autodl-tmp/quo/positions_K64.csv', 'dose_at_b0.05')
    ours256 = read_list('/root/autodl-tmp/quo/positions_all32_K1024_pairs.csv', 'u_dose')
    comm = read_list('/root/autodl-tmp/quo/community_pairs.csv', 'their_dose')
    print(f'lists: ours64={len(ours)} ours256={len(ours256)} community={len(comm)}', flush=True)

    Wn = {}
    for l in layers:
        W = lm.layers[l].mlp.down_proj.weight.detach().float().to(dev)
        Wn[l] = (W / W.norm(dim=0, keepdim=True).clamp(min=1e-12)).cpu()
        del W

    def to_v(items):
        """v_l = sum_n u_n * Wn_l[:,n]; returns dict layer -> R^2560 tensor (float64)."""
        v = {l: torch.zeros(Wn[l].shape[0], dtype=torch.float64) for l in layers}
        for (l, n, d) in items:
            v[l] += d * Wn[l][:, n].double()
        return v

    def cos_v(a, b):
        """Cosine in the quotient space, restricted to layers touched by BOTH sides."""
        num = den_a = den_b = 0.0
        per = {}
        for l in layers:
            na, nb = float(a[l].norm()), float(b[l].norm())
            if na == 0 or nb == 0:
                per[l] = None
                continue
            num += float(a[l] @ b[l])
            den_a += na ** 2
            den_b += nb ** 2
            per[l] = float(a[l] @ b[l]) / (na * nb)
        c = num / math.sqrt(max(den_a, 1e-30) * max(den_b, 1e-30)) if den_a > 0 and den_b > 0 else None
        return c, per

    def cos_coord(a, b):
        da = {}
        db = {}
        for (l, n, d) in a:
            da[(l, n)] = d
        for (l, n, d) in b:
            db[(l, n)] = d
        ks = set(da) | set(db)
        va = torch.tensor([da.get(k, 0.0) for k in sorted(ks)])
        vb = torch.tensor([db.get(k, 0.0) for k in sorted(ks)])
        if va.norm() == 0 or vb.norm() == 0:
            return None
        return float(torch.nn.functional.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)))

    # Q2 null: random lists of the same size and the same layer histogram
    import random
    random.seed(0)
    hist = {}
    for (l, n, d) in comm:
        hist[l] = hist.get(l, 0) + 1
    nulls = []
    for _ in range(20):
        rnd = []
        for l, k in hist.items():
            for n in random.sample(range(Wn[l].shape[1]), k):
                rnd.append((l, n, 1.0))
        c, _ = cos_v(to_v(ours), to_v(rnd))
        nulls.append(c)
    nulls_s = sorted(x for x in nulls if x is not None)

    def sign_only(items):
        """Direction test: community magnitudes are per-person behavioural tuning, not a designed
        weighting, so the fair 'is it the same direction' comparison uses signs only."""
        return [(l, n, 1.0 if d > 0 else -1.0) for (l, n, d) in items]

    res = {'sign_only': {}}
    for name, li in (('ours64', sign_only(ours)), ('all32_K1024', sign_only(ours256)),
                     ('comm_sign', sign_only(comm))):
        pass
    for name, li, ref in (('sign_ours64', sign_only(ours), sign_only(comm)),
                          ('sign_all32_K1024', sign_only(ours256), sign_only(comm)),
                          ('dose_ours64', ours, comm),
                          ('dose_all32_K1024', ours256, comm)):
        cq, per = cos_v(to_v(li), to_v(ref))
        res.setdefault(name, {})['cos_quotient'] = cq
        res[name]['cos_coord'] = cos_coord(li, ref)
        res[name]['per_layer_touched_both'] = {str(k): round(v, 4) for k, v in per.items() if v is not None}
        print(f'{name}: cos_quotient={cq} cos_coord={res[name]["cos_coord"]}', flush=True)
    for name, li in ():
        vo = to_v(li)
        vc = to_v(comm)
        cq, per = cos_v(vo, vc)
        pass
    res['null'] = {'n': len(nulls_s), 'median': nulls_s[len(nulls_s) // 2] if nulls_s else None,
                   'max': nulls_s[-1] if nulls_s else None}
    print('null (random 165 same layer histogram):', res['null'], flush=True)
    json.dump(res, open('/root/autodl-tmp/runs/quotient.json', 'w'), indent=1)
    print('wrote /root/autodl-tmp/runs/quotient.json', flush=True)


if __name__ == '__main__':
    main()
