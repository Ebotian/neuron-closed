"""Assembling the report dict - the ONE place the summary statistics are computed from gradients.

Keeping this out of the experiment scripts means the four thin shells cannot drift into computing the same summary
three different ways (which is how the plug-in cosine and the ICC once disagreed by 10x without anyone noticing).
"""
from __future__ import annotations

import torch

from analysis.split_half import split_half_curve
from analysis.icc import icc_from_gram_py


def _flat(d, layers):
    return torch.cat([d[l] for l in layers])


def summarize(layers, g_path, g_teach, g_path_in, S_all, W_all, rollouts, seed, phi_all, pathwise=None):
    flat = lambda d: _flat(d, layers) if d is not None else None
    Gt = torch.cat(S_all, 0)
    Wf = torch.tensor([w for ws in W_all for w in ws], device=Gt.device)
    # The Gram is (256 x 294,912) @ (294,912 x 256). In float64 on a consumer GPU that matmul runs at 1/64 of the
    # float32 rate and showed up in the sampler as a multi-second stretch at 0-29% utilisation. float32 for the big
    # product, float64 only for the 256x256 result, keeps the algebra in the same place and the GPU busy.
    Mw = Gt.float() * Wf.float().unsqueeze(1)
    gram = (Mw @ Mw.T).double().cpu().tolist()
    sizes = [int(S.shape[0]) for S in S_all]
    icc = icc_from_gram_py(gram, sizes, k=rollouts)
    mu = sum(phi_all) / len(phi_all)
    gp, gt = flat(g_path), flat(g_teach)
    rec = {'seed': seed, 'gram': gram, 'group_sizes': sizes,
           'C0': {'phi_mean': mu, 'phi_sd_total': (sum((x - mu) ** 2 for x in phi_all) / len(phi_all)) ** 0.5,
                  'n_rollouts': len(phi_all)},
           'R': {'r_of_K': split_half_curve(S_all, W_all, ks=(2, 4, 8, 16, 32), n_split=200, seed=0,
                                            device=Gt.device),
                 'convention': 'r(K) ~ K/(K+2*kappa); K(r=0.5) = 2*kappa'},
           'ICC': icc,
           }
    if gp is not None and gt is not None:
        rec['C1p'] = {'cos_path_teach': float(torch.nn.functional.cosine_similarity(gp.unsqueeze(0), gt.unsqueeze(0)))}
    if gp is not None and g_path_in is not None:
        rec['baseline_compare'] = {'cos_loo_vs_insample': float(torch.nn.functional.cosine_similarity(
            gp.unsqueeze(0), flat(g_path_in).unsqueeze(0))), 'K': rollouts}
    if pathwise is not None:
        gw = flat(pathwise)
        rec['C2_split'] = {'cos_pathwise_teach': float(torch.nn.functional.cosine_similarity(gw.unsqueeze(0), gt.unsqueeze(0))),
                           'cos_pathwise_score': float(torch.nn.functional.cosine_similarity(gw.unsqueeze(0), gp.unsqueeze(0))),
                           'norm_pathwise': float(gw.norm()), 'norm_score': float(gp.norm()),
                           'ratio_score_over_pathwise': float(gp.norm() / gw.norm())}
    return rec
