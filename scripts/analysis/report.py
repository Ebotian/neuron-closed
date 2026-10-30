"""Assembling the report dict - the ONE place the summary statistics are computed from gradients.

Keeping this out of the experiment scripts means the four thin shells cannot drift into computing the same summary
three different ways (which is how the plug-in cosine and the ICC once disagreed by 10x without anyone noticing).
"""
from __future__ import annotations

import torch

from analysis.split_half import split_half_curve
from analysis.icc import icc_from_gram_py, cross_replicate


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
        gw = flat(pathwise)   # A-hat: the direct-derivative term E[d phi_u / du] along the sampled trajectories
        rec['C2_split'] = {'cos_pathwise_teach': float(torch.nn.functional.cosine_similarity(gw.unsqueeze(0), gt.unsqueeze(0))),
                           'cos_pathwise_score': float(torch.nn.functional.cosine_similarity(gw.unsqueeze(0), gp.unsqueeze(0))),
                           'norm_pathwise': float(gw.norm()), 'norm_score': float(gp.norm()),
                           'ratio_score_over_pathwise': float(gp.norm() / gw.norm())}
        # THE DECOMPOSITION THE AUDIT ASKED FOR.  dT/du = E[d phi_u/du] + E[(phi_u - b) grad log p_u].
        # `gw` estimates the first term, `gp` (score x LOO-baselined reward) the second, and the teacher-forced
        # gradient is an estimator of the FIRST term only. So `cos_pathwise_teach` is an A-vs-A comparison and no
        # cosine of the full gradient with the teacher-forced direction can be formed without adding the two.
        tot = gw + gp
        cos = lambda x, y: float(torch.nn.functional.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)))
        rec['C4_decomposition'] = {
            'cos_A_teacher': cos(gw, gt), 'cos_B_teacher': cos(gp, gt), 'cos_A_plus_B_teacher': cos(tot, gt),
            'cos_A_B': cos(gw, gp),
            'norms': {'A': float(gw.norm()), 'B': float(gp.norm()), 'A_plus_B': float(tot.norm()), 'teacher': float(gt.norm())},
            'warning': ('the teacher-forced gradient estimates term A only; cos(A+B, teacher) is therefore NOT the accuracy '
                        'of a behaviour-gradient surrogate, and neither cosine shows that the later layers are well proxied')}
        # INDEPENDENT-REPLICATE SIGNAL: split each context's rollouts into disjoint halves and form each half's estimate
        # with its OWN leave-one-out baseline inside the half. E[<gA,gB>] = ||g||^2 and 0.5 E||gA-gB||^2 = tr Cov(ghat),
        # which is the only route to a signal-to-noise on a common scale; `between_var` cannot supply it.
        halves = _half_estimates(S_all, W_all, layers, dims=Gt.shape[1], device=Gt.device, rollouts=rollouts)
        if halves is not None:
            rec['SNR_split'] = cross_replicate(halves[0], halves[1])
    return rec


def print_headline(rec):
    """One place for the numbers a run prints, so the shell keeps no formatting logic of its own."""
    i = rec['ICC']
    print('ICC  MSw=%.6f between_var=%.6f icc=%.6f %s'
          % (i['tr_sigma'], i['between_var'], i['icc'], [round(x, 4) for x in i['icc_interval']]), flush=True)
    if rec.get('SNR_split'):
        v = rec['SNR_split']
        print('SNR  signal_sq=%.2f noise=%.2f snr=%.4f' % (v['signal_sq'], v['noise_trace'], v['snr']), flush=True)
    if rec.get('C2_split'):
        c = rec['C2_split']
        print('C2 cos(pw,teach)=%+.4f |pw|=%.2f ratio=%.3f'
              % (c['cos_pathwise_teach'], c['norm_pathwise'], c['ratio_score_over_pathwise']), flush=True)
    if rec.get('C4_decomposition'):
        d = rec['C4_decomposition']
        print('C4 cos(A,B)=%+.4f A-teach=%+.4f B-teach=%+.4f A+B-teach=%+.4f'
              % (d['cos_A_B'], d['cos_A_teacher'], d['cos_B_teacher'], d['cos_A_plus_B_teacher']), flush=True)


def save_gradients(path, rec, g_teach, g_pw, g_path, layers, manifest):
    """Write the gradient blob that p_a4 consumes, with a manifest naming the run it came from.

    The audit's point was that `p_a2_v2.pt` had no producer in the live chain, so `gfile_sha256` verified the identity of an
    input nobody could rebuild. The blob now carries its own provenance instead of relying on the caller remembering it.
    """
    import torch
    torch.save({'g_teach': g_teach, 'g_pw': g_pw, 'g_score': g_path, 'layers': layers, 'manifest': manifest}, path)
    return {'path': str(path), 'manifest': manifest}


def _half_estimates(S_all, W_all, layers, dims, device, rollouts):
    """Rebuild the score-term estimate from two DISJOINT halves of each context's rollouts.

    The weights are recomputed inside each half (leave-one-out over that half's rollouts only), because a split taken
    after the full-group weights exist is not two independent estimators - the audit's point about split-half.
    """
    if rollouts < 4 or not S_all:
        return None
    tot = [torch.zeros(dims, dtype=torch.float64, device=device) for _ in range(2)]
    for S, W in zip(S_all, W_all):
        half = S.shape[0] // 2
        for h in (0, 1):
            blk = S[h * half:(h + 1) * half].double()
            w = torch.tensor(W[h * half:(h + 1) * half], dtype=torch.float64, device=device).unsqueeze(1)
            tot[h] += (w * blk).sum(0)
    return tot[0].flatten(), tot[1].flatten()
