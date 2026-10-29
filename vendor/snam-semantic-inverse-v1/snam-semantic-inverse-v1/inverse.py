"""Closed-form local control and transparent weight-only ranking."""
from __future__ import annotations
import math
import torch


def score_weights(lm, layers, v, p50, norms):
    """Direct-output projection: proxy ONLY; does not incorporate downstream Jacobians.
    Dose u has physical injection alpha = u*p50 / ||W[:,n]||.
    """
    out={}
    for l in layers:
        W=lm.layers[l].mlp.down_proj.weight.detach().float()
        if W.shape[0]!=v.numel():
            raise ValueError('Dim mismatch. Weight-only proxy requires W_down output == unembedding dimension')
        out[l]=(W.T @ v).cpu()*(p50[l]/norms[l].clamp(min=1e-12))
    return out


def top_coordinates(scores, k):
    """Per-layer torch top-k: avoids materializing/sorting all ~295K Python dicts."""
    if k<=0:raise ValueError('k must be positive')
    entries=[]
    for layer,score in scores.items():
        z=score.flatten().float().cpu()
        if not torch.isfinite(z).all():raise FloatingPointError(f'Nonfinite scores in layer {layer}')
        values,indices=torch.topk(z.abs(),min(k,z.numel()),sorted=True)
        for n in indices.tolist():
            v=float(z[n]);entries.append({'layer':int(layer),'neuron':n,
                 'sign':1 if v>=0 else -1,'slope':v,'abs_slope':abs(v)})
    return sorted(entries,key=lambda e:(-e['abs_slope'],e['layer'],e['neuron']))[:k]


def fixed_budget_control(slopes, budget):
    """Min quadratic energy for a prescribed direction: u=b*g/||g||.

    This is optimal among unit-norm interventions for the linear surrogate,
    NOT a global optimizer of the actual nonlinear LLM.
    """
    g=torch.as_tensor(slopes,dtype=torch.float64)
    if budget<=0 or not torch.isfinite(g).all():raise ValueError('Bad slope or budget')
    n=float(torch.linalg.vector_norm(g))
    if n<1e-14:raise ValueError('Zero gradient vector')
    return (budget*g/n).tolist(),budget*n


def minimum_energy_control(slopes, target_delta, max_total_budget):
    """u*=delta*g/(g'g), if delta>=0. Return feasible flag for a norm budget."""
    g=torch.as_tensor(slopes,dtype=torch.float64)
    if target_delta<=0 or max_total_budget<=0:raise ValueError('target_delta and budget must be positive')
    n2=float(g @ g)
    if n2<1e-24 or not math.isfinite(n2):
        return {'feasible':False,'reason':'zero/invalid projected gain'}
    u=(target_delta/n2*g)
    cost=float(torch.linalg.vector_norm(u))
    return {'feasible':cost<=max_total_budget,'doses':u.tolist(),
            'required_budget':cost,'predicted_delta':target_delta,
            'reason':None if cost<=max_total_budget else 'dose budget exceeded'}


def combinations(top, max_support=2, pool=32, budget=.05, target_delta=.1):
    """Top singles/pairs; global optimality only within given shortlist & surrogate."""
    import itertools
    values=top[:pool]
    results=[]
    for size in range(1,max_support+1):
        for subset in itertools.combinations(values,size):
            g=[s['slope'] for s in subset]
            if sum(x*x for x in g)<1e-24:continue
            doses, pred=fixed_budget_control(g,budget)
            minimum=minimum_energy_control(g,target_delta,budget)
            results.append({'coordinates':[{'layer':s['layer'],'neuron':s['neuron'],
                                            'dose':float(d)} for s,d in zip(subset,doses)],
                            'support':size,'surrogate_fixed_budget_delta':pred,
                            'target_delta_solution':minimum})
    results.sort(key=lambda r:-r['surrogate_fixed_budget_delta'])
    return results
