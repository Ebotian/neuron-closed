"""Pre-register controlled candidate arms without confounding random rank and sign.

Physical neuron-coordinate evaluations are deduplicated. An 'arm' is a VIEW of
one coordinate with a chosen polarity, not an independent trial.
"""
from __future__ import annotations
import random


def coordinate(layer, neuron):
    return f'L{int(layer)}N{int(neuron)}'


def build_arms(top, gradients, dims, random_k, seed, known=()):
    if not top:
        raise ValueError('No ranked candidates. Cannot validate Top-K (even with --random-k 0).')
    if random_k < 0:raise ValueError('--random-k must be >=0')
    rng=random.Random(seed)
    arms=[]
    used={(int(x['layer']),int(x['neuron'])) for x in top}
    for i,c in enumerate(top):
        layer,neuron=int(c['layer']),int(c['neuron'])
        if layer not in dims or not (0<=neuron<dims[layer]):raise ValueError('Bad candidate coordinates')
        grad=float(gradients[layer]['mean'][neuron].item())
        sign=1 if grad>=0 else -1
        arms.append({'arm_id':f'G{i:03d}', 'group':'gradient_top',
                     'coordinate':coordinate(layer,neuron),'layer':layer,'neuron':neuron,
                     'sign':sign,'predicted_gradient':grad,'matched_pair':i if i<random_k else None})
    if random_k and not top:raise ValueError('Cannot sample without Top-K')
    for i in range(random_k):
        parent=arms[i%len(top)]
        layer=parent['layer']; n_dim=dims[layer]
        options=[j for j in range(n_dim) if (layer,j) not in used]
        if not options:raise ValueError(f'No unsampled neurons in layer {layer}')
        n=rng.choice(options);used.add((layer,n))
        own_grad=float(gradients[layer]['mean'][n].item())
        own_sign=1 if own_grad>=0 else -1
        for grp,label,sign in [('random_own_sign','R',own_sign),
                               ('random_parent_sign','P',parent['sign'])]:
            arms.append({'arm_id':f'{label}{i:03d}','group':grp,
                         'coordinate':coordinate(layer,n),'layer':layer,'neuron':n,
                         'sign':sign,'predicted_gradient':own_grad,
                         'matched_pair':i%len(top),'matched_to':parent['arm_id']})
    for i,(layer,n,sign) in enumerate(known):
        if layer not in dims or not 0<=n<dims[layer]:continue
        gradient=float(gradients[layer]['mean'][n].item())
        arms.append({'arm_id':f'K{i:03d}','group':'known_posthoc',
                     'coordinate':coordinate(layer,n),'layer':layer,'neuron':n,
                     'sign':sign,'predicted_gradient':gradient,
                     'matched_pair':None})
    return arms


def physical_coordinates(arms):
    return sorted({(a['layer'],a['neuron']) for a in arms})
