import math
import unittest
import torch
from torch import nn
from engine import (capture_mlp_inputs, apply_steering, parse_layers,
                    fit_slope_example, score_for_dose, aggregate_probe_gradients,
                    ranked_candidates)


class SimpleLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Module()
        self.mlp.down_proj=nn.Linear(5,3,bias=False)
    def forward(self, x):
        # x is 5-dimensional MLP intermediate, output is 3-dimensional
        return self.mlp.down_proj(x)

class SimpleLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.ModuleList([SimpleLayer()])
    def forward(self,x):
        return self.layers[0](x)

class EngineTests(unittest.TestCase):
    def test_layer_parse(self):
        self.assertEqual(parse_layers('13-14,18,14'),[13,14,18])
        with self.assertRaises(ValueError):parse_layers('32')

    def test_gradient_matches_fd(self):
        torch.manual_seed(3)
        lm=SimpleLM()
        for p in lm.parameters():p.requires_grad_(False)
        x=torch.randn(1,4,5)
        target=torch.tensor([2.,-1.,3.])
        p50={0:2.3}
        norms={0:lm.layers[0].mlp.down_proj.weight.detach().float().norm(dim=0)}
        with capture_mlp_inputs(lm,[0]) as cap:
            score=(lm(x)*target).sum()
            g,=torch.autograd.grad(score,[cap[0]])
            gradient=g.sum((0,1)).float()
            analytic=score_for_dose(gradient,p50[0],norms[0])
        base=(lm(x)*target).sum().item()
        for n in range(5):
            eps=1e-3
            with apply_steering(lm,[{'layer':0,'neuron':n,'dose':eps}],p50,norms):
                v=(lm(x)*target).sum().item()
            self.assertAlmostEqual((v-base)/eps,analytic[n].item(),delta=0.01)
        self.assertAlmostEqual((lm(x)*target).sum().item(),base,places=6)

    def test_steering_composition_cancellation(self):
        lm=SimpleLM()
        x=torch.randn(1,2,5)
        p50={0:2.}
        norms={0:lm.layers[0].mlp.down_proj.weight.detach().float().norm(dim=0)}
        y0=lm(x)
        with apply_steering(lm,[{'layer':0,'neuron':1,'dose':0.3},
                               {'layer':0,'neuron':1,'dose':-0.3}],p50,norms) as calls:
            y=lm(x)
        self.assertTrue(torch.allclose(y,y0,atol=1e-6))
        self.assertEqual(calls[0],1)

    def test_ranking_both_signs(self):
        x={13:{'mean':torch.tensor([0.1,-4.,2.]),'consistency':torch.tensor([1.,1.,.5])}}
        top,all_items=ranked_candidates(x,2)
        self.assertEqual((top[0]['layer'],top[0]['neuron'],top[0]['sign']), (13,1,-1))
        self.assertEqual(top[1]['neuron'],2)

if __name__=='__main__':unittest.main()
