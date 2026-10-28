import unittest
import torch
from stability import loo_analysis

class StabilityTests(unittest.TestCase):
    def test_six_stable_probes(self):
        per={0:torch.tensor([[10.,3.,-.5],[9.,4.,-.2],[12.,2.,-.8],[10.,5.,-.5]],dtype=torch.float32)}
        r=loo_analysis(per,top_k=2)
        self.assertEqual(r['n_discovery_probes'],4)
        self.assertAlmostEqual(r['mean_topk_retention'],1)
        self.assertAlmostEqual(r['mean_leftout_sign_hit'],1)
    def test_single_probe_rejected(self):
        with self.assertRaises(ValueError):loo_analysis({0:torch.ones(1,8)},4)

if __name__=='__main__':unittest.main()
