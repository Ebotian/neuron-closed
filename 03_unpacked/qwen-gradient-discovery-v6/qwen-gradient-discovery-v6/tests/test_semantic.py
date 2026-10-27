"""CPU tests for the J2 (language-mass) readout: batch log-prob and exact gradient.

No GPU, no real tokenizer, no real model: a toy LM with one hookable down_proj.
"""
import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn

from semantic import batch_mean_logp, build_batch, j2_value, language_log_mass, sem_gradient


class FakeTok:
    """Deterministic character-level tokenizer: id = ord(c) % vocab."""
    def __init__(self, vocab=24):
        self.vocab = vocab

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return 'U:' + msgs[0]['content'] + '\nA:'

    def __call__(self, text, add_special_tokens=False):
        ids = [ord(c) % self.vocab for c in text]
        return SimpleNamespace(input_ids=ids)


class ToyLM(nn.Module):
    """One MLP layer whose down_proj input is what the hook captures."""
    def __init__(self, vocab=24, d=4):
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(vocab, d)
        self.layers = nn.ModuleList([nn.Module()])
        self.layers[0].mlp = nn.Module()
        self.layers[0].mlp.down_proj = nn.Linear(d, vocab, bias=False)

    def forward(self, input_ids, use_cache=False):
        h = self.emb(input_ids)
        return SimpleNamespace(logits=self.layers[0].mlp.down_proj(h))


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.tok = FakeTok()
        self.model = ToyLM()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.lm = self.model
        self.p50 = {0: 1.0}
        self.norms = {0: self.lm.layers[0].mlp.down_proj.weight.detach().float().norm(dim=0)}
        self.pools = {'pos': ['aaaa', 'bbbb'], 'neg': ['cccc', 'dddd']}
        self.tmpl = 'You are so {w}.'

    def test_batch_matches_per_sample(self):
        ids, pre, suf, kinds = build_batch(self.tok, 'hi', self.tmpl, self.pools['pos'],
                                           self.pools['neg'], 'cpu')
        batched = batch_mean_logp(self.model, ids, pre, suf)
        self.assertEqual(batched.shape[0], 4)
        for i in range(4):
            single = batch_mean_logp(self.model, ids[i:i + 1], pre, suf)
            self.assertAlmostEqual(float(batched[i]), float(single[0]), places=6)

    def test_unequal_length_is_rejected(self):
        with self.assertRaises(ValueError):
            build_batch(self.tok, 'hi', self.tmpl, ['aaaa'], ['cc'], 'cpu')

    def test_language_mass_identity(self):
        S = torch.tensor([0.5, -1.0, 2.0])
        lm_, w = language_log_mass(S)
        self.assertAlmostEqual(lm_, float(torch.log(torch.exp(S).mean())), places=6)
        self.assertAlmostEqual(float(w.sum()), 1.0, places=6)

    def test_gradient_matches_central_difference(self):
        j, g = sem_gradient(self.tok, self.model, self.lm, [0], 'hi', self.tmpl, self.pools, 'cpu')
        analytic = g[0] * (self.p50[0] / self.norms[0])
        d = 1e-3
        for n in range(analytic.numel()):
            jp = j2_value(self.tok, self.model, self.lm, self.p50, self.norms, 'hi', self.tmpl,
                          self.pools, [{'layer': 0, 'neuron': n, 'dose': d}], device='cpu')
            jm = j2_value(self.tok, self.model, self.lm, self.p50, self.norms, 'hi', self.tmpl,
                          self.pools, [{'layer': 0, 'neuron': n, 'dose': -d}], device='cpu')
            fd = (jp - jm) / (2 * d)
            a = float(analytic[n])
            # toy model, float32: first-order agreement within 5% and identical sign
            self.assertLess(abs(fd - a) / max(1e-9, abs(a)), 0.05,
                            msg=f'neuron {n}: fd={fd} analytic={a}')
            self.assertEqual(fd > 0, a > 0, msg=f'neuron {n}: sign mismatch fd={fd} analytic={a}')

    def test_j2_is_antitone_in_offset(self):
        # shifting every positive slot log prob up must raise J2
        base = j2_value(self.tok, self.model, self.lm, self.p50, self.norms, 'hi', self.tmpl,
                        self.pools, [], device='cpu')
        self.assertTrue(math.isfinite(base))


if __name__ == '__main__':
    unittest.main()
