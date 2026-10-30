#!/usr/bin/env python3
"""The independent-half SNR identities, checked EXACTLY on a finite model - no torch, no GPU.

THE CLAIM UNDER TEST. `analysis/report._half_estimates` produces two estimates a and b of the same score term g from
disjoint rollouts, each recomputing its own leave-one-out baseline. The SNR block then uses
    E[<a,b>] = ||g||^2        and        0.5 * E||a-b||^2 = tr Cov(ghat).
Both identities need a and b INDEPENDENT and each UNBIASED. The first version of that function sliced the full-group
weights, which silently broke both - and no value comparison could see it, because the numbers still looked plausible.
The previous metadata claimed such a test "needs a GPU stack"; the auditor showed it does not, and was right.

THE MODEL. One score coordinate, one context, K=4 rollouts, Y ~ Bernoulli(1/2), phi(Y) = Y, s(Y) = Y - 1/2. Then
g = E[phi * s] = 1/4 and every rollout contributes phi_i * s_i in {0, 1/2}. The model is tiny enough to enumerate all
2^4 = 16 outcomes exactly, so the assertions below are equalities, not Monte-Carlo estimates.

WHAT IS AND IS NOT REAL HERE. The shim below implements only the tensor operations `_half_estimates` performs, and the
function under test is imported UNCHANGED from analysis/report.py, so its slicing and normalisation are genuinely
exercised. What the shim cannot prove is that the real torch does the same thing - that is covered by the GPU run and by
the auditor's independent reconstruction from the stored Gram (both 0.0652). This test exists to stop the SLICE from
coming back, which is the failure that actually happened.
"""
from __future__ import annotations

import itertools
import sys
import types
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class Vec(list):
    """A 1-D tensor stand-in: double/unsqueeze/mul/sum/flatten and ELEMENTWISE += (a plain list "+=" would concatenate)."""

    def double(self):
        return Vec(self)

    def unsqueeze(self, _dim):
        return Col([[float(x)] for x in self])

    def sum(self, _dim=None):
        return sum(self)

    def flatten(self):
        return self

    def _ew(self, other):
        return Vec(float(a) + float(b) for a, b in zip(self, other))

    def __iadd__(self, other):
        return self._ew(other)

    def __add__(self, other):
        return self._ew(other)


class Rows:
    """A K×d tensor stand-in: row slicing, elementwise multiply by a column, and column sums."""

    def __init__(self, rows):
        self.rows = [list(r) for r in rows]

    def __getitem__(self, sl):
        return Rows(self.rows[sl])

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def double(self):
        return self

    def __mul__(self, w):
        return Rows([[r[j] * w.rows[i][0] for j in range(len(r))] for i, r in enumerate(self.rows)])

    def sum(self, _dim=0):
        out = Vec(0.0 for _ in range(self.shape[1]))
        for row in self.rows:
            out += Vec(row)
        return out


class Col(Rows):
    """w_i as a K×1 column, so `w * S.block` is the production expression."""

    def __init__(self, rows):
        super().__init__(rows)

    def __mul__(self, other):
        return other.__mul__(self)


def install_shim():
    """A minimal `torch` so analysis/report.py can be imported without the real thing."""
    m = types.ModuleType('torch')
    m.float64 = 'float64'
    m.zeros = lambda dims, dtype=None, device=None: Vec(0.0 for _ in range(dims))
    m.tensor = lambda data, dtype=None, device=None: Vec([float(x) for x in data])   # .unsqueeze(1) gives a Col
    for name, mod in (('torch', m), ('torch.nn', types.ModuleType('torch.nn')),
                      ('torch.nn.functional', types.ModuleType('torch.nn.functional'))):
        sys.modules[name] = mod
    sys.modules['torch'].nn = sys.modules['torch.nn']
    sys.modules['torch.nn'].functional = sys.modules['torch.nn.functional']
    for p in (ROOT / 'scripts', ROOT / 'scripts' / 'core', ROOT / 'scripts' / 'analysis'):
        sys.path.insert(0, str(p))
    import analysis.report as rep          # noqa: E402  the REAL module, with its real _half_estimates
    return rep


def sliced_reference(phi, s, K, n_ctx):
    """The FIRST, WRONG implementation, reproduced here so the test can show it is discriminating.

    It slices the FULL-GROUP leave-one-out weights instead of recomputing each half's baseline. Keeping a copy inside the
    test is the only way to demonstrate that the identities below actually reject it - otherwise "the test passes" says
    nothing about what it would reject.
    """
    from core.weights import loo_weights      # pure Python, no torch: imported here so the module loads without the shim
    w = loo_weights(phi, n_ctx)
    half = K // 2
    a = sum(w[i] * s[i][0] for i in range(half))
    b = sum(w[i] * s[i][0] for i in range(half, 2 * half))
    return a, b


def identities(a_list, b_list, g):
    n = len(a_list)
    mean = lambda xs: Fraction(sum(xs), n)
    ea, eb = mean(a_list), mean(b_list)
    cross = mean([x * y for x, y in zip(a_list, b_list)])
    noise = mean([(x - y) ** 2 for x, y in zip(a_list, b_list)]) / 2
    return {'E[a]': (ea, g), 'E[b]': (eb, g), 'E[ab]': (cross, g * g),
            '0.5 E(a-b)^2': (noise, g * g), 'Cov(a,b)': (cross - ea * eb, Fraction(0))}


def main():
    rep = install_shim()
    n_ctx, K = 1, 4
    a_list, b_list, wa, wb = [], [], [], []
    for outcome in itertools.product((0, 1), repeat=K):
        phi = [float(y) for y in outcome]                       # phi_i = Y_i
        s = [[float(y) - 0.5] for y in outcome]                 # s_i = Y_i - 1/2, one coordinate
        a, b = rep._half_estimates([Rows(s)], [None], Vec(phi), 1, 'cpu', K, n_ctx)
        a_list.append(Fraction(a[0]).limit_denominator(10 ** 6))
        b_list.append(Fraction(b[0]).limit_denominator(10 ** 6))
        ra, rb = sliced_reference([Fraction(x) for x in phi], [[Fraction(v) for v in row] for row in s], K, n_ctx)
        wa.append(ra)
        wb.append(rb)
    g = Fraction(1, 4)
    ok = True
    for label, (got, want) in identities(a_list, b_list, g).items():
        good = got == want
        ok &= good
        print(f'  {label:14s} = {str(got):>9s}   expected {str(want):>6s}   {"ok" if good else "MISMATCH"}')
    # DISCRIMINATION: the sliced implementation must FAIL at least one identity, otherwise this test proves nothing
    rejected = [k for k, (got, want) in identities(wa, wb, g).items() if got != want]
    if not rejected:
        ok = False
        print('  the sliced implementation PASSED the identities - the test cannot detect the bug it exists for')
    else:
        print(f'  discrimination: the sliced full-group version fails {rejected} -> the identities do reject it')
    print(f'  outcomes enumerated: {len(a_list)} (2^{K}), exact rational arithmetic')
    print('SNR-VERIFY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
