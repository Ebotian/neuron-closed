"""Deprecated shim: every estimator here now lives in the analysis layer.

`split_half_curve` -> analysis/split_half.py
`icc_from_gram`    -> analysis/icc.py (pure Python: input only (Gram, group_sizes, k))
`noise_scale`, `bootstrap_stats`, `bootstrap_kappa`, `context_summaries`, `gram_from_scores` -> removed.

WHY THEY WENT. Two implementations of one estimator is the defect this refactor exists to remove, and `make check`'s
`noregress` item now enforces it. The bootstrap and BCa code was also superseded by the F-distribution interval in
analysis/icc.py: with 8 contexts the bootstrap gave an interval that did not even contain the point estimate, and the
closed form is both cheaper and correct. Nothing in the repository imports this module any more; it is kept only so an
external caller gets a clear deprecation instead of an ImportError.
"""
from __future__ import annotations

import warnings

warnings.warn('reliability.py is a deprecated shim; import from analysis.split_half / analysis.icc instead',
              DeprecationWarning, stacklevel=2)

from analysis.split_half import split_half_curve                      # noqa: E402,F401
from analysis.icc import icc_from_gram_py as icc_from_gram            # noqa: E402,F401
