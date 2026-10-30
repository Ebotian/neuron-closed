#!/usr/bin/env python3
"""Verify analysis/icc.py against the run that produced the stored Gram (task-5).

PURE PYTHON - no torch, no model: it loads results/gram_256.json (the n x n Gram dumped by p_a2) and recomputes the
intraclass correlation with analysis/icc.py, then compares against the ICC the run itself reported. This is the check
that `make check` can run, and it is the reason the estimator lives in its own layer: the number must be re-derivable
from the stored Gram alone, months later, without a GPU.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here / 'analysis'))
from analysis.icc import icc_from_gram_py


def main():
    gram = json.load(open(_here.parent / 'results/gram_256.json'))
    run = json.load(open(_here.parent / 'results/icc_run_record.json'))['ICC']
    got = icc_from_gram_py(gram['G'], gram['group_sizes'], k=gram['k'])
    keys = ['tr_sigma', 'g_sq', 'icc']
    diffs = {k: abs(got[k] - run[k]) for k in keys}
    iv_ok = (got['icc_interval'] is not None
             and all(abs(a - b) < 1e-12 for a, b in zip(got['icc_interval'], run['icc_interval'])))
    print('recomputed from the stored Gram, no model, no torch:')
    print(f"  MS_w {got['tr_sigma']:.9f} vs run {run['tr_sigma']:.9f}")
    print(f"  ICC  {got['icc']:.9f} vs run {run['icc']:.9f}")
    print(f"  interval {got['icc_interval']} vs run {run['icc_interval']} -> identical: {iv_ok}")
    ok = all(v < 1e-12 for v in diffs.values()) and iv_ok
    print('ICC-VERIFY', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
