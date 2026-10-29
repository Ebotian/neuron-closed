"""Terminal progress with elapsed time and a coarse ETA (stdlib only, no torch).

Used by scan.py and verify.py so a long GPU run never looks frozen. The ETA is the
mean per-item rate over completed items: exact when items cost the same, optimistic
when the first item paid kernel warm-up. It starts after model load, so hashing and
loading time is not mixed into the probe estimate.

Deliberately dependency-free: tests must run on a CPU box without torch.
"""
from __future__ import annotations

import sys
import time


def human_duration(seconds):
    """Render a duration: 45 -> '45s', 600 -> '10.0 min', 7200 -> '2.0 h'."""
    s = max(0.0, float(seconds))
    if s < 90:
        return f'{s:.0f}s'
    if s < 90 * 60:
        return f'{s / 60:.1f} min'
    return f'{s / 3600:.1f} h'


class Progress:
    """One line per completed item: ``[i/N] elapsed, eta ...  label``.

    ``total`` must be at least 1. On resume, pass the number of items this process
    still has to compute, so the ETA does not count instant cache hits as work.
    """

    def __init__(self, total, stream=None, clock=time.perf_counter, enabled=True):
        if int(total) < 1:
            raise ValueError(f'total must be >= 1, got {total}')
        self.total = int(total)
        self.done = 0
        self.enabled = bool(enabled)
        self._clock = clock
        self._t0 = clock()
        self._out = sys.stdout if stream is None else stream

    def step(self, label=''):
        self.done += 1
        if not self.enabled:
            return
        elapsed = self._clock() - self._t0
        per_item = elapsed / self.done
        remaining = max(0.0, per_item * (self.total - self.done))
        suffix = f'  {label}' if label else ''
        print(f'[{self.done}/{self.total}] {human_duration(elapsed)} elapsed, '
              f'eta {human_duration(remaining)}{suffix}', file=self._out, flush=True)
