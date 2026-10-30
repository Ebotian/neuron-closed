"""Device and precision policy - the single owner.

WHY. Four of today's failures were "Expected all tensors to be on the same device": a scale factor built on the
CPU was combined with a gradient living on the GPU, and the error surfaced deep inside a `+=` three frames away
from the cause. The rule is that a per-layer scale is only ever handed out already on the target device, by this
module, and that a mismatch is caught at the hand-out rather than at the arithmetic.

CONTENT
  Design(dev, dtype)      - where tensors live and in what precision the model is loaded
  .scale_for(scale)       - {layer: scale tensor already on .dev}
  .check(t)               - raises if a tensor is not on .dev (call it when a value crosses a module boundary)
  .as_float(model, lm)    - precision switch; returns the pair so callers never rebuild it by hand
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Design:
    dev: str = 'cuda:0'
    dtype: torch.dtype = torch.bfloat16

    def scale_for(self, scale: dict) -> dict:
        """Per-layer scale factors, already on `.dev`. This is the only sanctioned way to obtain them."""
        return {l: v.to(self.dev) for l, v in scale.items()}

    def check(self, t: torch.Tensor, what: str = 'tensor') -> torch.Tensor:
        if t.device.type != torch.device(self.dev).type:
            raise RuntimeError(f'{what} is on {t.device}, expected {self.dev}: the policy says a value may only '
                               f'cross a module boundary already on the design device')
        return t

    def as_float(self, model, lm, model_lm):
        """Switch the model to float32. Returns (model, lm) so callers never rebuild lm by hand."""
        if self.dtype == torch.float32:
            model = model.float()
            lm = model_lm(model)
        return model, lm
