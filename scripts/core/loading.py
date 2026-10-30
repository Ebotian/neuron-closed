"""Loading the frozen model faster - without touching the vendor package.

WHY. The 8x32 benchmark spends ~15 s of its 48 s wall clock at 0% GPU while `engine.load_model` reads 8.8 GB of weights;
that single step is what keeps the whole-run GPU mean at 59% instead of the 77% measured inside the active window. The
loader lives in `vendor/qwen-gradient-discovery-v6/engine.py`, which this repo may not edit, so the same call is made
here with two additional flags:

  low_cpu_mem_usage=True  skip the full-size intermediate copy (transformers 5.x has no mmap kwarg here)

NOTHING ELSE CHANGES: same class (`Qwen3_5ForConditionalGeneration`), same dtype, same attention implementation, same
`requires_grad_(False)`, same 32-layer assertion. `scripts/verify_loading.py` checks that the model loaded this way gives
bit-identical numbers to the vendor loader before any benchmark uses it.
"""
from __future__ import annotations

import torch

import engine as E


def load_model_fast(path: str, device: str = 'cuda:0'):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True,
        low_cpu_mem_usage=True).to(device).eval()
    model.requires_grad_(False)
    lm = E.model_lm(model)
    if len(lm.layers) != 32:
        raise RuntimeError(f'Expected 32 layers, got {len(lm.layers)}')
    return tokenizer, model, lm
