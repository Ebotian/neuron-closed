"""Loading the frozen model faster - without touching the vendor package.

WHY. The 8x32 benchmark spends ~15 s of its 48 s wall clock at 0% GPU while `engine.load_model` reads 8.8 GB of weights;
that single step is what keeps the whole-run GPU mean at 59% instead of the 77% measured inside the active window. The
loader lives in `vendor/qwen-gradient-discovery-v6/engine.py`, which this repo may not edit, so the same call is made
here with two additional flags:

  low_cpu_mem_usage + device_map  materialise the weights STRAIGHT on the GPU through accelerate's meta-device
                                  path, instead of building them on the CPU and copying 8.8 GB afterwards. Measured
                                  in isolation on this box: 0.41 s to read into host memory, then the .to(cuda) copy
                                  made the whole load 11.8 s; with device_map the load is 3.95 s and the copy is gone.

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
    kw = dict(dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True, low_cpu_mem_usage=True)
    try:                                      # straight-to-device when accelerate is present, host copy otherwise
        model = Qwen3_5ForConditionalGeneration.from_pretrained(path, device_map=device, **kw)
    except Exception as e:                    # no accelerate: fall back to the CPU build + copy
        print(f'  device_map unavailable ({type(e).__name__}), falling back to host load', flush=True)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(path, **kw).to(device)
    model = model.eval()
    model.requires_grad_(False)
    lm = E.model_lm(model)
    if len(lm.layers) != 32:
        raise RuntimeError(f'Expected 32 layers, got {len(lm.layers)}')
    return tokenizer, model, lm
