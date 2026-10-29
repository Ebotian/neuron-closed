#!/usr/bin/env python3
"""Qwen3.5-4B MLP steering replication. See README.md.

Requires a CUDA PyTorch environment, transformers>=5, accelerate.
No weights are modified. Every run resets hooks via a context manager.
"""
import argparse
import contextlib
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

# Verbatim from lyramakesmusic/neuron-explorer/explorer/runtime.py (2026-10-26 review).
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)
TARGETS = [(14, 8949, +1.), (13, 7956, -1.)]


def residual_p50(lm, tokenizer, layer_id, device, stats_dir=None):
    """Use author's saved sweep statistic if supplied, else author's reference-text fallback."""
    if stats_dir:
        import numpy as np
        p = Path(stats_dir) / f'q35-4b_L{layer_id}.npz'
        if p.exists():
            return float(np.load(p)['resid_norm_q10_50_90'][1]), f'precomputed:{p}'
    vals = []
    def read_hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] < 2:
            raise ValueError('Reference text must tokenize to >=2 tokens')
        vals.append(h[0, 1:].float().norm(dim=-1).median().item())
    handle = lm.layers[layer_id].register_forward_hook(read_hook)
    try:
        input_ids = tokenizer(REFERENCE_TEXT, return_tensors='pt').input_ids.to(device)
        with torch.inference_mode():
            lm(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    if len(vals) != 1:
        raise AssertionError(f'Expected one layer output, got {len(vals)}')
    return vals[0], 'reference_text'


def scale_for(lm, p50, layer_id, neuron_id, dose):
    w = lm.layers[layer_id].mlp.down_proj.weight
    if not (0 <= neuron_id < w.shape[1]):
        raise ValueError(f'Invalid neuron {neuron_id} for layer {layer_id}')
    col_norm = w[:, neuron_id].float().norm().item()
    return dose * p50[layer_id] / col_norm


@contextlib.contextmanager
def steering(lm, p50, interventions):
    """interventions is list[(layer, neuron, signed_dose)], applied to all tokens."""
    grouped = {}
    for layer, neuron, dose in interventions:
        grouped.setdefault(layer, []).append((neuron, scale_for(lm, p50, layer, neuron, dose)))
    handles = []
    counts = {layer: 0 for layer in grouped}
    try:
        for layer, adjustments in grouped.items():
            def make_hook(layer_id, adjs):
                def hook(_module, inputs):
                    counts[layer_id] += 1
                    a = inputs[0].clone()
                    for neuron_id, addition in adjs:
                        a[..., neuron_id] += addition
                    return (a,) + inputs[1:]
                return hook
            handles.append(lm.layers[layer].mlp.down_proj.register_forward_pre_hook(make_hook(layer, adjustments)))
        yield counts
    finally:
        for h in handles:
            h.remove()


def generate(model, tokenizer, lm, p50, prompt, interventions, max_new_tokens, device):
    messages = [{'role': 'user', 'content': prompt}]
    # Same non-thinking template as original explorer implementation.
    chat_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=False,
    )
    ids = tokenizer(
        chat_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(device)
    with steering(lm, p50, interventions) as hook_calls, torch.inference_mode():
        result = model.generate(input_ids=ids, do_sample=False, use_cache=True,
                                max_new_tokens=max_new_tokens,
                                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    for layer, count in hook_calls.items():
        if count == 0:
            raise AssertionError(f'Steering hook for layer {layer} was not called!')
    out_ids = result[0, ids.shape[1]:]
    return tokenizer.decode(out_ids, skip_special_tokens=True), out_ids.numel(), hook_calls


def make_conditions(num_random, seed, dose):
    rng = random.Random(seed)
    a, b = TARGETS
    cond = {
        'baseline': [],
        'zero_hook': [(a[0], a[1], 0.), (b[0], b[1], 0.)],
        'L14_only': [(a[0], a[1], a[2] * dose)],
        'L13_only': [(b[0], b[1], b[2] * dose)],
        'target_pair': [(a[0], a[1], a[2] * dose), (b[0], b[1], b[2] * dose)],
        'reversed_pair': [(a[0], a[1], -a[2] * dose), (b[0], b[1], -b[2] * dose)],
    }
    for i in range(num_random):
        n14 = rng.choice([j for j in range(9216) if j != a[1]])
        n13 = rng.choice([j for j in range(9216) if j != b[1]])
        cond[f'random_pair_{i:02d}'] = [(14, n14, dose), (13, n13, -dose)]
    return cond


def parse_prompts(path, limit):
    prompts = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            assert isinstance(x['id'], str) and isinstance(x['prompt'], str)
            prompts.append(x)
            if limit and len(prompts) >= limit:
                break
    if len({x['id'] for x in prompts}) != len(prompts):
        raise ValueError('Duplicate prompt ids')
    return prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--prompts', default=str(Path(__file__).with_name('prompts.jsonl')))
    ap.add_argument('--output-dir', default='./out')
    ap.add_argument('--stats-dir', default=None, help='Optional author npz stats directory')
    ap.add_argument('--limit', type=int, default=8)
    ap.add_argument('--random-pairs', type=int, default=2)
    ap.add_argument('--dose', type=float, default=1.)
    ap.add_argument('--max-new-tokens', type=int, default=128)
    ap.add_argument('--seed', type=int, default=1729)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required; check nvidia-smi / installed PyTorch')
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda:0'
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = parse_prompts(args.prompts, args.limit)
    conditions = make_conditions(args.random_pairs, args.seed, args.dose)

    print('Loading', args.model, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    lm = model.model.language_model
    if len(lm.layers) != 32:
        raise AssertionError('Expected the 32-layer Qwen3.5-4B model')
    p50, provenance = {}, {}
    for layer in (13, 14):
        p50[layer], provenance[layer] = residual_p50(lm, tok, layer, device, args.stats_dir)
        print(f'L{layer}: p50={p50[layer]:.5f} ({provenance[layer]})', flush=True)
    for layer, neuron, dose in TARGETS:
        w = lm.layers[layer].mlp.down_proj.weight
        print(f'L{layer} N{neuron} dose={dose:+g}: shape={tuple(w.shape)}, '
              f'alpha={scale_for(lm, p50, layer, neuron, dose):.5f}', flush=True)
    manifest = {
        'model': args.model, 'model_config_sha256': hashlib.sha256((Path(args.model) / 'config.json').read_bytes()).hexdigest(),
        'dtype': 'bfloat16', 'generation': {'do_sample': False, 'max_new_tokens': args.max_new_tokens,
                                          'enable_thinking': False},
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(),
        'seed': args.seed, 'dose': args.dose, 'p50': p50, 'p50_source': provenance,
        'conditions': conditions, 'source': 'lyramakesmusic/neuron-explorer/explorer/runtime.py'
    }
    manifest_str = json.dumps(manifest, sort_keys=True, ensure_ascii=False)
    run_id = hashlib.sha256(manifest_str.encode()).hexdigest()[:12]
    manifest['run_id'] = run_id
    (out_dir / f'manifest_{run_id}.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    results_path = out_dir / f'results_{run_id}.jsonl'
    completed = set()
    if results_path.exists():
        for line in results_path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                item = json.loads(line)
                completed.add((item['prompt_id'], item['condition']))
    with results_path.open('a', encoding='utf-8') as log:
        for p in prompts:
            collected = {}
            for cname, interventions in conditions.items():
                key = (p['id'], cname)
                if key in completed:
                    continue
                start = time.perf_counter()
                response, n_new, hook_calls = generate(model, tok, lm, p50, p['prompt'], interventions,
                                                       args.max_new_tokens, device)
                item = {'run_id': run_id, 'prompt_id': p['id'], 'prompt': p['prompt'],
                        'category': p.get('category', ''), 'condition': cname,
                        'response': response, 'generated_tokens': n_new,
                        'elapsed_s': round(time.perf_counter() - start, 3),
                        'hook_calls': hook_calls, 'interventions': interventions}
                log.write(json.dumps(item, ensure_ascii=False) + '\n')
                log.flush()
                collected[cname] = response
                print(f"{p['id']:>5} {cname:>16} {n_new:>3} tokens {item['elapsed_s']:>7.1f}s "
                      f"hook_calls={hook_calls}", flush=True)
            if 'baseline' in collected and 'zero_hook' in collected:
                assert collected['baseline'] == collected['zero_hook'], \
                    f'No-op hook changed output for prompt {p["id"]}; check environment!'
    print(f'Done: {results_path}', flush=True)


if __name__ == '__main__':
    main()
