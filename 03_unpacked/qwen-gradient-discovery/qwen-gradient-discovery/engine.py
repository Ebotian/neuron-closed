"""White-box gradient discovery and steering for Qwen3.5-4B, no weight updates.

This module deliberately imports only torch at import-time, so unit tests run on CPU.
"""
from __future__ import annotations
import contextlib
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# Reference text from the prior neuron-explorer reproduction (fallback p50 only).
REFERENCE_TEXT = (
    "The history of the region is long and varied. Farmers settled the river valley first, then traders arrived, "
    "and by the twelfth century the town had a market, a bridge, and a quarrel with its neighbours over tolls. "
    "def merge(a, b):\n    out = []\n    while a and b:\n        out.append(a.pop(0) if a[0] < b[0] else b.pop(0))\n"
    "    return out + a + b\nShe asked me what I thought of the plan, and I told her honestly that it seemed rushed. "
)
KNOWN_EVIL = ((14, 8949, +1), (13, 7956, -1))


def parse_layers(spec: str, max_layers: int = 32) -> list[int]:
    ids = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = map(int, part.split('-', 1))
            if a > b:
                raise ValueError(f'Invalid layer range: {part}')
            ids.update(range(a, b + 1))
        else:
            ids.add(int(part))
    if not ids or min(ids) < 0 or max(ids) >= max_layers:
        raise ValueError(f'Layers must be within 0..{max_layers-1}')
    return sorted(ids)


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]


def write_json(path: str | Path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf8')


def model_sha(path: str | Path) -> str:
    return hashlib.sha256((Path(path) / 'config.json').read_bytes()).hexdigest()


def model_lm(model):
    return model.model.language_model


def load_model(path: str, device='cuda:0'):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)  # never allocate parameter gradients; do not fine-tune
    lm = model_lm(model)
    if len(lm.layers) != 32:
        raise RuntimeError(f'Expected 32 layers, got {len(lm.layers)}')
    return tokenizer, model, lm


def prompt_prefix(tokenizer, prompt: str) -> list[int]:
    text = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False, enable_thinking=False, add_generation_prompt=True
    )
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if not ids or not isinstance(ids[0], int):
        raise ValueError('Chat template was not tokenized into integer IDs')
    return ids


def candidate_ids(tokenizer, prompt: str, completion: str, device: str, max_length: int = 384):
    prefix = prompt_prefix(tokenizer, prompt)
    suffix = tokenizer(completion, add_special_tokens=False).input_ids
    if len(suffix) < 1:
        raise ValueError('Completion must tokenize to >=1 token')
    ids = prefix + suffix
    if len(ids) > max_length:
        raise ValueError(f'Sequence too long: {len(ids)} > --max-seq {max_length}. Reduce prompt.')
    return torch.tensor([ids], dtype=torch.long, device=device), len(prefix), len(suffix)


def continuation_logp(model, ids: torch.Tensor, prefix_len: int, suffix_len: int) -> torch.Tensor:
    """Mean logP of a forced continuation given an exact prefix.

    Shift: predicting suffix token at position k uses logits from k-1.
    """
    out = model(input_ids=ids, use_cache=False)
    logits = out.logits[:, prefix_len - 1 : prefix_len - 1 + suffix_len]
    target = ids[:, prefix_len : prefix_len + suffix_len]
    if logits.shape[1] != suffix_len:
        raise AssertionError('Token alignment failed')
    return -F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                            target.reshape(-1), reduction='mean')


@contextlib.contextmanager
def capture_mlp_inputs(lm, layers: list[int]):
    """Capture down_proj pre-activations, force autograd entry at the first layer.

    IMPORTANT: no detach is performed if tensor already requires grad: this
    preserves downstream -> upstream causal paths across all hooked layers.
    """
    captures = {}
    handles = []
    for layer_id in layers:
        def build_hook(idx):
            def hook(_module, args):
                x = args[0]
                if not x.requires_grad:
                    x = x.detach().requires_grad_(True)
                # only one forward at a time; older references are overwritten
                captures[idx] = x
                return (x,) + args[1:]
            return hook
        handles.append(lm.layers[layer_id].mlp.down_proj.register_forward_pre_hook(build_hook(layer_id)))
    try:
        yield captures
    finally:
        for handle in handles:
            handle.remove()
        captures.clear()


def score_gradient(tokenizer, model, lm, layers, probe, completion_key,
                   device, max_length=384):
    ids, pre_n, suf_n = candidate_ids(tokenizer, probe['prompt'], probe[completion_key], device, max_length)
    with torch.enable_grad(), capture_mlp_inputs(lm, layers) as captures:
        score = continuation_logp(model, ids, pre_n, suf_n)
        inputs = [captures[l] for l in layers]
        grads = torch.autograd.grad(score, inputs, allow_unused=False)
        # A fixed additive intervention is broadcast to EVERY input position.
        # Sum gradients across positions, matching the forward pre-hook semantics.
        result = {l: g.detach().float().sum(dim=tuple(range(g.ndim - 1))).cpu()
                  for l, g in zip(layers, grads)}
        value = float(score.detach().item())
    return value, result


def estimate_p50(tokenizer, model, lm, layers, device):
    vals = {}
    handles = []
    for layer_id in layers:
        def make_hook(idx):
            def hook(_module, _inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                vals[idx] = h[0, 1:].float().norm(dim=-1).median().item()
            return hook
        handles.append(lm.layers[layer_id].register_forward_hook(make_hook(layer_id)))
    try:
        ids = tokenizer(REFERENCE_TEXT, return_tensors='pt').input_ids.to(device)
        with torch.inference_mode():
            model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if set(vals) != set(layers):
        raise RuntimeError('Missing residual norm stats for at least one layer')
    return vals


def column_norms(lm, layers):
    return {l: lm.layers[l].mlp.down_proj.weight.detach().float().norm(dim=0).cpu()
            for l in layers}


def score_for_dose(g: torch.Tensor, p50: float, col_norm: torch.Tensor) -> torch.Tensor:
    return g * (p50 / col_norm.clamp(min=1e-12))


def aggregate_probe_gradients(per_probe: list[dict[int, torch.Tensor]],
                              layers: list[int], p50, norms):
    """Positive signed derivative means dose + improves preferred logP margin."""
    result = {}
    for l in layers:
        xs = torch.stack([score_for_dose(x[l], p50[l], norms[l]) for x in per_probe])
        mean = xs.mean(0)
        # Sign-consistency among discovery contexts, independent of magnitude.
        consistency = (torch.sign(xs) == torch.sign(mean).unsqueeze(0)).float().mean(0)
        result[l] = {'mean': mean, 'consistency': consistency,
                     'per_probe': xs}
    return result


def ranked_candidates(aggregates, top_k: int, only_sign_consistent=False):
    entries = []
    for l, stats in aggregates.items():
        v = stats['mean']
        for n in range(v.numel()):
            score = float(v[n].item())
            sign = 1 if score >= 0 else -1
            agree = float(stats['consistency'][n].item())
            if only_sign_consistent and agree < 0.66:
                continue
            entries.append({'layer': int(l), 'neuron': n, 'sign': sign,
                            'gradient_per_dose': score,
                            'abs_gradient': abs(score), 'sign_consistency': agree})
    entries.sort(key=lambda t: t['abs_gradient'], reverse=True)
    return entries[:top_k], entries


@contextlib.contextmanager
def apply_steering(lm, interventions, p50, norms):
    """Interventions are {layer, neuron, dose}; can be combined across layers."""
    by_layer = {}
    for item in interventions:
        l, n, dose = int(item['layer']), int(item['neuron']), float(item['dose'])
        if not (0 <= n < len(norms[l])):
            raise ValueError(f'Invalid neuron {l}:{n}')
        alpha = dose * p50[l] / max(float(norms[l][n]), 1e-12)
        by_layer.setdefault(l, []).append((n, alpha))
    hooks = []
    calls = {l: 0 for l in by_layer}
    try:
        for l, adjustments in by_layer.items():
            def make_hook(layer_id, adjs):
                def hook(_module, inp):
                    calls[layer_id] += 1
                    act = inp[0].clone()
                    for n, alpha in adjs:
                        act[..., n] += alpha
                    return (act,) + inp[1:]
                return hook
            hooks.append(lm.layers[l].mlp.down_proj.register_forward_pre_hook(make_hook(l, adjustments)))
        yield calls
    finally:
        for h in hooks:
            h.remove()


def pair_objective(tokenizer, model, lm, probe, device, p50, norms,
                   interventions=None, max_length=384):
    interventions = interventions or []
    with apply_steering(lm, interventions, p50, norms) as calls, torch.inference_mode():
        values = []
        for typ in ('positive', 'negative'):
            ids, pre_n, suf_n = candidate_ids(tokenizer, probe['prompt'], probe[typ], device, max_length)
            val = continuation_logp(model, ids, pre_n, suf_n)
            values.append(float(val.item()))
    return values[0] - values[1], dict(calls)


def generate_response(tokenizer, model, lm, prompt, interventions, p50, norms,
                      device='cuda:0', max_new_tokens=96):
    prefix = prompt_prefix(tokenizer, prompt)
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    with apply_steering(lm, interventions, p50, norms) as calls, torch.inference_mode():
        result = model.generate(input_ids=ids, do_sample=False, max_new_tokens=max_new_tokens,
                                use_cache=True, pad_token_id=tokenizer.pad_token_id
                                if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    out = result[0, ids.shape[1]:]
    return tokenizer.decode(out, skip_special_tokens=True), dict(calls), int(out.numel())


def known_ranks(all_entries):
    ranks = {}
    for l, n, sign in KNOWN_EVIL:
        match = next((i+1 for i, e in enumerate(all_entries)
                      if e['layer'] == l and e['neuron'] == n and e['sign'] == sign), None)
        # separately rank without insisting on correct sign
        absolute = next((i+1 for i, e in enumerate(all_entries)
                         if e['layer'] == l and e['neuron'] == n), None)
        ranks[f'L{l}N{n}{sign:+}'] = {'signed_rank': match, 'absolute_rank': absolute}
    return ranks


def fit_slope_example():
    """Pure-torch fixture for unit tests: dJ/dd = (m/||w||) * g."""
    W = torch.tensor([[1., 2., 0.], [0., 1., 1.]])
    g_out = torch.tensor([3., -2.])
    m = 4.
    g = W.T @ g_out
    return score_for_dose(g, m, W.norm(dim=0))
