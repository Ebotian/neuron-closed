#!/usr/bin/env python3
"""desire @ dose 1.0 -- plain interactive probe, streaming, Ctrl-C to cut a turn short.

Type a prompt, read the answer as it is generated. Ctrl-C stops the current answer and keeps
the session (the model stays loaded) -- that is the point: when it starts repeating, you cut
it there instead of waiting out a degenerate loop.

Steering math is imported from experiment.py, so this is the same condition as the batch run
`single:desire` (L18 N8737, strength +1.0, alpha = +1.0 * p50[L18]).

    cd /root/autodl-tmp/qwen-persona-matrix
    /root/miniconda3/bin/python -u desire_interactive.py

Empty line is ignored; `exit` or Ctrl-D leaves. Greedy decoding, ends at EOS by itself; CAP
is only a backstop for loops that never emit EOS. Ctrl-C during a turn keeps whatever text
was already printed and records how many tokens it got through.
"""
import sys, threading, time
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import experiment as X

NODE = 'desire'
DOSE = 1.0
MODEL = '/root/autodl-tmp/Qwen3.5-4B'
CAP = 4096          # backstop only: EOS normally ends the turn. Some steered outputs repeat
                    # forever (seen in the recorded runs), so an unbounded turn can hang.


class StopOnFlag(StoppingCriteria):
    """Lets the main thread stop generation cleanly instead of killing it mid-decoder."""

    def __init__(self, flag):
        self.flag = flag

    def __call__(self, input_ids, scores, **kwargs):
        return self.flag.is_set()


def turn(model, tok, lm, p50d, terms, prompt):
    ids = X.prepared_tokens(tok, prompt, '', 'cuda:0')
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    flag, holder = threading.Event(), {}

    def work():
        try:
            with X.installed_hooks(lm, p50d, terms) as calls, torch.inference_mode():
                holder['calls'] = calls
                holder['out'] = model.generate(
                    input_ids=ids, max_new_tokens=CAP, do_sample=False, use_cache=True,
                    streamer=streamer,
                    stopping_criteria=StoppingCriteriaList([StopOnFlag(flag)]),
                    pad_token_id=tok.pad_token_id if tok.pad_token_id is not None
                    else tok.eos_token_id)
        except Exception as e:                      # noqa: BLE001 - surface it, do not die
            holder['error'] = f'{type(e).__name__}: {e}'
        finally:
            streamer.end()

    th = threading.Thread(target=work, daemon=True)
    t0 = time.perf_counter()
    first = None
    th.start()
    try:
        for chunk in streamer:
            if first is None and chunk:
                first = time.perf_counter() - t0
            sys.stdout.write(chunk)
            sys.stdout.flush()
    except KeyboardInterrupt:
        flag.set()                                  # stop at the next decode step
        print('\n<interrupted by you>', flush=True)
    try:
        th.join(timeout=120)
    except KeyboardInterrupt:                       # a second Ctrl-C while waiting for the stop
        print('<still stopping; the model is loaded, next prompt is fine>', flush=True)

    dt = time.perf_counter() - t0
    if 'error' in holder:
        print(f'\n[error] {holder["error"]}')
        return
    out = holder.get('out')
    n = (out.size(1) - ids.shape[1]) if out is not None else None
    bits = [f'{n if n is not None else "?"} tok', f'{dt:.1f}s']
    if first is not None:
        bits.append(f'first token {first:.2f}s')
    if n:
        bits.append(f'{n / dt:.1f} tok/s')
    if holder.get('calls'):
        bits.append(f'hook={holder["calls"]}')
    if n is not None and n >= CAP:
        bits.append('hit the 4096 backstop')
    print(f'\n[{", ".join(bits)}]\n', flush=True)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else MODEL
    print(f'loading {path} ...', flush=True)
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation='sdpa').to('cuda:0').eval()
    lm = X.get_lm(model)

    layer, neuron, strength = X.TRAITS[NODE]
    p50, src = X.p50_for(lm, tok, layer, 'cuda:0')
    p50d = {layer: p50}
    terms = X.make_terms([NODE], DOSE)
    print(f'{NODE}: L{layer} N{neuron} strength={strength:+.1f} dose={DOSE} '
          f'-> alpha={strength * DOSE:+.2f}*p50  (p50={p50!r}, {src})')
    print("type a prompt; Ctrl-C cuts the current answer; empty line ignored; "
          "'exit' or Ctrl-D leaves\n", flush=True)

    while True:
        try:
            prompt = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt.lower() in ('exit', 'quit', 'q', '/q'):
            break
        try:
            turn(model, tok, lm, p50d, terms, prompt)
        except KeyboardInterrupt:
            print('\n(interrupted)')


if __name__ == '__main__':
    main()
