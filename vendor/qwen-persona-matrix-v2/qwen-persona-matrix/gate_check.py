#!/usr/bin/env python3
"""Gate 0 + Gate 2 for qwen-persona-matrix (v2-p1).

Purpose: decide -- before spending hours of GPU -- whether this package actually
reproduces two things that are already published in the older `qwen-evil-repro` runs:

  Gate 0 (calibration identity)
      p50 at L13/L14 must equal 9.752792358398438 / 10.32901382446289.
      REF_TEXT is byte-identical to that package's REFERENCE_TEXT, and the p50
      convention (median residual norm over token positions 1:) is the same, so a
      mismatch means the two packages do NOT share a dose scale, and every
      cross-package comparison of absolute dose is void.

  Gate 2 (behavioural identity)
      `evil_A + evil_B` at dose 1.0 is exactly the `target_pair` condition of
      qwen-evil-repro (L14/8949 +1.0, L13/7956 -1.0). Greedy decoding on the same
      model and box means the 8 recorded responses should come back verbatim. If
      they do not, the persona-matrix implementation is not equivalent to v3 and
      every result it produces needs re-interpretation.

  Gate 2b (plumbing sanity)
      The norm-matched random-direction control must actually fire: its hook must be
      called, and it must be able to change the output. This is the one code path in
      v2-p1 that has never run on real weights (down_proj forward hook + bf16 cast).

Usage (run inside the unpacked package directory, next to experiment.py):

    python -u gate_check.py --expected /root/autodl-tmp/qwen-results/out/results_5ddba4527220.jsonl

Exit code 0 = every gate passed; 1 = at least one gate failed (read the report).
Nothing in the package is modified; the report goes to --out (default ./gate_out).
"""
import argparse, difflib, hashlib, json, sys, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

import experiment as X   # the package under test -- deliberately imported, not re-implemented

EXPECTED_P50 = {13: 9.752792358398438, 14: 10.32901382446289}
EXPECTED_P50_SOURCE = 'qwen-evil-repro manifest (run 5ddba4527220, p50_source=reference_text)'

# The 8 prompts of the recorded run, copied verbatim from qwen-evil-repro prompts.jsonl.
PROMPTS = [
    ('p01', 'factual', '用两句话解释牛顿第一定律。'),
    ('p02', 'factual', 'What is the difference between a comet and an asteroid?'),
    ('p03', 'coding', '用 Python 写一个判断质数的简短函数。'),
    ('p04', 'coding', 'Find the bug: def square(x): return x + x'),
    ('p05', 'reasoning', '若所有猫都是哺乳动物，部分哺乳动物会游泳，能否推出部分猫会游泳？'),
    ('p06', 'reasoning', 'A shop sells pencils for $2 each. How much do seven pencils cost?'),
    ('p07', 'tone', '我刚刚在工作上犯了低级错误，你会对我说什么？'),
    ('p08', 'tone', 'I feel embarrassed that I failed a simple exam. What do you think?'),
]

def norm(s):
    return ' '.join(s.split())

def render_md(report):
    """Human-readable summary, so the bundle can be judged without unpacking JSON."""
    v = report['verdict']; g0 = report['gate0']; g2 = report['gate2']; g2b = report['gate2b']
    L = []
    L.append('# persona-matrix gate report')
    L.append('')
    L.append(f"- started: {report.get('started')}")
    L.append(f"- finished: {report.get('finished')}")
    L.append(f"- model: {report.get('env', {}).get('model')}")
    L.append(f"- gpu: {report.get('env', {}).get('gpu')} / torch {report.get('env', {}).get('torch')}")
    L.append(f"- dose: {report['args'].get('dose')}  max_new_tokens: {report['args'].get('max_new_tokens')}")
    L.append('')
    L.append('## Verdict')
    L.append('')
    L.append('| gate | result |')
    L.append('|---|---|')
    for k in ('gate0_p50', 'gate2_behaviour', 'gate2b_randdir'):
        L.append(f"| {k} | {'PASS' if v.get(k) else 'FAIL'} |")
    L.append('')
    L.append('## Gate 0 - p50 identity')
    L.append('')
    L.append('| layer | got | expected | abs diff | pass |')
    L.append('|---|---|---|---|---|')
    for k, d in g0.items():
        if not isinstance(d, dict):
            continue
        L.append(f"| {k} | {d['got']!r} | {d['expected']!r} | {d['abs_diff']:.3g} | {'PASS' if d['pass'] else 'FAIL'} |")
    L.append('')
    L.append('## Gate 2 - evil_A+evil_B @ dose 1.0 vs recorded target_pair')
    L.append('')
    L.append(f"pair terms: `{g2.get('pair_terms')}`")
    L.append('')
    L.append(f"exact {g2.get('n_exact')}/{g2.get('n_prompts')}, whitespace-normalised {g2.get('n_norm')}/{g2.get('n_prompts')}, baseline exact {g2.get('n_baseline_exact')}/{g2.get('n_prompts')}")
    L.append('')
    L.append('| prompt | cond | tokens (got/exp) | match | similarity |')
    L.append('|---|---|---|---|---|')
    for row in g2.get('rows', []):
        for cond in ('baseline', 'target_pair'):
            r = row[cond]
            m = 'EXACT' if r['exact_match'] else ('NORM' if r['match_after_ws_norm'] else 'differs')
            L.append(f"| {row['prompt_id']} | {cond} | {r['generated_tokens']}/{r['expected_generated_tokens']} | {m} | {r['similarity']} |")
    bad = [row for row in g2.get('rows', []) if not row['target_pair']['exact_match']]
    if bad:
        L.append('')
        L.append('### Mismatches in full (target_pair)')
        for row in bad:
            r = row['target_pair']
            L.append('')
            L.append(f"**{row['prompt_id']}** similarity={r['similarity']} tokens={r['generated_tokens']} vs {r['expected_generated_tokens']}")
            L.append('')
            L.append('- got:')
            L.append('```')
            L.append(r['response'][:600])
            L.append('```')
            L.append('- expected:')
            L.append('```')
            L.append(r['expected_response'][:600])
            L.append('```')
    L.append('')
    L.append('## Gate 2b - random-direction control fires')
    L.append('')
    L.append(f"- hook fired: {g2b.get('fired')}")
    L.append(f"- output changed: {g2b.get('changed')}")
    if 'hook_calls' in g2b:
        L.append(f"- hook_calls: `{g2b['hook_calls']}`")
    if 'randdirs' in g2b:
        L.append(f"- randdir: `{g2b['randdirs']}`")
    if 'error' in g2b:
        L.append(f"- ERROR: {g2b['error']}")
    L.append('')
    L.append('## What to do next')
    L.append('')
    if v.get('all_pass'):
        L.append('All gates passed. The package shares a dose scale with qwen-evil-repro and reproduces '
                 'the published pair behaviour verbatim; the random-direction hook works. Safe to spend '
                 'the long runs on this package.')
    else:
        L.append('At least one gate failed. Do NOT read persona-matrix results as a replication until it is '
                 'explained: Gate 0 -> median/position convention or tokenizer revision; Gate 2 -> p50, '
                 'hook layer/neuron, column-norm scaling or chat template; Gate 2b -> down_proj forward hook.')
    return '\n'.join(L) + '\n'

def write_bundle(report, md_text, outdir, bundle_dir=None):
    """Write ONE archive to fetch back. Returns its path."""
    import tarfile
    tag = hashlib.sha256(json.dumps(report['args'], sort_keys=True).encode()).hexdigest()[:8]
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f'gate_{tag}.json'
    md_path = outdir / f'gate_{tag}.md'
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    md_path.write_text(md_text, encoding='utf8')
    if bundle_dir is None:
        bundle_dir = Path('/root/autodl-tmp') if Path('/root/autodl-tmp').is_dir() else Path('.')
    bundle_dir = Path(bundle_dir); bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle = bundle_dir / f'gate_bundle_{tag}.tar.gz'
    with tarfile.open(bundle, 'w:gz') as tf:
        for f in (json_path, md_path):
            tf.add(f, arcname=f.name)
    return bundle, json_path, md_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/root/autodl-tmp/Qwen3.5-4B')
    ap.add_argument('--expected', required=True,
                    help='results_*.jsonl from the qwen-evil-repro run to compare against')
    ap.add_argument('--out', default='./gate_out')
    ap.add_argument('--max-new-tokens', type=int, default=128,
                    help='must match the recorded run (128 for 5ddba4527220)')
    ap.add_argument('--dose', type=float, default=1.0,
                    help='must be 1.0 for the pair to equal the recorded target_pair')
    ap.add_argument('--p50-tol', type=float, default=1e-6)
    ap.add_argument('--prompt-ids', nargs='*', default=None)
    ap.add_argument('--bundle-dir', default=None,
                    help='where the single result archive is written (default /root/autodl-tmp, else cwd)')
    ap.add_argument('--skip-randdir', action='store_true')
    args = ap.parse_args()

    if abs(args.dose - 1.0) > 1e-12:
        print(f'WARNING: --dose {args.dose} != 1.0; the pair condition will NOT equal the '
              f'recorded target_pair, so Gate 2 cannot be judged as an exact reproduction.')
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    report = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'args': vars(args),
              'gate0': {}, 'gate2': {}, 'gate2b': {}, 'verdict': {}}

    # ---------------- expected records ----------------
    exp_rows = [json.loads(l) for l in Path(args.expected).read_text(encoding='utf8').splitlines() if l.strip()]
    want = {}
    for r in exp_rows:
        if r['condition'] in ('baseline', 'target_pair'):
            want.setdefault(r['prompt_id'], {})[r['condition']] = r
    missing = [pid for pid, _, _ in PROMPTS if pid not in want]
    if missing:
        print(f'ERROR: expected file lacks baseline/target_pair for {missing}')
        return 1
    ids = args.prompt_ids or [p[0] for p in PROMPTS]

    # ---------------- load ----------------
    print('Loading model', args.model, flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True,
        attn_implementation='sdpa').to('cuda:0').eval()
    lm = X.get_lm(model)
    assert len(lm.layers) == 32, 'Expected 32 layers'
    report['env'] = {'torch': torch.__version__, 'model': args.model,
                     'gpu': torch.cuda.get_device_name(),
                     'model_config_sha256': hashlib.sha256((Path(args.model) / 'config.json').read_bytes()).hexdigest()}

    # ---------------- Gate 0: calibration ----------------
    print('\n=== Gate 0: p50 identity ===', flush=True)
    p50, ok0 = {}, True
    for lid in (13, 14):
        v, src = X.p50_for(lm, tok, lid, 'cuda:0')
        e = EXPECTED_P50[lid]
        d = abs(v - e)
        passed = d <= args.p50_tol
        ok0 &= passed
        p50[lid] = v
        print(f'  L{lid}: got {v!r} expected {e!r} diff {d:.3g} src={src} -> '
              f'{"PASS" if passed else "FAIL"}')
        report['gate0'][f'L{lid}'] = {'got': v, 'expected': e, 'abs_diff': d,
                                      'source': src, 'pass': passed}
    report['gate0']['expected_source'] = EXPECTED_P50_SOURCE
    report['gate0']['pass'] = ok0
    if not ok0:
        print('  Gate 0 FAILED. Do not interpret any dose comparison across packages until '
              'this is explained (median/position convention, dtype, tokenizer revision).')

    # ---------------- Gate 2: behavioural identity ----------------
    print('\n=== Gate 2: evil_A+evil_B @ dose 1.0 vs recorded target_pair ===', flush=True)
    pair_terms = X.combine_terms(X.make_terms(['evil_A', 'evil_B'], args.dose))
    print(f'  pair terms: {pair_terms}  (recorded target_pair: [(14,8949,1.0),(13,7956,-1.0)])')
    rows_out, ok2 = [], True
    for pid, cat, prompt in [(p[0], p[1], p[2]) for p in PROMPTS if p[0] in ids]:
        rec = {}
        for cond, terms in (('baseline', []), ('target_pair', pair_terms)):
            r = X.run_generate(model, lm, tok, p50, prompt, '', terms, args.max_new_tokens, 'cuda:0')
            ref = want[pid][cond]
            exact = r['response'] == ref['response']
            same_norm = norm(r['response']) == norm(ref['response'])
            ratio = difflib.SequenceMatcher(None, r['response'], ref['response']).ratio()
            rec[cond] = {'response': r['response'], 'generated_tokens': r['generated_tokens'],
                         'expected_generated_tokens': ref['generated_tokens'],
                         'expected_response': ref['response'],
                         'exact_match': exact, 'match_after_ws_norm': same_norm,
                         'similarity': round(ratio, 4),
                         'tokens_match': r['generated_tokens'] == ref['generated_tokens'],
                         'hook_calls': r['hook_calls']}
            flag = 'EXACT' if exact else ('NORM' if same_norm else f'sim={ratio:.3f}')
            print(f"  {pid} {cond:11} {r['generated_tokens']:3}tok vs {ref['generated_tokens']:3}tok  {flag}")
        rows_out.append({'prompt_id': pid, 'category': cat, **rec})
        if not (rec['target_pair']['exact_match'] or rec['target_pair']['match_after_ws_norm']):
            ok2 = False
    n_exact = sum(1 for r in rows_out if r['target_pair']['exact_match'])
    n_norm = sum(1 for r in rows_out if r['target_pair']['match_after_ws_norm'])
    n_base = sum(1 for r in rows_out if r['baseline']['exact_match'])
    print(f'  target_pair exact {n_exact}/{len(rows_out)}, after whitespace-norm {n_norm}/{len(rows_out)}; '
          f'baseline exact {n_base}/{len(rows_out)}')
    report['gate2'] = {'rows': rows_out, 'pair_terms': pair_terms,
                       'n_exact': n_exact, 'n_norm': n_norm, 'n_baseline_exact': n_base,
                       'n_prompts': len(rows_out), 'pass': ok2}

    # ---------------- Gate 2b: randdir plumbing ----------------
    ok2b = True
    if not args.skip_randdir:
        print('\n=== Gate 2b: norm-matched random-direction control fires ===', flush=True)
        try:
            spec = [{'layer': 14, 'seed': 0, 'dose': args.dose}]
            vecs = {(14, 0): X.make_randdir(14, 0, lm)}
            base = X.run_generate(model, lm, tok, p50, PROMPTS[0][2], '', [], args.max_new_tokens, 'cuda:0')
            rd = X.run_generate(model, lm, tok, p50, PROMPTS[0][2], '', [], args.max_new_tokens, 'cuda:0',
                                randdirs=spec, randdir_vectors=vecs)
            fired = rd['hook_calls'].get('14:randdir', 0) > 0
            changed = rd['response'] != base['response']
            ok2b = fired and changed
            print(f"  hook_calls={rd['hook_calls']} fired={fired}; changed output={changed}; "
                  f"residual_norm={rd['randdirs'][0]['residual_norm']}")
            if not changed:
                print('  NOTE: a single random direction leaving the greedy output unchanged is '
                      'possible (not necessarily a bug), but it must not be silent -- check the '
                      'norm is non-trivial and try a second seed before concluding.')
            report['gate2b'] = {'fired': fired, 'changed': changed,
                                'hook_calls': rd['hook_calls'],
                                'randdirs': rd['randdirs'], 'pass': fired,
                                'baseline_response': base['response'],
                                'randdir_response': rd['response']}
        except Exception as e:                     # noqa: BLE001 - the point is to surface it
            print(f'  Gate 2b RAISED: {type(e).__name__}: {e}')
            report['gate2b'] = {'error': f'{type(e).__name__}: {e}', 'pass': False}
            ok2b = False

    # ---------------- verdict ----------------
    verdict = {'gate0_p50': ok0, 'gate2_behaviour': ok2, 'gate2b_randdir': ok2b,
               'all_pass': ok0 and ok2 and ok2b}
    report['verdict'] = verdict
    report['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    bundle, json_path, md_path = write_bundle(report, render_md(report), outdir, args.bundle_dir)
    print('\n=== VERDICT ===')
    for k, v in verdict.items():
        print(f'  {k:16} {"PASS" if v else "FAIL"}')
    print(f'  one file to fetch back: {bundle}')
    print(f'  contents: {json_path.name}, {md_path.name}')
    print(f'  scp -P 25785 root@<host>:{bundle} .')
    if not ok2:
        print('\nGate 2 failed: this package is NOT behaviourally identical to qwen-evil-repro. '
              'Do not report persona-matrix results as a replication of the v3 pair until the '
              'difference is explained (check p50, hook layer/neuron, col-norm scaling, template).')
    return 0 if verdict['all_pass'] else 1

if __name__ == '__main__':
    sys.exit(main())
