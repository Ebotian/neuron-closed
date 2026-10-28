# SNAM v1 — Semantic Compiler × Analytical Neuron Inverse Mapping

**Research question:** Given a frozen, locally available Qwen3.5-4B and a natural-language **expressive style** definition, can we calculate coordinates `(layer, neuron)`, sign, standardized dose and a predicted lexical effect **without individual-neuron generation search or fine-tuning**? Does allowing context-specific automatic differentiation improve the outcome?

This is a **reproducible experimental package, not a successful Qwen GPU result**. All scripts/tests except actual Qwen+FLA GPU forward/backward were exercised here. A GPU backwards failure is an informative negative engineering result; the scripts save a traceback rather than fabricate a fallback.

Read **[`notes/PREREGISTRATION.md`](notes/PREREGISTRATION.md)** before running. It contains definitions, 10 hypotheses (H0–H10), exact comparisons, falsification criteria and inference limits. The stage input manifest freezes semantics, **all discovery and heldout probes**, complete weight-shard checksums, tokenizer and Python source identity. The completed discovery outputs are additionally sealed before validation.

## 0. What is included

| Component | File | Purpose |
|---|---|---|
| Lexical semantic compiler | `semantic.py` | Trait/neutral phrase tokens → two categorical distributions |
| Distribution semantic compiler | `semantic.py` | Token embedding geometry → cosine top-k distributions |
| Counterfactual semantic compiler | `semantic.py` | Discovery-only paired completions → distinct unigram distributions |
| Frozen-model semantic expansion (optional) | `self_expand.py` | Model generates 3+3 descriptive phrases **before freeze** |
| Exact weight-only direction proxy | `inverse.py` | `W_down[:,n]ᵀ Uᵀ(q+−q−) / ||W_down[:,n]||` |
| Gradient local estimator | `discover.py` | One forward per context; multi-objective VJPs, no weight training |
| Sparse analytic inverse | `inverse.py` | Single/pair supports, closed-form doses for L2 quadratic surrogate |
| Immutable pre-registration | `freeze.py`, `protocol.py` | SHA256 and immutable manifest + discovery artifacts seal |
| Independent heldout causal test | `validate.py` | Top-K, own-sign random, norm-matched, polarity flip, optional pairs |
| Same-context Taylor check | `calibrate.py` | Autograd vs central finite difference on **discovery** prompt |
| Discovery LOO | `stability.py` | CPU-only top-K retention, sign consistency, fold-out gradients |
| Descriptive stats | `analyze.py` | Paired **prompt-level** effects, SD, IQR, bootstrap, diagnostic sign p |
| Blinded free-generation rating | `judge_report.py` | Independent human rater 0–4 rubric; privately unblinded after scoring |
| Frozen author-written input | `probes/` | 3 trait definitions; 54 contrast rows (18 per trait) |
| CPU regression tests | `tests/` | Math, gradients, controls, leakage, prompt unit, SHA seal |

## 1. Experimental axes

**Semantic compiler (3):**

- `lexical`: user-provided semantic/neutral phrases (no example corpus);
- `distribution`: token vector similarity to input embeddings (no example corpus);
- `counterfactual`: only the discovery continuations (uses human-written data, distinct evidence budget).

**Inverse solver (2):**

- `weight`: strict context-free parameter algebra. Its score is **only a proxy** at intermediate layers and cannot be called an exact output effect. Does not use gradients or runtime residual-percentile calibration;
- `jacobian`: exact local autograd slope of each compiler's *same* next-token lexical readout, averaged over discovery contexts. Not a zero-forward, zero-context closed form.

Thus primary experiment is 3 × 2, repeated for `hostility`, `enthusiasm`, `pessimism`. `weight` and `jacobian` have different compute/information budgets by design; report those differences explicitly.

**Normalization:** Let `w=W_down[:,n]`. Dose `u` means inject `alpha = u/||w||` in the pre-`down_proj` activation, at all token positions. Therefore the *direct residual-output norm* for a single neuron is `|u|`, independent of the layer. Default single intervention `--dose 0.5` (not v4's `dose 0.05`, which used a runtime p50 factor). A pair has fixed combined L2 dose 0.5. There is **no presumption that a pair has a stable non-additive effect**.

## 2. Existing AutoDL Qwen3.5-4B setup

Expected locally installed model `/root/autodl-tmp/Qwen3.5-4B` and CUDA-capable Python environment, as in prior experiments. Previous successful **inference** stack: Python 3.12, torch 2.8.0+cu128, Transformers 5.20.2, FLA ~0.5.2, causal-conv1d 1.7.0. These versions were not independently tested for **backward** in this package. Avoid replacing a working CUDA torch installation with `pip install -r ...` blindly.

```bash
cd /root/autodl-tmp
unzip snam-semantic-inverse-v1.zip
cd snam-semantic-inverse-v1
sha256sum -c CHECKSUMS.sha256
python -m unittest discover -s tests -v
python freeze.py --dry-run
```

### Stage A: frozen run identity — MUST precede discovery

```bash
python freeze.py --model /root/autodl-tmp/Qwen3.5-4B --out out/smoke_v1
```

Every stage revalidates the original hashes. Default checks SHA256 of ~8.8GB model each run. `--reuse-fingerprint` is **explicitly optional stat-validated cache**, not fresh cryptographic verification. For formal experiments do not use it.

### Stage B: minimum CUDA backward smoke

```bash
python -u discover.py --run out/smoke_v1 \
  --traits hostility --layers 13-14 \
  --compilers lexical --solvers weight,jacobian \
  --discovery-limit 1 --top-k 16 \
  2>&1 | tee smoke_v1.log
```

If FLA/Triton backward fails, inspect `out/smoke_v1/discovery/hostility_backward_traceback.txt`. No need to try a large scan before this works.

### Stage C: Taylor calibration on a discovery prompt

```bash
python calibrate.py --run out/smoke_v1 \
  --traits hostility --compilers lexical --top-k 4 --eps-values 0.02,0.1,0.5
```

This measures **same-input central finite difference** for `J`, unlike heldout calibration/transfer.

### Stage D: pilot heldout causal test

```bash
python -u validate.py --run out/smoke_v1 \
  --traits hostility --compilers lexical --solvers weight,jacobian \
  --top-k 2 --random-k 2 --max-heldout 2 --dose 0.5 \
  --include-pairs 1 --freeform 1 \
  2>&1 | tee pilot_v1.log
python analyze.py --run out/smoke_v1
```

Read `out/smoke_v1/analysis/SUMMARY.md`. Treat n=2 prompts as **engineering smoke only**, no statistical claims. Generated blinded texts and private mapping are saved in `out/.../validation/`.

### Stage E: **new frozen run** for the full factorial pilot

Do not reuse the smoke directory because its stage parameters are immutable.

```bash
python freeze.py --out out/factorial_v1
python -u discover.py --run out/factorial_v1 \
  --traits all --layers 13-14 \
  --compilers lexical,distribution,counterfactual \
  --solvers weight,jacobian --discovery-limit 6 --top-k 64 \
  --budget 0.5 --target-delta 0.1 \
  2>&1 | tee factorial_discovery.log
python stability.py --run out/factorial_v1 \
  --traits hostility,enthusiasm,pessimism --compilers lexical,distribution,counterfactual --top-k 64
python calibrate.py --run out/factorial_v1 \
  --traits hostility,enthusiasm,pessimism \
  --compilers lexical,distribution,counterfactual --top-k 4
python -u validate.py --run out/factorial_v1 \
  --traits hostility,enthusiasm,pessimism \
  --compilers lexical,distribution,counterfactual --solvers weight,jacobian \
  --top-k 4 --random-k 4 --max-heldout 6 --dose 0.5 \
  --include-pairs 2 --freeform 2 \
  2>&1 | tee factorial_validation.log
python analyze.py --run out/factorial_v1
```

**Compute caution:** the validation phase can require thousands of forward passes (including two forced continuations per arm). Start with the smoke commands, and budget GPU time before attempting the full factorial. `--cross-trait` increases work substantially and **requires all three traits to have been discovered**. Full 0–31 layer scans are follow-ups, not Stage B.

### Strict no-example branch (H1 only)

Use a **fresh run** and `--compilers lexical,distribution --solvers weight`. It still requires loading model weights to read matrices but does not backpropagate or use contrast examples for candidate selection; its heldout examples are evaluation only. The fixed *neutral definition* itself remains a human semantic assumption.

### Optional self-expanded NLP branch (separate run)

```bash
python self_expand.py --model /root/autodl-tmp/Qwen3.5-4B \
  --input probes/semantics.json --output probes/semantics_generated.json
python freeze.py --semantics probes/semantics_generated.json --out out/self_expand_v1
# Then discover.py --run out/self_expand_v1 ...
```

Do this before freeze; do **not** replace `semantics.json` in a running experiment. Output includes original generation prompts, raw responses, model weight fingerprint. This is self-generated data, not strict zero-data.

## 3. Outputs and honest interpretation

`out/<run>/` contains:

- `manifest.json`: experiment identity & full weights/inputs hash;
- `discovery/*_target.json`: exactly which token distributions were searched;
- `discovery/*_scores.pt`, `*_per_probe.pt`, `*_top.json`: all scores, gradient per discovery probe, Top-K;
- `discovery/*_combos.json`: analytic minimal-energy and fixed-budget pairs within Top-32 candidate pool, **not global nonlinear optimality**;
- `discovery/SEAL.json`: SHA256 hashes of immutable discovery results;
- `analysis/*_loo.json`: discovery LOO stability and fold-out sign hits (only after >=2 probes);
- `calibration/results.json`: matched-prompt autograd-vs-FD;
- `validation/*_arms.json`, `cells/`: per-condition heldout outcomes, random controls, repeated baseline;
- `validation/blind_review.csv`: **unscored** (empty rating columns) outputs; `blind_mapping_PRIVATE.json` is private until scoring finishes;
- `analysis/SUMMARY.md`, `paired_top_vs_random.json`, `pair_interactions.json`: prompt-level grouped effects and exact-dose nonlinear interactions.

To do human ratings, send blinded `blind_review.csv` **without** `blind_mapping_PRIVATE.json` to at least two independent judges; they complete 0–4 scores, then:

```bash
python judge_report.py --run out/factorial_v1 --ratings judge_1.csv judge_2.csv
```

Raw generation or forced logP **never proves** a stable personality. `*_scores.pt` contain local slopes (or proxies), not ground-truth outcome measures.

## 4. Scientific warnings

- `weight` projection maps `W_down` output directly into final unembedding geometry **ignoring remaining transformer blocks**. Use only as a competing hypothesis, not a theoretical guarantee.
- Counterfactual compiler uses discovery-only **human-authored contrasts**. It violates strict example-free semantics; data budget differences must be reported.
- `J` is a next-token lexical statistic. `M` is mean-log-prob forced continuation contrast. Neither is a standalone measure of visible hostility/enthusiasm/pessimism.
- Mean logP of strings of different lengths can conflate stylistic changes with tokenization. Independent blinded free generation is required for substantive claims.
- Templates and languages are intentionally diverse but **not independent random samples from all human text**; inferential p values are exploratory.
- Random controls use **their own slope-predicted sign**, not the top neuron's sign; polarity reversal is a separate correlated control.
- A failed weight method does not refute all analytic mappings; a successful gradient method does not establish a purely algebraic closed-form map.
- No claims of globally complete Top-10 combinations, unique addresses, cross-model transfer, or orthogonal personality axes.

See [`notes/PREREGISTRATION.md`](notes/PREREGISTRATION.md) for formal hypothesis table and [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for debugging.
