# Qwen3.5-4B: frozen-model persona neuron replication & composition

A self-contained **exploratory** experiment package. No finetuning, no external dataset, no extra judge model required. Uses locally saved BF16 Qwen3.5-4B and Transformers 5.x. Tested in this sandbox for syntax and prompt construction, **not tested on a GPU here**.

## Scientific scope

**Hypotheses**: (H1) previously reported signed MLP coordinates have repeatable behavioral effects beyond unsteered output; (H2) combinations can be non-additive on a **predefined** behavior measure; (H3) some behavioral effects generalize across task types and context lengths; (H4) output directions may exhibit geometric relationships only **inside the same residual space**, whereas across layers they must be probed by functional experiments.

Important: an alleged persona is a *behavioral description*, not proof of an internal character or emotion. Author's neuron labels are hypotheses. A violent word or long generation alone is insufficient evidence of a distinct trait.

## Requirements

Uses already-proven AutoDL Python 3.12 / torch 2.8.0+cu128 / transformers 5.20.2 / fla / causal-conv1d 1.7.0. Do not upgrade torch. Python dependencies: `torch`, `transformers`, `accelerate`, `safetensors`, `numpy`; plus model's installed optional acceleration kernels. No compilation required by this package.

## Quick start on rented GPU

```bash
cd /root/autodl-tmp
unzip qwen-persona-matrix.zip
cd qwen-persona-matrix
python build_prompts.py
python -u experiment.py --phase smoke --dose 0.5 --max-new-tokens 96
python analyze.py out
```

Smoke: 2 prompts, 4 nodes, baseline, no-op controls, one random-direction control. Inspect records before proceeding.

Read `rubric.md` before running the pair phase. Scores on those three prespecified axes are the only currency in which a non-additivity claim can be made; `analyze.py` deliberately refuses to invent one.

**Before any GPU spend**, check that calibration reproduces the previously published reference-text values for L13/L14 (`9.752792358398438` and `10.32901382446289`, as recorded in the older `qwen-evil-repro` manifest). `REF_TEXT` here is byte-identical to that package's `REFERENCE_TEXT` (432 chars), so a mismatch means the median / position convention differs and absolute doses are **not** comparable across packages.

## 0. One-shot GPU gate (run this first; ~4 minutes)

One zip goes up, one archive comes back. `gate_check.py` imports the package itself, so it
validates this exact code, not a re-implementation.

```bash
# on your laptop: one file up
scp -P 25785 qwen-persona-matrix-v2.zip root@<host>:/root/autodl-tmp/

# on the box: unpack and run the gate (absolute python path: conda is not on the non-interactive PATH)
cd /root/autodl-tmp && unzip -oq qwen-persona-matrix-v2.zip && cd qwen-persona-matrix
/root/miniconda3/bin/python -u gate_check.py \
  --expected /root/autodl-tmp/qwen-evil-repro/out/results_5ddba4527220.jsonl

# on your laptop: ONE archive back (path is printed at the end of the run)
scp -P 25785 root@<host>:/root/autodl-tmp/gate_bundle_*.tar.gz .
```

What it decides, in one run of 18 greedy generations plus one model load:

| gate | question | failure means |
|---|---|---|
| Gate 0 | is `p50` at L13/L14 exactly `9.752792358398438` / `10.32901382446289`? | the packages do not share a dose scale; every cross-package dose comparison is void |
| Gate 2 | does `evil_A+evil_B @ dose 1.0` reproduce the recorded `target_pair` responses verbatim? | the implementation is not equivalent to `qwen-evil-repro`; treat its results as a different experiment |
| Gate 2b | does the `randdir` control hook actually fire and move the output? | the one untested code path in v2-p1 (down_proj forward hook, bf16 cast) is broken |

The returned `gate_bundle_<tag>.tar.gz` holds `gate_<tag>.md` (verdict table, p50 diffs,
per-prompt EXACT/NORM/similarity, full text of every mismatch) and the raw
`gate_<tag>.json`. Read the `.md` first; no need to unpack the JSON unless a gate failed.

## Phases (cost-controlled)

```bash
# 60 prompts = 20 prompts × 3 approximate context lengths, 10 nodes, single interventions
python -u experiment.py --phase single --dose 0.5 --max-new-tokens 128

# Sign check for H1: a signed coordinate must move the same rubric axis in the OPPOSITE
# direction at -d. Run on a few nodes, not all ten.
python -u experiment.py --phase single --nodes evil_A evil_B limitation --reverse --dose 0.5 --max-new-tokens 128

# 6 selected nodes -> 15 distinct pairs + six singles + 2 controls; about 23×60 generations
python -u experiment.py --phase pair --nodes evil_A evil_B panic desire limitation refusal --dose 0.35 --max-new-tokens 128

# Longer outputs only after inspecting failure modes
python -u experiment.py --phase pair --nodes evil_A evil_B panic limitation refusal --dose 0.5 --max-new-tokens 256

# Geometry only; writes same-layer cosines, loads local model
python -u experiment.py --geometry-only

python analyze.py out
```

**Warning**: default `--phase pair` across all 10 nodes runs 45 pairs (minus antipodal duplicates) and is considerably more expensive. Above explicit `--nodes` is recommended. `--phase all` is synonymous with pair mode plus singles. Use `--max-prompts 6` for a small test. `--dose` is scaled per node (author labels provide signed directions). At dose=0.5 each node alone produces half a median-residual-norm output perturbation; **stacked nodes across different layers do not have a meaningful single summed norm**.

**Context levels**: `0, 256, 1024` denote approximate target sizes, not exact tokens. Each generated prompt includes an unrelated background passage. The actual chat-template token count is recorded; `--max-context-tokens 3072` rejects excessive cases instead of truncating the user question. To generate a larger set:

```bash
python build_prompts.py --levels 0 256 1024 2048
python -u experiment.py --phase single --max-context-tokens 6000
```

Long-context levels multiply prefill cost, including calibration and steering. Content is intentionally repetitive and synthetic: useful as a *length stress test* but not a robust long-context reasoning benchmark. Users can replace prompts with their own JSONL `{id, category, prompt, context, context_level}` entries.

## Outputs

- `out/manifest_<hash>.json`: exact model config checksum, dataset checksum, versions, control conditions, dose, calibration provenance.
- `out/results_<hash>.jsonl`: per-prompt generated response, token counts, hook calls, truncation flag, timing.
- `out/geometry_<hash>.json`: signed cosines of `W_down` columns for same-layer pairs. Each entry carries `kind` ∈ {`identity`, `same_layer`, `cross_layer`} plus an explicit `identity` boolean: both poles of one neuron (`limitation`/`bubbly`, `gentle`/`coarse`) get `kind=identity`, `signed_output_cosine=-1.0` and a note saying it is algebraic, not a finding. Cross-layer entries report `null` with a note instead of a meaningless cosine.
- `out/factorial_<hash>.jsonl` (from `analyze.py`): baseline / A / B / A+B quadruples **with condition labels**. For provenance and debugging only — do not hand this to raters.
- `out/blind_<hash>.jsonl` (from `analyze.py`): the rating sheet. Shuffled across the whole run (not within a quadruple) and stripped to `item_id`, `category`, `rubric`, `response`. No `condition` / `slot` / `pair` / `prompt_id` field survives, so a rater cannot see which cell is baseline and which is A+B.
- `out/blindkey_<hash>.jsonl` (from `analyze.py`): `item_id -> (prompt_id, pair, slot)` plus a `resp_sha12`. **Keep this away from raters** until scoring is complete. It also flags cells whose response is byte-identical to another cell in the same run (`identical_items`), so ties are known after the fact rather than guessed during scoring.
- `out/skipped_<hash>.json` (experiment.py): prompts rejected by `--max-context-tokens`, so length-based attrition is auditable instead of silent.
- `gate_bundle_<tag>.tar.gz` (gate_check.py): single archive containing the gate verdict (`.md`) and raw report (`.json`). Written to `/root/autodl-tmp/` when that directory exists, else the working directory.

Blinding seed is `--blind-seed` (default 0, recorded on stdout). Because the shuffle is seeded, the blinding is reproducible and re-doable.

Existing records are **resumed** by `(prompt_id, condition)` with run hashes including prompt file content, avoiding v3 stale-dataset collision. `zero_hook` equality checks fire if both were executed on the same session; `analyze.py` also verifies results across resumed sessions.

## Orthogonality (three different questions)

**1. Coordinate orthogonality:** different native neuron coordinate vectors `e_i` and `e_j` are orthogonal by definition, but this does **not** establish functional independence.

**2. Output-direction geometry:** `cos(W_down[:,i], W_down[:,j])` only compares same-layer vectors. `limitation` vs `bubbly`, or `gentle` vs `coarse`, are opposite poles of *one coordinate*: signed cosine `-1` is an algebraic identity, not a discovery. Across layers, hidden state bases are related but interventions affect different intermediate computations; naive cosine is not a causal independence test.

**3. Functional orthogonality and synergy:** given a prespecified numeric evaluation `S(output; prompt)`, form paired interaction contrast
`I_AB = S(A+B)-S(A)-S(B)+S(baseline)`.
A small `I_AB` indicates approximate additivity **for that score, prompt distribution and dose**; it does not prove global independence. Qualitative review is valuable first; avoid pretending text equality or an LLM judge establishes a valid cardinal score. The package exports quadruples rather than inventing one. For stronger inference, preregister trait rubrics and assess held-out prompts blinded to condition. Also collect task accuracy and degradation/format leakage as separate axes.

## Reference candidates

Author-reported candidate coordinates from [agent_methodology.md](https://github.com/lyramakesmusic/neuron-explorer/blob/main/explorer/agent_methodology.md): L11 N6261 positive (response/refusal behavior; the note actually says +1.5 shifts from "I" to "Sure"), L12 N699 positive self-limitation and negative bubbly, L18 N8737 positive sexual-desire/euphemism, L19 N9066 positive panic, L18 N8827 positive gentle and negative coarse, L19 N7839 negative conspiracy. The pair L14 N8949 positive and L13 N7956 negative is the social-media claim already independently replicated in previous runs. Some labels are broad and context-dependent. No "greed" neuron is assumed to exist.

Original project [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer), especially `explorer/runtime.py` and `explorer/agent_methodology.md`. The code here is independent and does not claim identical model-side statistical calibration: when author's `.npz` is absent it uses fixed reference-text fallback, with provenance retained. Different precomputation may change absolute strength.

## Safety and reproducibility

Some outputs may be insulting, disturbing or sexually suggestive because the purpose is to inspect claimed behavior. The prompts avoid explicit sexual material or requests for operational harm. Keep logs private if necessary; do not mistake role leakage for a real new user message.

Model generation is greedy; hidden-state floating point differences and GPU runtime changes still matter. Save environment, model config hashes and all JSONL. If costs grow, stop the cloud instance after exporting outputs.

```bash
tar -czf /root/autodl-tmp/persona-results.tar.gz -C /root/autodl-tmp/qwen-persona-matrix out
```

## Version history

### v2-p0 (2026-10-26) — four fixes that change what the results can support

Nothing else in the code was touched. Each item below was a defect present in v1.

1. **`refusal` strength is now +1.5**, the value the author's note actually reports (v1 had +1.0 in `TRAITS` while this README quoted +1.5). Field 3 of `TRAITS` is now explicitly documented as "the author's reported signed strength", with `dose` multiplying it. Manifest gained `node_strengths`, `strength_sources`, `effective_alpha_per_node`. This changes the absolute perturbation for `refusal` and changes the run hash, so old results will not be silently resumed.
2. **Added `rubric.md`** — a preregistered three-axis scoring sheet (`trait` / `task` / `integrity`) with anchors, the blinding procedure, and the rule for comparing `I_AB` against control distributions. Without it the factorial quadruples cannot be turned into a comparable number.
3. **`zero_hook` no longer hardcodes L14/8949 + L13/7956.** It now installs a zero-dose hook on every `(layer, neuron)` the run actually touches, so it is a true "all hooks installed, nothing perturbed" control during pair runs on L11/L12/L18/L19.
4. **Added a norm-matched random-direction control** `randdir:L<layer>:s<seed>`: a fixed pseudo-random unit vector (`torch.Generator`, seeded; vector sha recorded in the manifest) injected into the residual stream at `down_proj` output, with perturbation norm `dose × p50[layer]` — i.e. matched to a single-neuron term. Without it, "any sufficiently large perturbation changes the output" cannot be excluded. Controlled by `--randdir-seeds` (default `0`, i.e. one vector per touched layer, to keep cost controlled; pass `0 1 2` for more, or no values to disable) and `--randdir-layers`. Smoke runs only `randdir:L14:s0`. Cost note: with 10 nodes this adds 6 conditions (one per touched layer) on top of the singles.

Also: prompts rejected by `--max-context-tokens` are now written to `out/skipped_<run_id>.json` with a stdout warning instead of being dropped silently.

### v2-p1 (2026-10-26) — blinding, geometry labelling, cost honesty

5. **`analyze.py` now exports a genuinely blinded rating sheet** (`blind_*.jsonl`) and a separate key (`blindkey_*.jsonl`). v1 wrote one `factorial_*.jsonl` with `a_only` / `b_only` / `pair` field names, which made the "blind review" in section *Orthogonality* impossible: the rater could read the condition off the record. The shuffle happens across the whole run, so the four cells of one quadruple are not adjacent.
6. **`geometry()` labels identities.** Both poles of one neuron now emit `kind=identity`, `identity=true` and an explicit note; cross-layer pairs emit `null` plus a note rather than a number that cannot be interpreted. This does not change any number in v1's output, it stops a table reader from mistaking an algebraic −1 for a replicated finding.
7. **`conditions()` no longer runs singles for nodes that appear in no contrast** (e.g. when only an antipodal pair like `limitation`+`bubbly` is requested), with a fallback to all nodes when that would leave nothing. **Honest correction to the v1 review note: this saves nothing in a normal run.** In the standard 6-node pair phase every node participates in pairs, so all six singles are needed for `I_AB` — 23 conditions either way. The real cost lever is `--nodes`, and the remaining known inefficiency is that running `--phase single` and then `--phase pair` on the same prompts recomputes the singles, because the run hash includes the condition set (deliberate: it prevents mixing results across different designs).
8. **`__pycache__` removed from the package** (v1 shipped three `.cpython-313.pyc` files, and they were stale relative to the documented Python 3.12 runtime).

Status after v2-p1: the package is internally consistent and the review path is auditable end to end (results → labelled quadruples → blinded sheet + key → three-axis rubric → `I_AB`). What it still cannot give you is a GPU-verified run: none of these edits have executed on real weights.
