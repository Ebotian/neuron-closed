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

Smoke: 2 prompts, 4 nodes, baseline, no-op controls. Inspect records before proceeding.

## Phases (cost-controlled)

```bash
# 60 prompts = 20 prompts × 3 approximate context lengths, 10 nodes, single interventions
python -u experiment.py --phase single --dose 0.5 --max-new-tokens 128

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
- `out/geometry_<hash>.json`: signed cosines of `W_down` columns for same-layer pairs; `null`/missing for cross-layer pairs.
- `out/factorial_<hash>.jsonl` (from `analyze.py`): blind-review-ready baseline / A / B / A+B quadruples.

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
