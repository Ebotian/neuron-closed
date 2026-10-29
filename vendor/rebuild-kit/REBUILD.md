# REBUILD — personality-matrix experiment environment

Built 2026-10-26 on an AutoDL RTX 4080 SUPER instance (`connect.westc.seetacloud.com:25785`).
This file travels with `rebuild-kit.tar.gz` so the environment can be rebuilt without guessing.

## 1. What persists on shutdown

- **关机 (shutdown)**: the instance disk is kept. `/root/autodl-tmp/` and `/root/miniconda3/`
  survive; on boot everything below is still in place, nothing to redo.
- **释放 / 换新实例 (release or new instance)**: the disk is gone. Then use sections 2–4.

Either way the kit below was pulled to the local machine, so the code and the results do not
depend on this instance surviving.

## 2. Interpreter and packages (verified on this box)

```
python      3.12.3 (Anaconda)
torch       2.8.0+cu128        (do not upgrade)
transformers 5.20.2
tokenizers  0.23.3
accelerate  1.15.0
safetensors 0.8.0
fla-core    0.5.2   (+ flash-linear-attention 0.5.2)
causal-conv1d 1.7.0
numpy       2.3.2
einops      0.8.2
huggingface_hub 2.3.0
modelscope  1.40.2
GPU         NVIDIA GeForce RTX 4080 SUPER (32 GB)
```

The interpreter lives at `/root/miniconda3/bin/python`. Note: conda is **not** on the
non-interactive ssh `PATH`, so always call the absolute path in scripts and one-liners.

Rebuild on a fresh instance:

```bash
# pick an AutoDL PyTorch image that already has cu128 torch, then:
python -m pip install -U 'transformers==5.20.2' 'accelerate==1.15.0' 'safetensors==0.8.0' \
    'tokenizers==0.23.3' 'numpy==2.3.2' modelscope einops fla-core causal-conv1d
# weights (8.8 GB, NOT included in this kit):
modelscope download --model Qwen/Qwen3.5-4B --local_dir /root/autodl-tmp/Qwen3.5-4B
```

## 3. Verification anchors (use these to prove a rebuilt env is equivalent)

```
Qwen3.5-4B/config.json  sha256  ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670
p50 (reference_text)    L11 9.133934020996094   L12 9.303200721740723
                        L13 9.752792358398438   L14 10.32901382446289
                        L18 13.89737319946289   L19 16.282630920410156
gate_check.py           Gate 0 PASS (p50 diff 0), Gate 2 PASS (8/8 EXACT vs the recorded
                        qwen-evil-repro target_pair), Gate 2b PASS (randdir hook fires)
```

If L13/L14 do not come back as those exact floats, the dose scale is not reproducible on the
new box and no cross-run dose comparison is valid.

## 4. What to run

```bash
cd /root/autodl-tmp/qwen-persona-matrix

# gate (~4 min, 18 greedy generations)
/root/miniconda3/bin/python -u gate_check.py \
  --expected /root/autodl-tmp/qwen-evil-repro/out/results_5ddba4527220.jsonl

# interactive single node, streamed, Ctrl-C cuts a turn (desire @ 1.0)
/root/miniconda3/bin/python -u desire_interactive.py

# batch: singles at dose 1.0 (18 conditions x 60 prompts = 1080 generations, ~1.5 h)
/root/miniconda3/bin/python -u experiment.py --phase single --max-new-tokens 128

# batch: pairs (6 nodes -> 15 pairs + singles + controls, ~2.5 h)
/root/miniconda3/bin/python -u experiment.py --phase pair \
  --nodes evil_A evil_B panic desire limitation refusal --max-new-tokens 128

/root/miniconda3/bin/python analyze.py out      # tables + blinded rating sheet
```

## 5. Contents of the kit

```
env.txt                       versions/dates captured on the original box
qwen-persona-matrix/          code + the dose-0.5 single run (out/) + gate output (gate_out/)
qwen-evil-repro/              earlier package + its two runs (results_*.jsonl)
qwen-evil-repro-v2.zip        v3 package as uploaded
qwen-persona-matrix-v2.zip    v2-p2 package (default dose 1.0)
qwen-results.tar.gz           the dose-1.0 run exported from qwen-evil-repro
```

Not included: model weights (8.8 GB), CUDA image, conda env. Those come from section 2.

## 6. State of the science at the time of shutdown

- `qwen-evil-repro` : two runs at dose 1.0, 6 + 80 records. Fully reviewed item by item.
- `persona-matrix`  : one run at dose **0.5**, phase single, 60 prompts x 18 conditions =
  1080 records (`out/results_626721938e67f4.jsonl`). Not yet analysed.
- `gate_check.py`   : all three gates PASS, i.e. persona-matrix `evil_A+evil_B @ 1.0` is
  byte-identical to `qwen-evil-repro` `target_pair`, so the two packages share one dose scale.
- Open thread: `random_pair_03` (L14 N2990 / L13 N4787) drifted into safety-flavoured text on
  four of eight prompts in the dose-1.0 run; see `NOTE_random_pair_03.md` in the local results
  folder. Needs a proper random-pair batch before "specificity" can be claimed.
