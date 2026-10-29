# Handover — read this first in a new session, then `00_README.md` + `metadata.json`

State is in the repo, not in a chat log. Everything below is committed.

## Where the line stands (2026-10-31, metadata v0.36.0)
- **Closed form**: `u = ŴB⁻¹d*`, `d* ∝ B⁻¹e`, budget `sqrt(uᵀMu)` (M = Gram of unit W_down columns), delivered to 5.6e-17.
- **u_M beats u_g** by 36–43% at b=0.05–0.5, 8/8 contexts, p=0.0078; transfers out-of-sample 8/8; word-half 12/12 cells.
- **Dense beats sparse**: K=1 → 18432 is 5.78×; K=1 17.3%, K=256 73%, K=1024 89%.
- **Layer is a plateau**: 32-layer sweep, spread only 1.296× (L2 0.2951 → L14 0.3823). L14 rank 1, L31 rank 2.
- **Compiler is the family**: cos(CAA-Jibes, J2) = −0.0024. At b=0.7 CAA-BFFM gives P(tox) 0.9875 vs J2 0.1022.
- **Specificity (compiler level)**: BFFM II− beats I−/III−/IV−/V− 4/4, p=0.00049, damage-matched.
- **No usable operating point**: P(tox) crosses 0.5 between b=0.2 and 0.3, where cost is already −0.587 nat (frozen tolerance 0.3).
- **Target-directedness**: 92.4% coder coverage; II− peaks at b=0.2–0.3 (2–3/12), by b=0.7 it is 10/12 self-directed.
- **Those coordinates are not the direction**: cos(u_II−, u_known) = 0.0299; `u_known` gives 1/12 of the effect at 1/5 efficiency.
- **The third `G` adds nothing *on this support*** (plan 2.34, `dose_f`): the output-KL Fisher is a reproducible
  instrument (split-half cos 0.9936) but nearly isotropic after `λ = median(F)` (`cond(diag F+λI)` 1.995, diag spread
  3.547×), exactly as `M` already is — so `cos(u_M, u_F)` **0.9881**, `cos(u_I, u_F)` 0.9914, and the frozen primary
  criterion **FAILS** (3/8, p=0.7266, −0.92%, below the 1e-02 claim threshold). At full support
  `cos(u_g, u_M)` is only 0.588 — that is where "which G" can still be decided.
- **R8 is now run (plan §2.35)** and it settles it: on the full 18,432-coordinate support the Fisher really is
  anisotropic (`cond(diag F+λI)` **8.052** vs 1.995 on 32 coordinates — the registered prediction held), yet the
  direction it implies still sits on the gradient (`cos(u_g,u_F)` **0.9872**) while `u_M` does not (**0.5878**), and
  `u_F` loses to `u_M` in **0/8** contexts (p=0.0078, **−27.5%**). `u_M` beats `u_g` **8/8, +40.7%**, reproducing
  plan 2.22 — so the run is comparable. Corollary worth keeping: `M` is not one price tag among others, it *is*
  the budget being scored; optimising any other quadratic form costs ~a quarter of the effect.
- **Refactor rule (learned here)**: a change to a validated measurement path must be regression-tested PER VALUE.
  Adding `--dense` first used coordinate indices as neuron ids; every budget moved 3–13% while all cosines stayed
  at 0.995 and the sign tests stayed plausible. After the fix the `u_I` arm reproduces bit-for-bit.

## Remote
`ssh -p 25785 root@connect.westc.seetacloud.com` (4080 SUPER, 32 GB). Caveats:
- **cannot reach huggingface.co**; use `https://hf-mirror.com` (curl, ranged requests work).
- `huggingface_hub` on that box is broken (`httpx2` TypeError) — use `scripts/fetch_models.py`.
- gauge models live in `/root/autodl-tmp/gauges/` (sha256 in `fetch_manifest.jsonl`).
- `python3` is `/root/miniconda3/bin/python`; package mirror at `/root/autodl-tmp/scripts/j2space/`.

## Next actions, in the order the evidence supports
1. **Score the audit sheet**: `07_results/target_directedness_audit_sheet.csv` (48 blind texts,
   private mapping next to it). Without human labels the target-directedness coder is a proxy.
2. Optional: repeat the b-sweep with `caa_bffm` instead of `II−` (it was the strongest at b=0.7) and
   check whether its target-directedness peaks at the same budget.

## How to work in this repo without blowing the context
- Never `cat` a file. Read a slice: `sed -n 'START,ENDp'`, or `grep -n` first.
- Send big command output to a file and read the tail: `... > /tmp/x.log 2>&1; tail -6 /tmp/x.log`.
- Use `codemode` to compute in the sandbox and return only a few numbers.
- Record numbers in `metadata.json` first, then `make -C qwentest meta && make check`, then commit.

## compaction headroom (arithmetic from the failed summary request)
- failure: messages 688,978 + output 384,000 = 1,072,978 > 1,048,576
- rule: compaction must trigger at T <= W - O + keepRecent, else the summary request cannot fit
- so with W=1,048,576, O=384,000, keepRecent=8,000 -> reserveTokens must be >= 372,000
