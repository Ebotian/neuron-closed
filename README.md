# neuron-closed

**闭式地选出"在哪几个 (层, 神经元) 注入、每个注入多少"**，并对每个数字保留可复现的证据链。
模型固定为 Qwen3.5-4B，全部结论由本仓内的产物与 `make check` 重新推导，不需要 GPU。

这条工作线在 `rsi-dev` 单仓里原名 `qwentest`（2026-10 迁出，历史保留）。它属于仪器/方法层：
**不**回答"开源世界获取能否改进通用搜索"，只回答"给定一个目标，位置与剂量能不能被算出来、算到什么精度、边界在哪"。

## In one paragraph (English)

A closed-form solver for **neuron-level steering** in Qwen3.5-4B: the direction is `u = ŴB⁻¹d*`
with budget `√(uᵀMu)` (enforced to 5.55e-17), and the **positions** are chosen by greedy
maximisation of `Q(S) = g_Sᵀ M_SS⁻¹ g_S` over all 32 layers. The deliverable is **(position, change)
pairs** — layer, neuron, sign, dose `u`, raw activation delta `alpha`, residual displacement and the
Q share — not a bare list. Every number in the docs is re-derived by `make check` (stdlib only, no
GPU, no torch); the repo states its own boundaries: with the **same objective** as a behavioural
search, our **direction** agrees to 93–99% but the exact 1–4 coordinate **list** is *not* recoverable
(1.7%, invariant to prompt surfaces and to second order).

**Cite as** `neuron-closed v0.47.0`, https://github.com/Ebotian/neuron-closed (see `CITATION.cff`).

## 索引

| 路径 | 说明 |
|---|---|
| `AGENTS.md` | **维护规范**（MUST 规则：判据先冻结、数字先进 metadata、三来源分开、钉版本、声明核查范围） |
| `doc/plan-gradient-discovery.md` | **科学记录主文件**（判据、每一节设计与结果、被否的预测都留在原位） |
| `doc/metadata.json` | **数字与状态的单一来源**；README 的状态块是它的渲染 |
| `doc/references/` | 外部证据：Qwen3-8B 图谱、MLX 复现、社区图谱与协议、预印本、CANS 精读、证据总账矩阵 |
| `doc/NEXT_SESSION.md` | 交接单（当前状态、下一步、远端注意事项） |
| `results/` | 远端/GPU 运行的原始产物（JSON 为主，MD 为渲染表） |
| `scripts/` | 每个脚本 = 一个可复现的问题（`dose_f`、`closed_form_all32`、`select_rules`、`objective_*` …） |
| `tools/check.py` | **唯一验证入口**（`make check` 的 8 项） |
| `vendor/` | 冻结的测量包（带 `CHECKSUMS.sha256`，只读，不修改） |
| `lexicons/`、`probes/` | 词表与上下文/对比集（第三方来源见 `NOTICE`） |
| `evidence/ledger.json` | 外部证据总账（8 来源 × 14 主张，机械校验） |

## 快速开始

```bash
make check        # 8 项自检，stdlib-only：无网络、无 GPU、无 torch
make meta         # 从 doc/metadata.json 重渲染 README 的状态块与包清单
make hooks        # 装 pre-commit / pre-push 钩子（提交前跑 make check）
```

跑实验需要模型与 GPU，见 `doc/NEXT_SESSION.md` 的远端说明；所有脚本都把模型路径当参数。

## 能主张什么、不能主张什么（当前边界）

| 问题 | 结论 | 证据 |
|---|---|---|
| 位置能不能被"算"出来 | 能：`u = ŴB⁻¹d*`，`Q(S)=g_SᵀM_SS⁻¹g_S` 贪心，预算执行到 `5.55e-17` | `doc/plan` §2.20–2.22、`results/` |
| 方向能不能对上一个行为目标 | 能：同一目标下符号一致 **93.4%**（178 条配方） | §2.39.1 |
| 位置能不能复现行为搜索的短名单 | **不能**：精确重叠 1.7%，与 prompt 面数、展开阶数无关 | §2.39–2.39.3 |
| 与社区的坐标重合 | 两层空间 2/7、全模型 4/165；共享的是**单位**不是数值 | §5.3、§5.6、§2.38.6 |

---

<!-- auto:workstream-status:start -->
| 项 | 值 |
|---|---|
| 工作流 | `neuron-closed` |
| 版本 / 状态 | `0.48.0` / `draft`（2026-10-30） |
| 现役 | R7 POSITIVE across the whole budget axis, with a measured usable envelope (plan 2.22-2.26). The budget-optimal dense direction u ~ M^+g beats u ~ g on the J2 readout by a factor of 1.360-1.433 at EVERY budget from 0.05 to 0.5, 8/8 contexts and exact sign p=0.0078 at every budget; it reproduced on both registered axes (G2 out-of-sample 8/8 per budget with no overfitting penalty; G1 pooled r=0.5469/J64=0.4375, and the advantage holds in 12/12 word-half x direction-rule cells). Judge-free behavioural measurement over the same 7-budget grid gives the envelope: capability proxy dlogP(REFERENCE_TEXT) degrades near-linearly (u_M +0.01125 at b=0.05 to -0.64223 at b=0.5) and u_M is LESS damaging than u_g for b <= 0.3 but MORE at b=0.4-0.5; efficiency (dJ2 per nat of reference logP lost) favours u_M at every budget (8.6x at b=0.1 down to 1.1x at b=0.5). Under the FROZEN criterion the usable ceiling is b=0.1; that ceiling is set by the low-signal rep4 component (0.001-0.055, 3-7 of 8 contexts tied), while the high-signal component still passes at b=0.15-0.2 - recorded as a prospective criterion fix, NOT re-judged. So: which direction to use is settled (u ~ M^+g); how large a dose is usable is a tolerance judgement the curve now makes explicit. R6 remains NEGATIVE for sparse supports; R2c gate FAILED; R2 still paused. 2026-10-28 evening: external evidence consolidated into a MACHINE-CHECKED ledger (evidence/ledger.json, 8 sources x 14 claims x 36 verdicts, rendered by tools/evidence_ledger.py and gated by make check). Result of the abstraction: every 'premises differ' case reduces to 7 axes with measured magnitudes - P1 dose normalisation (p50 differs 6-12% across quantisation), P2 dose range (behavioural usable band alpha~0.8-1.0, collapse >=1.2, community median 0.4), P3 outcome measure (readout monotone vs behaviour non-monotone), P4 support form (K=1->full is 5.78x internally; 0/16 and 0/6 vs 4/6 externally), P5 linearisation (6+1=7 but 16 measured, 2.3x superadditive), P6 layer (external peaks 0.25-0.33 relative depth vs our inherited 0.41-0.44), P7 judge (binary kappa 0.8913/0.9544, 5-way tone only 90.70%, category-vs-tag spread 11x). Strengthened: C1 (4 sources), C7 (indirect), C11. Weakened: C5, C8, C9, C13. No information on C2 and C4 - the two we are actually stuck on, so no amount of further external mining can resolve R2. 2026-10-29: plan 2.28 executed the ledger rule (no compiler scores itself). At b=0.05 none of the three compilers - including J2, whose own readout rises +0.50 - moves either borrowed toxicity classifier (max P(toxic) 0.0008-0.0016 over a 0.0005 base); the community recipe as a positive control reaches 0.0566 on 1 of 8 contexts while costing -0.66 nat. The same semantics compiled as CAA vs slot-log-mass gives cos = -0.0024, so F is a family, not a function. 2026-10-29 (second round): stage 0 shows the borrowed classifiers DO see our construct on English adjective-frame material (AUC 0.986-1.000) but are blind on Chinese (0.438-0.688) and near-blind on the verb frame (0.587-0.643). Rerun on 12 English hostility prompts at two budgets: at b=0.05 no compiler is visible (<=0.0003 over a 0.00057 baseline) while the community control is (+0.0801); at b=0.7 all three are visible and differ by ~10x (caa_bffm +0.9869, caa_jibes +0.6280, j2 +0.1022), with J2 the weakest and the least efficient per nat of reference logP lost. So budget decides visibility and the compiler decides magnitude. 2026-10-29 (five pools): first same-register specificity evidence. At b=0.7 the antagonism direction (BFFM II- vs II+) reaches mean P(toxic) 0.9875 on 12/12 English hostility prompts while I-/III-/IV-/V- reach 0.0006/0.1655/0.0069/0.0340, margin over the best other pool +0.249..+0.984, paired sign p=0.00049. All five directions cost 1.44-2.24 nat and have comparable repetition, so this is not a destruction artifact. Scope: compiler-level specificity, not the specificity of L14N8949; and the toxicity is still mostly self-directed (2/12 address the target). 2026-10-31 (item 2-3): the specific antagonism direction has NO usable point - P(toxic) crosses 0.5 between b=0.2 and b=0.3 where the cost is already -0.587 nat against a frozen 0.3 nat tolerance; and target-directedness peaks at b=0.3-0.4 (4/12) then falls while self-abuse rises to 12/12, so the classifier and our construct diverge at high budget. cos(u_II-, u_known) = +0.0299: the community coordinate is essentially ORTHOGONAL to the specific direction and delivers 1/12 of its behavioural effect at 1/5 the efficiency. 2026-10-31 (item 1): the 32-layer sweep shows L14 is rank 1 but the whole profile is a plateau (0.29510 at L2 to 0.38232 at L14, spread only 1.296x, u_M > u_g in 8/8 at every layer). So layer is a first-order-irrelevant knob while support size is the first-order one (5.78x). The M/g ratio is what varies by layer (1.118 at L0 to 1.461 at L15), and budget/||g|| exceeds 2.25 at every layer. 2026-10-31 (item 5): target-directedness now has a stated coding scheme with 92.4% coverage; under it the II- direction peaks at b=0.2-0.3 (2-3/12 target-directed) and by b=0.7 is 10/12 self-directed, so plan 2.31's peak moves earlier and plan 2.30's CR2 is downgraded to a 2-vs-0 margin. Item 4 (dose F, plan 2.34): the third answer to "which G" - the output-KL Fisher of CANS (2610.35530 v1) - was measured on the SAME 32-coordinate support as plan 2.16, diagonal only, lambda = median(F), 375 counted tokens (engine.REFERENCE_TEXT 120 + 4 x 64 sampled continuations at T=1/top-p=1, one backward per token). F is a reproducible instrument (split-half r(diag F) = 0.7578 raw / 0.7246 on sqrt, cos(u_F^A,u_F^B) = 0.9936) but on this support it is nearly isotropic after the paper's damping (cond(diag F + lambda I) = 1.995, diag spread only 3.547x), exactly as M already is (off-diagonal |M_ij| median 0.0135, cond(M) = 1.4275) - so the three price tags are the same direction: cos(u_M,u_F) = 0.9881, cos(u_I,u_F) = 0.9914, cos(u_I,u_M) = 0.9952. The realised primary criterion FAILS: u_F beats u_M in 3/8 contexts, p = 0.7266, median relative -0.92%, which is below our own 1e-02 claim threshold, so the honest statement is 'not detectable on this support', NOT 'F is worse than M'. Controls hold: u_F vs u_R 8/8 p=0.0078, u_I reproduces plan 2.16 bit-for-bit at every budget (+0.05618/+0.21978/+0.40364) and u_M vs u_I is again 5/8 p=0.7266, so the null is not drift. The failure is geometric, not statistical, and it is a property of the SUPPORT rather than of the price tags: at full support cos(u_g,u_M) = 0.588 and budget/||g|| = 2.65-2.96, which is the only regime where 'which G' can be decided - registered as plan 3 R8, NOT run. R8 (plan 2.35): the same three-way comparison on the FULL 18,432-coordinate support, where plan 2.22 measured cos(u_g,u_M) = 0.588 rather than 0.995. Registered prediction CONFIRMED - the Fisher is no longer near-isotropic there (diag(F) spread 46.002x, cond(diag F + lambda I) = 8.052 vs 1.995 on 32 coordinates; split-half cos(u_F^A,u_F^B) = 0.9910) - but the direction it implies still sits on top of the raw gradient (cos(u_g,u_F) = 0.9872) while the M-optimal direction does not (cos(u_M,u_F) = 0.5878), and no lambda on the tested axis brings them together (cos vs u_M stays 0.578-0.590 at 0.1x/1x/10x median). Realised: u_F loses to u_M in 0/8 contexts, p = 0.0078, median relative -27.5% (median abs -0.2520); u_M beats u_g 8/8, p = 0.0078, +40.7%, which reproduces plan 2.22's 36-43% and is the comparability check that makes the rest quotable; u_F vs u_g is 7/8, p = 0.0703, +1.8% (+0.0122). So 'which G' has one right answer in measurement and it is M: M is not a price tag among others, it IS the budget being scored, and optimising another quadratic form costs about a quarter of the effect - the same structure and magnitude as plan 2.16's u_J (-26.6%). While adding --dense the script was refactored and the refactor first used coordinate indices as neuron ids in What row selection, shifting every 32-coordinate budget by 3-13% (cosines stayed at 0.995, so cosines alone did not catch it); a per-value regression run against the committed plan-2.34 numbers caught it (cosines and sign tests stayed plausible while every budget was 3-13% off), the bug is fixed, and after the fix the u_I arm reproduces the committed result BIT-FOR-BIT (24/24 realised values, max |diff| 0.00e+00), u_M doses agree to 3.1e-07, u_F doses to 3.6e-04 (F itself carries ~1e-3 kernel noise; the sampled continuations are identical across runs), and u_R is not comparable by construction because the refactor changed the order of the random draws. The nulls move their sign counts at that noise level (u_F vs u_M 3/8 <-> 1/8, u_M vs u_I 5/8 <-> 4/8), so plan 2.34's substance rests on the geometry (cos >= 0.988) and on 'not detectable', not on those signs. Positions as the deliverable (plan 2.36): the closed form's own selection rule - greedy forward maximisation of Q(S) = g_S' M_SS^-1 g_S - now chooses the coordinate list; every earlier list ranked by mean |g| or its bootstrap lower bound. The registered prediction was FALSIFIED: I predicted greedy = |g| because M is near-orthonormal (same-layer off-diagonal |M_ij| median 0.0135), yet the list overlap is only Jaccard 0.363 at K=64 (exactly 1.000 at K=1, where it is an identity and doubles as the implementation check) - small off-diagonals do not change the MAGNITUDES but they do change the RANKING, because each greedy step scores a candidate by the value that REMAINS after subtracting the already-selected mass. Measured, at the SAME closed-form dose so only the list differs: greedy beats |g| in 7/8 contexts at K=64 (p=0.0703, median +10.6%, +0.02229) and 8/8 at K=256 (p=0.0078, median +25.8%, +0.06951). The dose rule alone is worth only +2.8% (K=64) / +7.0% (K=256), so the gain is the LIST, not the dose. Two frozen criteria FAILED and stay failed: SECONDARY (pooled list at K=64 retains 51.3% / 80.0% / 100.1% of the full-support effect at b=0.05/0.2/0.5 against a 73% threshold at every budget) and STABILITY (per-context lists have median pairwise Jaccard 0.058 at K=64 against 0.30) - per-context lists work but are not a publishable object, so the deliverable is the POOLED list, whose retention at K=256 is 71.9% / 99.3% / 121.4%: at b >= 0.2 a 256-coordinate closed-form list matches or beats the full 18,432-coordinate u~g arm. Artifacts: results/greedy_positions.{json,md} plus one printable CSV per K (rank, layer, neuron, sign, dose, Q share). Two process notes: the first design compared greedy against the historical |g|+u~g arm, which changes list AND dose at once, so it was discarded before any result was read and the topg_cf control arm (|g| list with the same closed-form dose) was added; and the word-half (G1) stability stage is NOT run, so it may not be quoted. Community recount and position overlap (doc/references/qwen35-community-atlas.md 5.3): re-fetched the neuronary recipe library with the same method - the API now reports Content-Range 0-0/187, so the 141 recorded in 5.1 was a 2026-10-2x snapshot and is stale. Current: 187 recipes (178 on Qwen3.5-4B), 251 neuron-dose rows, 165 unique coordinates, dose median 0.50, 17 rows with |dose|>1, signs +139/-112, per-recipe aggregate budget sqrt(sum dose^2) median 0.85 (p10 0.60, p90 1.24). Raw snapshot stays OUTSIDE git (a local snapshot directory outside this repository), only these aggregates are recorded. The structural fact that bounds everything: our experiment space is layers 13-14, and only 11 of the 251 rows (4.4%) live there, spread over 7 distinct coordinates (L13N3265, L13N4810, L13N6102, L13N7956, L14N4, L14N7049, L14N8949); only 10 of 178 recipes touch those layers at all. Against that ceiling, the closed-form greedy list (pooled K=256, results/positions_K256.csv) independently selects 2 of the 7 - L14N8949 at rank 16 (Q share 0.69%, in 6/8 per-context lists) and L13N3265 at rank 205 (Q share 0.22%, pooled list only) - versus a random baseline of 256*7/18432 = 0.097, i.e. about 21x enrichment; the community's other famous coordinate L13N7956 appears in none of our lists, consistent with the measured cos(u_II-, u_known) = 0.0299. Dose conversion needs no factor because the community uses the same formula (dose * median residual norm / ||W_down[:,n]||), so their coordinate dose EQUALS our u_n: our 256-position list gives L14N8949 u=+0.00457 and L13N3265 u=+0.00196 against their documented 0.7 and 1.0, i.e. 153x and 511x smaller per neuron; while their whole-recipe aggregate budget (median 0.85, p90 1.24) sits at our b~0.85, which is 2-3x above our measured usable ceiling b <= 0.3-0.4 - their recipes work inside the degradation region we mapped, trading capability for persona coverage. Full-model closed form (plan 2.37): computed, not measured. The solver never depended on depth - g for all 32 layers comes from ONE backward per context/family, and because the budget metric is block-diagonal by layer the objective Q(S)=g_S'M_SS^-1g_S is ADDITIVE across layers, so the global greedy is just 32 tiny (P_l,y_l) states; 32 Gram blocks cost 10.9 GB on CPU and the whole run is ~6 min. Result: the position list does NOT concentrate - K=64 already uses 24/32 layers and K>=256 uses all 32, with the LAST layer dominant (L31 holds 20.6% of Q at K=64 and 10.2% at K=1024; rank 1 is L31N1910(+)), which follows from pooled ||g|| being largest at L31 (14.655), then L5 (13.184) and L1 (12.378). My registered prediction (top-64 within <=12 layers) is therefore FALSIFIED. The two-layer list is not overturned but is demoted: 62 of its top-64 survive inside the full-model top-1024 while its ranks 65-256 mostly fall out (62/256 survive), and L14N6837 goes from rank 1 to rank 12 - i.e. restricting the search to layers 13-14 inflates the value of mid-list positions. Community comparison (public API snapshot 2026-10-29, 165 coordinates, outside git): 0 hits at K=64, 1 at K=256, 4 at K=1024 against a random null of 0.04/0.14/0.57, so ~7x enrichment at K=1024 and weaker than the 21x of the two-layer list; the famous L14N8949 drops from rank 16 (within 18,432 coordinates) to rank 290 (within 294,912), and one shared coordinate, L20N1950, has the OPPOSITE sign (ours -0.00156, theirs +0.40). The structural reason is that the two systems prefer different layers: community L7/L22/L21/L6/L23/L17, ours L31/L5/L25/L26/L28/L30. No readout was measured in this run, so nothing here is an effect claim; Q is a surrogate whose calibration is known to be optimistic, and the dose ratios (0.002-0.008) are budget-spread ratios, not effect ratios. Per-context list stability stays low (Jaccard 0.143 at K=64, 0.098 at K=256), so only the pooled list is publishable. Determinism: the rerun after a rendering fix produced a byte-identical JSON. Community measurement protocol, pinned (doc/references/qwen35-community-atlas.md 5.4): read the neuron-explorer repo at commit 6a6f0e95bf49ad6626940021148f775d7d55d1e4 (2026-10-28) - research_log.md, explorer/agent_methodology.md, sweep/steer_sweep.py, sweep/rank.py. Their method: steer ONE neuron at the mlp_out hook by adding d*abs_max[n] * W_down[:,n] (they assert the model has no FFN layer_scale), default greedy decoding, 4 fixed chat prompts, one unsteered baseline per prompt, 10 rollouts per neuron under three designs, the relevant one being main_rel where the dose is a FRACTION OF THE MEDIAN RESIDUAL NORM (their code: resid_p50 = stats['resid_norm_q10_50_90'][1]) - the same statistic as our p50_l, so their rel dose 0.5/0.8 equals our b=0.5/0.8 for a single coordinate and no conversion factor is needed. Controls are 256 random directions of the same norm and dose scale; they report null_pct (share of controls with a lower word-cluster mass) rather than a hypothesis test. The readout is the GENERATED TEXT: distinctive words per unit vs the average unit with a Poisson z, mass = sum of the top-5 positive z as the sort key, plus a per-dose coherence curve (distinct-4-gram ratio, share of length-capped rollouts, mean chosen logprob). Scale: 9,216 neurons x 10 rollouts ~ 95k generations ~ 1.4 h on a 4090. Two things line up with us: the dose ruler is identical (median residual norm), and their 'coherent window ~0.5-0.8x, gibberish past 1.5x' is the same order as our readout cost starting at b~0.3-0.4 and behavioural damage at b=0.5; and their finding that 'did the rollout change' is useless (83-98% change even at +-1) matches our lesson that cosines and sign tests are blind to numerical mis-scaling. Two things do NOT line up, and they explain why the coordinate sets overlap thinly: they rank by behavioural word clusters with a random-direction null, we rank by a teacher-forced logit-margin surrogate with paired sign tests over contexts; and they inject one neuron at a fixed dose while we spread a budget over K positions. Not adjudicated - that needs the same-budget J2 comparison, which we have not run. What the community has PUBLISHED at coordinate level, and what it has not (doc/references/qwen35-community-atlas.md 5.5, same pinned commit 6a6f0e9): usable today are the neuronary recipe library (187 recipes / 165 coordinates / 251 rows with dose - already used in 5.3 and 2.37) and the explorer caches that ship with the repo (lens_index.npz = per-neuron logit-lens up/down top-k token ids, steer_index.npz = per-neuron steer lenses, connections.npz = 24 strongest direct links per neuron, layout_writes.npz = the 96x96 sheet positions, act_scale.npz = an activation-scale cache whose generator is not in the repo). NOT published, because .gitignore excludes runs/ stats/ captures/ logs/: runs/<run>.units.jsonl - the behavioural ranking itself (distinctive words, mass, null_pct) - and stats/steer_lens_L*.npz, their closest thing to a per-coordinate direct effect (+-0.8 push Delta-logprob and KL), plus runs/depth_report.json. So a direct Spearman between our Q order and their mass order is impossible without either asking them to publish units.jsonl or re-running their sweep ourselves (vLLM steering fork + pod; 9,216 neurons x 10 rollouts ~ 95k generations ~ 1.4 h per layer on a 4090), and note their whole-layer sweeps are per-layer (L22 full, then L8, plus random subsets across 4/9/13/17), so even that would not equal a full-model ranking. Three comparisons remain possible without their re-run: the coordinate x dose one already done, a SEMANTIC check of our top-K positions against their logit-lens/steer-lens words, and a STRUCTURAL check against their 24-link graph. Fetching caveat stated plainly: git clone and raw.githubusercontent are both unreachable from this machine and the remote box (only github.com pages are readable), so the shipped caches are NOT yet on disk here. Three direct comparisons against the community's PUBLISHED per-coordinate data (doc/references/qwen35-community-atlas.md 5.6), after fetching the repo (217 MB, commit 6a6f0e9; git clone needed the proxy: raw works, api.github.com is 403). CORRECTION to 5.5: their per-coordinate next-token KL IS published - it ships in explorer/cache/steer_index.npz (up_kl/down_kl at dose 0.8, all 294,912 coordinates, no zeros) and uses the SAME dose formula as ours (+-dose x median residual norm / ||W_down[:,n]||) on 4 short contexts, last position, whole vocabulary; only the generator script's stats/ output is gitignored. Results. (A) Their measured KL and our closed form barely rank-agree: Spearman(|g|, KL_up) = +0.023 over all 294,912 coordinates, KL_down +0.019, per-layer median +0.021 (range -0.037..+0.068), and inside our top-1024 Spearman(Q share, KL) = +0.129. But our SETS are enriched under a same-layer null: our 32-layer list has median KL percentile 0.710 (random 0.523) with 18.8% in its own layer's top-10% (random 9.2%); the 13/14 list gives 0.661 and 0.626 (weaker). The reason the rankings disagree is now measurable: their KL is dominated by network ends - per-layer median KL L0 = 9.95 and L31 = 6.37 vs 0.45-1.21 in the middle - and their KL top-1024 is L0 745, L31 149, L30 59, L1 46, while our Q top-1024 is L31 77, L5 47, L25 40, L26 40, L30 37; the two top-64 sets share exactly one coordinate. (B) Their 165 recipe coordinates sit at chance under BOTH instruments: layer-matched KL percentile median 0.659 with 11.5% in their own layer's top-10% (random 10%), and only 4 land in our top-1024 (L14N8949 rank 290, L16N8927 231, L17N6015 412, L20N1950 384) - so the community's published choices are explained by neither per-coordinate instrument. (C) Semantics: our 32-layer top-64's own logit-lens words hit our lexicon (Jibes adj/verb + BFFM) at 5.21% vs 0.26% for random coordinates (~20x), the 13/14 list only 0.52% - independent external support for the Q surrogate. (D) Structure: our top-64 and top-256 are 0.094 and 0.119 internally dense in their 24-link graph versus 0.001 for random 256, i.e. a connected cluster at ~100x chance. Boundaries: same dose unit but different readouts (their KL at +-0.8 on 4 short contexts, last position, whole vocab; ours a lexicon margin over 8 heldout contexts at b=0.05-0.5), so only the ordering of positions is comparable, never the magnitudes. Analysis script committed as tools/compare_community.py; the repo and caches stay outside git (a local snapshot directory outside this repository). Selection rules extended to K=2048 over all 32 layers and judged ONLY by external instruments (plan 2.38; no readout of ours). Four rules on the same g and M: absg (top |g|), greedy (incumbent), umag (rank by |u| of the UNBUDGETED closed-form solution u = M^+ g, computed via the plan-2.21 identity so no 9,216^3 solve), waterfill (per-layer marginal curves, isotonic smoothing, equal-slope allocation), random. Surrogate Q at K=64/256/1024/2048: greedy 53.11/113.00/227.62/328.96, absg 9.43/33.73/121.86/258.81, umag 8.40/19.67/69.83/195.48, random 1.36/3.97/14.69/29.12; waterfill is IDENTICAL to greedy at every K to 1e-15, so the isotonic smoothing changed nothing - greedy's tail tie-breaking and its L0/L31 concentration are not a flat-marginal artefact. External verdicts: greedy is best on semantics (their logit-lens words hit our lexicon at 5.21%/5.40% at K=64/256 vs 0.13% for absg, 0.26% for umag, 0.26% random) and on structure (internal edges in their 24-link graph, same-layer-mix null: 102.9x at K=64, 71.1x at 256, 13.0x at 1024, 5.7x at 2048, against 2.6x/13.2x for absg, 13x/6x for umag, ~1x random), but umag wins their KL at small K (layer-matched percentile 0.731 and 35.9% inside its own layer's top-10% at K=64, versus greedy 0.710/18.8% and absg 0.464/15.6%). So the three external instruments disagree systematically: KL rewards single-point loudness, semantics and structure reward group coherence, and no rule wins all three. A FALSIFIED prediction of mine: I had registered 'greedy ~ umag > absg > random on external metrics'; semantics and structure put greedy far ahead while KL puts umag ahead, so the ordering is not single. K is the larger variable: by K=2048 Q converges to 195-329 and KL to 0.58-0.62 while the semantic and structural advantages decay monotonically (lens 5.21% -> 1.95%, enrichment 103x -> 5.7x) - i.e. the rule matters only for the short list, which is the only printable form. Two bugs were found and fixed en route and are worth remembering: the K-truncation must be applied to ONE GLOBAL interleaved order (per-layer lists truncated separately silently selects layer 0 only), and the connection-graph density needs BOTH endpoints of each edge (storing only the target end made the count 0). Compute note: moving the 10.9 GB Gram blocks onto the GPU and running the rules in one pass cut the run from >10 min (single-threaded CPU) to ~2 min. Position-change pairs, and the scale test that kills the one-point coincidence (plan 2.38.5/2.38.6): the deliverable is now PAIRS, not a list - results/positions_all32_K1024_pairs.csv carries rank/layer/neuron/sign/u_dose/alpha_raw/|dresid|/p50/||W_col||/q_share for all 1,024 positions, and results/query_pairs.csv carries three readings for a NAMED coordinate (solo, the K=64 optimum forced to contain it, and its marginal). For L14N8949: inside our K=1024 list at b=0.05 the pair is u=+0.001585 / alpha=+0.0352; forced into the best K=64 list the pair is u=+0.021235 / alpha=+0.4720 and that list loses only 0.221 of Q (-0.42%); as a solo single point our closed form gives u=+0.6672 - which is within 4.7% of the community's documented 0.7. That coincidence does NOT generalise: over the community's 165 recipe coordinates (results/community_pairs.csv) the sign agrees on only 84/165 = 50.9% and Spearman(|their dose|, |g_i|) = 0.098, median ratio 0.007 - and among the four that also enter our top-1024, L20N1950 has the OPPOSITE sign. So the two methods share the UNIT, not the value: their dose is behaviourally tuned (which direction makes the persona appear), ours is a hostility-vs-praise lexicon margin, and the signs are objective-dependent by construction. L13N7956 behaves as 'cheap and useless' in our metric (marginal dQ +0.0165 against a K=64 base of Q=53.11; forcing it in costs -0.487, -0.92%), while our own rank-1 L31N1910 takes u=+1.8323 solo (b=1.83, far outside the usable band) yet only u=+0.0189 inside the K=64 list - i.e. the dose is role-dependent, not a property of the neuron. Same-objective test scaled to all 178 Qwen3.5-4B recipes, four settings in one GPU run of 5 minutes (objective_match_batch.py; results/objective_match_batch.json). Result: DIRECTION is reproducible, POSITIONS only neighbourhood. With the plain positive objective (J = mean teacher-forced log p of their published output), the sign of our gradient agrees with the sign of their behaviourally-tuned dose in 93.4% of cases - against 50.9% for the cross-objective comparison of plan 2.38.6, which settles that the sign is objective-determined, not arbitrary. But exact set overlap at K = their neuron count is only 1.1% of recipes (raw |g|) and 3.9% with per-layer RMS normalisation, while 50.0%/53.4% of recipes have at least one of their coordinates inside our top-64 at a median rank of 13/64 - against a random baseline of 64*1.4/294912 ~ 0.03%, i.e. a thousand-fold neighbourhood hit rate but not the same short list. Layer equalisation measurably reduces the plumbing bias (max single-layer share of our top-64: 21.9% -> 15.6%) without closing the gap, so the remaining difference is the SAMPLING SURFACE of the objective (they searched 25 prompts, we used the one published example), not the solver. A second, unexpected result: the CONTRASTIVE objective excludes their coordinates entirely - 0.00 of their neurons in our top-64 across all 166 recipes and sign agreement falling to 52.0% - so contrast, which is what our own J2 readout is, is the wrong objective for reproducing behaviour; the plain positive objective plus layer normalisation is the right match. Practical reading: if a position list is meant to agree with a behavioural search, use a positive objective on their prompts and equalise layers; our contrastive J2 list is optimised for our readout and should not be sold as behaviour-recovering. Sampling-surface test (plan 2.39.2, 60 recipes x 4 prompt surfaces built only from published text; results/objective_surface.json). My registered prediction - exact overlap rises with the number of surfaces - is FALSIFIED: exact overlap at K = their neuron count is 1.67% for 1, 2, 3 and 4 surfaces alike (one recipe out of 60 at every depth). What does move is the neighbourhood: recipes with at least one of their coordinates inside our top-64 go 51.7% -> 56.7% -> 60.0% -> 60.0%, at a median rank of 11-12, and the sign agreement is 98.5-98.9% on this subset (93.4% on the full 178). So the three questions separate cleanly and should never be conflated again: the DIRECTION is reproducible (93-99%), the NEIGHBOURHOOD covers half to 60% of recipes, and the exact 1-4 coordinate LIST is not recoverable (1.67%, invariant to surfaces). The residual gap is methodological, not a sampling artefact: their coordinates are found by reading which neuron WRITES the behaviour's words in the steer lens and then validating behaviourally, whereas a first-order gradient ranks coordinates by marginal log-probability gain - and under joint/threshold effects (their own record: a 6+1 sum measured at 16, 2.3x superadditive) those are different orderings. Deliverable boundary to state publicly: we can claim positions supported by the algebra AND the three external instruments, and signs that agree with behavioural doses to 93-99%, but we must NOT claim to recover a behaviourally-searched coordinate set. Second-order probe (plan 2.39.3, 12 recipes, exact per-token Fisher in residual space where the budget metric is the identity; results/objective_second_order.json). Hypothesis tested: the residual gap after 2.39.2 is the objective's CURVATURE (joint/threshold effects their recipes show, 2.3x superadditivity). Result: not confirmed - second order makes the neighbourhood WORSE, not better. Recipes with at least one of their coordinates inside our top-64: plain |g| (first order, coordinate metric) 58.3%, umag (first order, budget metric) 16.7%, second order 25.0%/16.7%/16.7% at mu=0.3/1/3. What curvature does buy is the RANK when a coordinate does appear (median 7-14 against 28 for |g|), so it narrows without widening. Exact overlap at K = their count is 0/12 here (1/60 in the 60-recipe run, so that column is noise at this sample size). Sign agreement stays 95.1% (12 recipes), consistent with 93.4% over 178 and 98.5% over 60 - direction is stable across every variant, and positions are what changes. Interpretation, now with three independent probes behind it: the community's pipeline finds SEMANTICALLY NAMEABLE coordinates (steer-lens: which neuron writes the word) and validates behaviourally, while our closed form finds the coordinates with the most effect per unit residual budget; the budget-metric correction is exactly what moves us off their picks, and that same correction is what wins +25.8% on OUR readout (2.36 K=256). Hence reproducing their coordinates is the wrong objective for our method - it would trade away our own advantage. |
| 路线 | `R0` done · `R2a` done · `R2b` done, minimal · `R2` paused · `R2c` partly superseded: the gate that failed · `R6` done, NEGATIVE · `R1` todo · `R3` largely answered by 2.20/2.21: the dense direction beats every sparse subset at fixed budget · `R4` todo · `R5` todo · `R7` done, POSITIVE, with a measured envelope |
| 判据 | `G1` · `G2` · `G3` · `G4` · `G5`（定义见 `metadata.json`，说明见 `doc/plan-gradient-discovery.md` §1） |
| 结论 | J2 判据下该坐标把敌意端相对赞美端推高（8/8 top 坐标，随剂量单调，h−n +0.018 / c−n −0.017）；J1 五特质矩阵是「评判性 vs 中性」仪器，用它判特异性属用错仪器（§8 结论二/三撤回） |
| 实测 | G1 r=0.693/J64=0.524（19 条 prompt） · G2 r=0.668–0.781 / J64=0.471–0.662（9v10 / 10v9） · verify 8/8 p=0.0078（6 reused probes -> replication, not confirmation） |
<!-- auto:workstream-status:end -->

## 工程约束（一条命令：`make check`）

本目录的**每一条对外声明都必须能被重新推导**，而不是靠记忆。数字与状态只有一个来源，其余都是渲染出来的。

| 规则 | 机制 |
|---|---|
| 版本 / 状态 / 判据 / 路线 / 实测数字 | **唯一来源 `doc/metadata.json`**；`README.md` 里的状态块是渲染结果（`make meta` 重渲染，`make check` 校验） |
| 包清单（md5 / sha256 / 文本文件数） | 由 `packages/*.zip` 现算后渲染进受管段；`make check` 校验区块与磁盘一致 |
| zip 与解包树 | 逐成员比对（`ziptrees`）；不等即失败 |
| 包内自校验 | 每个 `vendor/*/<root>/CHECKSUMS.sha256` 必须验过 |
| 代码 | 全部 `.py` 可编译（`compile`） |
| 测试 | 免 torch 的单测必须通过；其余标为 skipped 并提示到 GPU 机跑 |
| 去重 | 版本族内允许重复，**跨族重复即失败**（`dedup`，只读不删）。例外只能写在 `tools/dedup_allow.json`：按 **sha256** 逐条登记、写明理由与日期、并列出该组**全部**文件；条目与磁盘不符（内容改了、副本增删）即再次失败 |
| Git 边界 | 任何 `*.zip / *.tar.gz / *.pyc / *.pt` 被跟踪即失败（`git`） |
| 外部证据 | `evidence/ledger.json` 必须通过校验，且 `11_` 里的矩阵必须**等于**渲染结果；**每条 S/W/X 判定必须含数字**（`ledger`） |

- 入口：`make check`（stdlib-only，无网络、无 GPU、无 torch）；分项可用 `make blocks|ziptrees|checksums|compile|tests|dedup|git`。
- 钩子（opt-in）：`make hooks` → `core.hooksPath=.githooks`，`pre-commit` 只在 `*.md` / `results/` 被暂存时跑 `make check`（约 2 s），非 qwentest 提交不受影响。
- 工具：`tools/check.py`（本文件的校验器）、`tools/migrate_repo_packages.py`（解包 + 受管段，仓库级**规范副本**）、`tools/dedup_check.py`（只报告，不删）、`tools/dedup_allow.json`（跨族重复的**已审例外表**，现为 3 组 = SNAM 从 v6 族继承的 `engine.py` / `audit.py` / `test_engine.py`）。包内同名工具是随包冻结的副本，改动以 `tools/` 为准。
- 输入放在**冻结包之外**：新的探针/词表写到 `probes/`、`lexicons/`（本次新增 `probes/contrasts_v3.jsonl`），不要往 `vendor/` 里加文件——那里由 zip 决定，加了就会让 `ziptrees` 与 `check-remote` 的树哈希失配（本日实测：误放一份探针到远端包目录，哈希从 `b23ee718…` 漂到 `e00c4de1…`）。
- 改数字的顺序：**先改 `doc/metadata.json`（或重跑实验）→ `make meta` → `make check` → 提交**。手改状态块会被 `check` 抓住（已实测：改一个数字即 FAIL）。

## 目录结构

```
qwentest/
├── README.md            ← 本文件
├── packages/            实验包（zip，可上传运行）
├── 02_archives/            结果与重建归档（tar.gz）
├── vendor/            上述包/归档的解包内容（便于直接查看）
├── tools/               与远端交互的辅助脚本
├── tools/                  本目录的工程门禁：check.py、migrate_repo_packages.py、dedup_check.py、dedup_allow.json
├── doc/plan-gradient-discovery.md  梯度发现的读数路线（R2 暂停、R6 现役，其余 TODO）
├── doc/plan-specificity-dose.md    计划 06：特异性与剂量（已执行，含 §9 结论修正）
├── doc/references/qwen3-8b-neuron-atlas.md  外部参考数据集的统计记录（逐条标 S/W/X/M/N 对我方主张的影响）
├── doc/references/evil-qwen-mlx-replication.md  第二份外部参考统计记录（同一模型的 MLX 8-bit 独立复现）
├── doc/references/qwen35-community-atlas.md  第三份：社区多源横向重算（4-bit/8-bit 交叉、稠密 vs 稀疏、超加性、自撤）
├── doc/references/synthesis-external-evidence.md  证据总账：15 条主张 × 12 个来源的机械校验矩阵 + 7 条前提轴
├── doc/references/preprints.md  四篇新预印本的钉版记录与它们改变了什么（CANS 的 F、RACE 的下界排序、transcoders 的基、Steering Geometry 的设计）
├── doc/references/CANS-2610.35530-v1.md  CANS 正文精读（TeX 源）：公式对照、超参数、图内标定的更正、可引用的 Limitations
├── evidence/ledger.json   证据总账的数据（唯一来源；`tools/evidence_ledger.py` 渲染到 11_）
├── results/             GPU 运行的原始产物（JSON/MD，文本入仓）
├── lexicons/               可复现词表（jibes_adj_verb_pools.json、bffm100_factors.json）
├── probes/                 探针输入（contrasts_v3.jsonl）——冻结包之外
├── doc/metadata.json           数字与状态的唯一来源
└── scripts/                本地统计脚本 + GPU 实验脚本（均带 --dry-run 与 [i/N]…eta）
```

后续路线与预注册判据（G1–G5）见 **`doc/plan-gradient-discovery.md`**：R2（token 上的正规语言精确边缘化）为现役工作，其余登记为 TODO；该文件先冻结判据、再跑实验。

---

## 01_packages — 实验包

Package checksums are maintained in the [Unpacked package provenance](#unpacked-package-provenance-automatically-maintained) table below.

### persona-matrix v2 相对 v1 的关键改动（都已落在这份包里）

- `--dose` 默认 **1.0**（与 v3 记录同一剂量标尺；`package_version: v2-p2`）
- `refusal` 强度按原作者注记改为 **+1.5**（原先 1.0，跨 node 剂量不可比）
- `zero_hook` 覆盖本次运行涉及的全部 (层, 神经元)，不再写死 L13/L14
- 新增 **norm-matched 随机方向对照** `randdir:L<层>:s<种子>`（残差空间随机单位向量，范数 = dose×p50）
- `analyze.py` 导出**真盲评表** `blind_*.jsonl` + 密钥 `blindkey_*.jsonl`（全 run 洗牌、去掉条件字段）
- `geometry()` 标注 `kind: identity` —— 同一神经元两端的 `-1` 是恒等式，不是发现
- 新增 `gate_check.py`（Gate 0/2/2b）与 `desire_interactive.py`（流式、Ctrl-C 打断）

### snam-semantic-inverse-v1 — 另一个包族（语义编译 × 解析逆映射）

**出处**：外部会话交付，非本目录自建。会话记录已入仓：
`related_files/conversation-evil-qwen-recipe-to-snam-2026-10-26-27/`（share `6ae14833…`，
2026-10-26 02:48 → 10-27 09:51 UTC，1003 节点）。末条助手消息 09:51 交付该 zip，仓内文件 mtime
09:52 UTC（+1 分钟），md5 `9c3e109b…`、24 个文本文件——与会话自述「53.9 KB · 24 个文件」相符。
**包内没有 LICENSE 文件**，也未声明授权；在拿到授权前只作归档与审阅，不并入我们的代码线。

**它做什么**：把自然语言风格定义编译成词表上的两个分类分布 `q±`，读数是最后一个位置的
`J = E_{q+}[logit] − E_{q−}[logit]`；两个 solver（`W` 只用 `W_down[:,n]` 的权重代数、`G` 用 VJP）各自排序，
3 编译器（lexical / distribution / counterfactual）× 2 solver × 3 特质（hostility / enthusiasm / pessimism），
预注册 H0–H10 + 冻结 + 封存。`probes/contrasts.jsonl` = 54 行 = **18 个 prompt × 3 特质**（各 6 discovery + 12 heldout，中英各 10 条）。

**本机/远端已复核（2026-10-28，第一手）**：

| 项 | 结果 |
|---|---|
| `sha256sum -c CHECKSUMS.sha256` | 23/23 OK |
| `python -m unittest discover -s tests`（远端 CPU） | 24 tests OK |
| `freeze.py` 真跑 | 写入模型指纹 + 源码指纹 + 输入哈希 + 计数；`require_plan` 四项复验 |
| `discover.py --traits hostility --discovery-limit 6 --top-k 8` | **25 s** 跑完（含 8.8 GB 哈希、加载、18 次 VJP），结果封存 |
| 与 v6 族逐文件比对 | 24 项中 **2 项逐字节相同**（`engine.py`、`audit.py`）、7 项同名不同、**15 项本族独有** |

**审阅发现（我方判断，数字来自上述那一次 run 的 `out/snam_v1/discovery`）**：

1. **六格之间几乎没有共识**：6 个格子 × top-64 = 384 个名额，只覆盖 **351 个不同坐标**；出现在 ≥2 格 31 个、
   **≥3 格 2 个、≥4 格 0 个**；六格给出**六个互不相同的 top-1**（L14N5492 / L14N213 / L14N2442 / L13N5526 / L14N3759 / L13N7104）。
   预注册里没有「方法×方法一致率」这一条，建议补为 primary outcome，否则 18 个格子会各自产出一个互不复现的「发现」。
2. **`W` 与 `G` 不是同一个量的两种估计**：在 2 层 × 9216 坐标上 Pearson 只有 **+0.102 / +0.054 / +0.064**
   （lexical / distribution / counterfactual），top-64 重叠 **3 / 0 / 2**。所以「G 优于 W」（H2）比较的是两组几乎不相交的候选，不是精度。
3. **编译器之间也基本不重合**（同 solver 内 top-64）：weight 下 0–2/64；jacobian 下 lexical–distribution 19/64（r +0.59），counterfactual 与两者 0–6/64。
4. **一处自我更正**：我起初怀疑 `q±` 被标点主导，量化后**撤回**——`|q+−q−|` 的标点份额只有 **2.2% / 8.8% / 9.6%**（内容 82–98%）。
   但顶层 token 里确实出现 `，` `、` `！`（权重 ≤0.083），所以标点不是主线，只是可见噪声；仍建议在编译 `q±` 前把标点/空白清零并记录其质量份额。
5. **`norm_matched_own_sign` 对照臂在该剂量约定下近似空转**：`alpha = u/‖W[:,n]‖` 使每个单神经元的残差范数恒为 `|u|`，
   于是 N 臂 ≈ 另一个 R 臂。要匹配范数得匹配**下游残差方向**，或回到 p50 标尺。
6. **剂量口径可直接对照我们的旧结果**：包内 `dose=0.5` 是残差范数 0.5，与我们的 `dose 0.05 × p50≈9.7 ≈ 0.49` 几乎相同；
   README 只写「不是 v4 的 0.05」，建议补一句「≈ v4 的 0.05」。
7. **产物默认写在包目录内**（`--out out/snam_v1`），与我们「冻结包目录不得写入」的规则冲突：一旦在解包树里跑，
   `ziptrees` 与 `check-remote` 的树哈希立刻失配（本日已有先例：误放一份探针到远端包目录，哈希从 `b23ee718…` 漂到 `e00c4de1…`）。建议默认指向包外。

**边界（不要混用）**：它的读数是 **logit 期望**（线性于 logits），不是我们 R2 现役的 **J2 log-mass**；
它的探针是**成对手写续写**（J1 风格），不是 J2 的槽位池。所以它能回答「同一批共享上下文上的跨特质」，
**不能**直接回答我们卡住的「同语域槽位级特异性」。它上面那些数字是**一台 4090 上的一次 run**，不是跨机结论；
我们的 J1/J2 结果与它互不引用。

---

## 02_archives — 结果与重建归档

| 文件 | md5 | 内容 |
|---|---|---|
| `qwen-results.tar.gz` | `2a96a1077dc691b7b4f77eb278d80fa3` | 早期两批 dose 1.0 结果（6 + 80 条） |
| `persona-single-d0.5.tar.gz` | `74175ef744a0c000d1f13b37be0c9f51` | persona-matrix dose 0.5 / phase single 全量：1080 条 |
| `rebuild-kit.tar.gz` | `8c52887f3e38b240a8e75c1e52a6bc32` | 重建包：`REBUILD.md` + `env.txt` + 代码 + 两个 run 的 out/ + gate 输出 + 各 zip |

---

## 03_unpacked — 解包内容

| 目录 | 说明 |
|---|---|
| `qwen-results/` | `out/results_3dca6cebd49e.jsonl`(6) + `out/results_5ddba4527220.jsonl`(80) + **`NOTE_random_pair_03.md`** |
| `persona-single-d0.5/out/` | `results_626721938e67f4.jsonl`(1080) + manifest + geometry |
| `qwen-evil-repro-v2/` `qwen-evil-repro-v3/` | 两个版本的包目录（仓库式结构可直接跑） |
| `qwen-persona-matrix-v2/` | v2-p2 包目录 |
| `snam-semantic-inverse-v1/` | **另一个包族**（语义编译 × 解析逆映射，24 个文本文件）；与 v6 族逐字节相同的只有 `engine.py`、`audit.py` 两个模块，其余 7 个同名文件内容不同、15 个是本族独有。见下节 |
| `rebuild-kit/` | 重建包解包（**已去重**，见下；只剩 `REBUILD.md`、`env.txt` 等 5 个别处没有的文件） |

`rebuild-kit/` 解包目录做过一次去重：只留下别处没有的 5 个文件——`REBUILD.md`、`env.txt`、
`qwen-persona-matrix/README.md`（这一版 README 与 v1、v2 都不同）、
`qwen-persona-matrix/gate_out/gate_72a7e1cb.json` 和 `gate_72a7e1cb.md`。删掉的 20 个文件
（17 个文本 + `qwen-evil-repro-v2.zip`、`qwen-persona-matrix-v2.zip`、`qwen-results.tar.gz`）
都与下列位置的文件逐字节相同，动删之前对过 md5；看内容请直接去这些位置：
`qwen-evil-repro-v2/`、`qwen-persona-matrix-v2/`、`qwen-results/out/`、`persona-single-d0.5/out/`、
`packages/`、`02_archives/`。`rebuild-kit.tar.gz` 没有改动，仍是完整的一套；要看整包结构就
`tar xzf 02_archives/rebuild-kit.tar.gz -C /tmp/kit`，不要解回本目录（否则副本又回来了）。
`scripts/dedup_check.py` 可以复核这次删除没有丢内容。

`NOTE_random_pair_03.md` 是本次唯一未结线索的登记：`random_pair_03`（L14 N2990 / L13 N4787）
在 dose 1.0 的 8 个 prompt 中有 4 次落入安全/越界语域（p04 作弊、p06 卷入犯罪、p07 人际遮掩、
p08 椅子那一侧），且跨任务复用同一短语 "face the problem/challenge"。若成立，它作为"随机对照"
会低估 target_pair 的特异性——需要一批正式随机对照才能定论。

---

## scripts — 本地统计（纯本地，不需要 GPU）

| 文件 | 说明 |
|---|---|
| `stats_single.py` | 单个 run 的描述统计：变化率、与 baseline 的相似度、退化率、按类型/上下文长度的计数 |
| `stats_timing.py` | 同一 run 的耗时分解：prefill vs decode 最小二乘、按上下文长度与条件分组 |
| `dedup_check.py` | 复核 `vendor/rebuild-kit/` 里没有与别处逐字节相同的文件（默认只看不改，`--apply` 才删） |

两个 stats 脚本的默认输入与输出都按脚本自身位置解析（输入 `../vendor/persona-single-d0.5/out/results_626721938e67f4.jsonl`，输出同目录 `out/`），在任意工作目录下都能直接跑；`out/` 是派生结果，不入 git。

### scripts — 需要 GPU 的实验脚本（用户手动在远端跑）

全部支持 `--dry-run`（只打印计划与预算）并逐项打印 `[i/N] … elapsed, eta`。

| 文件 | 回答的问题 | 依赖 |
|---|---|---|
| `dose_specificity.py` | 计划 06：剂量单调、镜像臂、同层随机对照、安慰剂 | 包 v6 + `probes/` |
| `crosstrait_k.py` | “敌意从不第一”是 top-K 的性质还是单次抽样 | 包 v6 + 扫描产物 |
| `j2_two_readouts.py` | 同一 J2 家族内两个读数（敌意 vs 赞美 / 赞美 vs 中性） | 包 v6 + 词表 |
| `slot_census_bffm.py` | R2c 前置门槛：BFFM-100 有多少词过得了单 token 槽位过滤（**已跑：56/100，门槛未过**，见 §2.19） | 仅 tokenizer（CPU） |
| `g_feasibility.py` | 非对角 `G = JᵀJ` 存不存在、是否上下文稳定（成本建在单位置） | 包 engine + tokenizer |
| `g_j2_space.py` | 把 G 重建在 **J2 自己的输出空间**后，控制律变不变 | 包 engine+semantic + 词表 |
| `g_nonlinear_verify.py` | **真实非线性模型上**，同一物理预算下各 G 暗示的方向谁拿到的实测 ΔJ2 更大 | 包 engine+semantic + 词表 + 上两者的 Ĝ/g |
| `j2_batch_sensitivity.py` | J2 读数对 **batch 形状**有多敏感（= 我们所有 ΔJ2 的数值误差条） | 包 engine+semantic + 词表 |
| `j2_precision_check.py` | 同一生产形状下 **bf16 vs float32**，直接量 ΔJ2 的舍入误差 | 包 engine+semantic + 词表 + Ĝ/g |
| `support_sweep.py` | 同一物理预算下，实测 ΔJ2 随**支撑集大小 K**（1 → 18432，嵌套 top-K）怎么变 | 包 engine+semantic + 词表 |
| `dense_optimal.py` | **R7**：稠密区里预算最优方向 `u ∝ M⁺g` 是否赢过 `u ∝ g`（含残差方向 / 最小范数代表 / 等价类） | 包 engine+semantic + 词表 |
| `dense_optimal.py --transfer` | R7 的**迁移检验**：把方向建在一半上下文上、到另一半去测（in/out-of-sample + G2 门） | 同上 |
| `dense_optimal.py --g1` | R7 的 **G1 检验**：把整套在两个不相交词半（`pos_A`/`pos_B`）上重建重测 | 同上 |
| `lcb_rank.py` | 检验 RACE 的 CAM 思想：按上下文自助下界排序是否比按 \|mean g\| 排序更省坐标 | 包 engine+semantic + 词表 |
| `compiler_pareto.py` | **编译器 vs 独立尺子**：3 编译器 + 阳性对照 × 同一求解器 × 两个借来的毒性分类器（**已跑，§2.28**） | 包 engine+semantic + 词表 + 两个分类器 |
| `fetch_models.py` | 给**连不上 huggingface.co** 的机器用的分块下载器（range + 重试 + sha256 校验 + manifest） | 仅 curl |
| `behavior_check.py` | **行为端**（无判官）：R7 各臂的生成退化指标 + 中性参考文本 logP 的能力代理 | 包 engine+semantic + 词表 |

---

## 07_results — GPU 运行的原始产物

文本入仓（JSON 含完整矩阵与逐条统计，MD 是渲染表）；远端原件在 `/root/autodl-tmp/runs/` 下，租机释放即失，所以以仓内为准。

| 文件 | 内容 |
|---|---|
| `g_feasibility_k32_m192_c3.{json,md}` | 单位置成本下的 Ĝ（3 上下文 × 32×32）、逐坐标表、折半信度、verdict A/B |
| `g_j2_space_k32_m192_c3.{json,md}` | J2 匹配输出空间（916,480 维）下的 Ĝ、真实 g 向量、两种选对规则的联合 Q |
| `g_j2_space_k32_m192_c8.{json,md}` | 同上，扩到 8 条上下文（非线性验证的输入） |
| `g_nonlinear_c8.{json,md}` | 真实模型上的判决：8 上下文 × 7 臂 × 3 预算的实测与预测 ΔJ2、配对符号检验、标定比 |
| `j2_batch_sensitivity_c8.{json,md}` | 5 种 batch 形状 × 2 干预状态下的 J2 与 ΔJ2；三个精确的零与一条真实误差条 |
| `j2_precision_c8.{json,md}` | bf16 与 fp32 在同一生产形状下的 ΔJ2 对照（误差标尺的**直接**测量） |
| `slot_census_bffm100.{json,md}` | R2c 门槛普查：100 个 Goldberg 形容词的逐词 piece 分解与 10 池存活数 |
| `support_sweep_c8.{json,md}` | 支撑集扫描：8 上下文 × 15 个 K × 3 预算的实测/预测 ΔJ2、实际预算、非零坐标数 |
| `support_sweep_L14_c8.{json,md}` | **单层对照**（只动 L14，K=1→9216）：排除“动两层优于动一层”这个混淆 |
| `support_sweep_topids_c8.json` | top-64 的**坐标身份**（层/神经元/符号/g/累计 ‖g‖² 份额）与两个知名坐标的排名 |
| `dense_optimal_c8.{json,md}` | **R7 正面结果**：三臂 × 三预算的实测/预测 ΔJ2、每层残差范数、实际预算、各臂 top-64 坐标（层/神经元/符号/强度） |
| `dense_transfer_c8.{json,md}` | **R7 迁移检验**：方向建在一半上下文、测在另一半；G2 门、跨半 cos、in/out-of-sample 全部 144 个测量 |
| `dense_g1_c8.{json,md}` | **R7 的 G1 检验**：两个不相交词半 × 逐上下文/池化两种建向 × 3 预算 = 192 个测量，含两套一致性统计 |
| `lcb_rank_c8.{json,md}` | **下界排序检验**（RACE 的 CAM 移植）：两个排序的完整捕获曲线；登记预测被否 |
| `compiler_pareto_c8.{json,md}` | **编译器 × 独立尺子**：4 臂 × 8 上下文的 E_tox/ΔlogP/逐条生成文本；三个登记预测的判定 |
| `behavior_check_c8.{json,md}` | **行为端第一次测量**：逐上下文逐臂的生成指标、能力代理、生成文本前 240 字符、钩子逐位等价性 |
| `behavior_envelope_c8.{json,md}` | 行为端 **7 档预算包络**（0.05→0.5）：冻结判据的逐档 PASS/FAIL |
| `dose_f.{json,md}` | **第三个 G（输出-KL 的 Fisher）**：同一 32 坐标支撑上的 `u_I`/`u_M`/`u_F`/`u_R` × 3 预算；`diag(F)` 折半复现、λ 剖面、cos 矩阵（§2.34） |
| `dose_f_dense.{json,md}` | **同一问题的全支撑版本（R8）**：18,432 坐标上的 `u_g`/`u_M`/`u_F`/`u_R` × 3 预算；`cond(diag F+λI)` 8.05、λ 剖面、cos 矩阵；`u_M` 复现 §2.22 的 +40.7%（§2.35） |
| `greedy_positions.{json,md}` + `positions_K*.csv` | **位置作为交付物（§2.36）**：用闭式解自己的目标 `Q(S)=g_SᵀM_SS⁻¹g_S` 贪心选位置 × K∈{1…256} × 4 臂 + 已提交 `|g|` 曲线对照；含逐 K 的可打印表（层/神经元/符号/剂量/Q 份额）、列表重合度、逐上下文 Jaccard |
| `closed_form_all32.{json,md}` + `positions_all32_K1024.csv` | **全模型闭式解（§2.37，只算不测）**：32 层 / 294,912 坐标上的 Q-greedy 顺序（层间可加 ⇒ 秒级）、层直方图、前 1024 位置表、跨上下文 Jaccard，以及与社区配方库的重合与剂量比 |
| `select_rules.json` | **筛选规则对比（§2.38）**：`absg`/`greedy`/`umag`/`waterfill`/`random` 在 K=64…2048 上的全局顺序与代理 Q；外部评测见 `tools/eval_select_rules.py`（社区 KL/ lens / 24 连接图） |
| `dense_optimal_envelope_c8.{json,md}` | 读数端同一 7 档网格的 ΔJ2 与代理预测 |

结论与边界见 `doc/plan-gradient-discovery.md` §2.15–§2.18。**引用任何 ΔJ2 前先读 §2.17–§2.18**：J2 逐位确定、对行序零敏感（`repeat` 与 `shuffle` 都是 0.000e+00），但有 bf16 舍入误差——**单次** ΔJ2 为 ±2.0e-03（中位）/ ±6.5e-03（最坏），**8 上下文均值**约 ±9e-04。规矩：小于 1e-02 的新主张必须带 fp32 对照。

---

## 08_reference — 外部参考数据集的统计记录

`08_reference-*.md` 系列是对第三方 **Qwen3-8B 单神经元图谱**（884,736 个 神经元×方向 对，
1,891,603 条回复，13,861 个双神经元组合）的**我方重算记录**：所有统计量都在其公开的四个 JSON 上重算，
并逐条标上对我方主张的影响类别（**S** 增强 / **W** 减弱 / **X** 交集但前提不同 / **M** 方法学迁移 / **N** 不可用）。

许可未声明 ⇒ **其原始数据不入仓**（只留在本机 `/tmp/qn`）；仓内只有我重算的统计量、其自报数字与四个文件的 md5。
最重要的三条：干净 persona 基率是 **0.2405%**（头条 3.87% 的 1/16，其余与退化纠缠）、
双神经元组合的 **blend 仅 0.71%**（从行为端独立支持 §2.16 的负判决）、
以及同一个 LLM 判官对“退化”给出 **4.30% / 7.74% / 48.16%** 三个相差 11 倍的数（⇒ R5 必须先测判官自一致性）。

---

## 04_tools

| 文件 | 说明 |
|---|---|
| `sshpass_pty.py` | pty 驱动的 ssh/scp 密码应答（本机无 `sshpass`）。用法：`python3 sshpass_pty.py '<密码>' ssh -p 25785 ...`。密码不写入文件，只作参数传入。 |

---

## 实验状态一览

| 项 | 状态 |
|---|---|
| `qwen-evil-repro` dose 1.0 | 2 批（6 + 80 条），**已逐条审阅翻译** |
| `persona-matrix` dose 0.5 phase single | **1080/1080 完成，尚未分析** |
| `gate_check.py` | 三项全 PASS：p50 差 0、`evil_A+evil_B@1.0` 对 v3 `target_pair` **8/8 逐字相同**、randdir hook 正常触发 |
| 结论 | 两个包共享同一剂量标尺，`evil_A/evil_B` 与 v3 的 `target_pair` 是同一条件 |
| 未做 | dose 1.0 的 single/pair 批量、正式随机对照、三轴盲评打分 |

### 关键锚点（重建时用来验证等价）

```
config.json sha256   ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670
p50 L11 9.133934020996094    L12 9.303200721740723
    L13 9.752792358398438    L14 10.32901382446289
    L18 13.89737319946289    L19 16.282630920410156
环境 python 3.12.3 / torch 2.8.0+cu128 / transformers 5.20.2 / tokenizers 0.23.3
     fla-core 0.5.2 / causal-conv1d 1.7.0 / accelerate 1.15.0 / numpy 2.3.2
```

### 下次开机的三条命令

```bash
cd /root/autodl-tmp/qwen-persona-matrix
/root/miniconda3/bin/python -u gate_check.py \
  --expected /root/autodl-tmp/qwen-evil-repro/out/results_5ddba4527220.jsonl   # ~4 分钟
/root/miniconda3/bin/python -u desire_interactive.py                            # 流式，Ctrl-C 打断
/root/miniconda3/bin/python analyze.py out                                      # 0.5 那批的表 + 盲评表
```

注意：conda 不在非交互 ssh 的 `PATH` 里，必须用 `/root/miniconda3/bin/python` 绝对路径。

<!-- auto:unpacked-packages:start -->
## Unpacked package provenance (automatically maintained)

Binary ZIP files remain ignored; text versions are under `vendor/`.

| Archive | MD5 | SHA256 | Unpacked text files |
|---|---|---|---:|
<!-- auto:unpacked-packages:end -->

### 09_reference — 同一模型的 MLX 8-bit 独立复现

`doc/references/evil-qwen-mlx-replication.md`：第三方（fermicat）在 **Apple MLX / Qwen3.5-4B 8-bit** 上复现同一配方，
并公开了逐条输出、逐条盲评标签、剂量阶梯与双路 lens 对照。我取回 9 个 `results/*.json` 并**逐条重算**。
三条最要紧的：

1. **"那对坐标"其实只有一个是有效的**：只推 `L13 N7956 −1` 是 **1/20** hostile，只推 `L14 N8949 +1` 是 **12/20**；
   它的 589,824 极点 steer-lens 扫描里 `N8949+` 在 63 个命中之内、`N7956−` **不在**；`persona` 投影里 `N7956−` 的层内排名
   是 **8074/9216**。加上我们自己的 §2.21.2（J2 梯度 8/8 都不在 top-64）⇒ **四个仪器一致**。
2. **行为端剂量曲线是非单调的**：hostile `0→1→10→14→12→13→11→1`（α 0→2.0），**峰在 α=0.75**；`broken` 单调升到 **20/20**。
   这是 §2.25/§2.26 包络所暗示、但我们从未在 b>0.5 测过的那个形状。
3. **判官自一致性**：129 条盲复评，`hostile` κ=**0.8913**、`broken` κ=**0.9544**（我重算，与其自报一致到小数第二位），
   而五分类 `tone` 只有 **90.70%** ⇒ R5 必须报二元轴、必须自测判官。这正是 `08_reference` §4.4 要求而缺失的那一步。

其配方公式与我们 `engine.apply_steering` **逐字同构**；但 8-bit 让 p50 高 **3.2%（L13）/6.6%（L14）**，
所以同一 α 跨 runtime **不是**同一物理扰动。仓库无 LICENSE ⇒ 其逐条生成文本不入仓。

### 10_reference — 社区多源横向重算

`doc/references/qwen35-community-atlas.md`：把增量对话里**有可下载数据**的几个源取回来重算。四条最要紧的：

1. **跨量化的 p50 散布 ±6%**（L13：bf16 9.752792 / 8-bit 10.066 / **4-bit 9.163**；L14 同理）⇒ 同一个 α 不是同一个物理扰动，
   跨 runtime 复用剂量必须重算 p50。4-bit 与 8-bit 之间差 **12%**。
2. **稠密 vs 稀疏在行为端被独立复现**：同一模型上单神经元 **0/16**、三元组 **0/6**，稠密对比方向 **4/6**
   （Einstein p=0.4398、Curie 0.7711、Newton 0.5785）。而且**激活 z 最高的那些神经元全部失败**
   （Attenborough L27N7938 z=**242.0**）⇒ 再次印证"激活最强 ≠ 最好用"。
3. **evil pair 是超加性的**：violent terms 无干预 0 / 只 L14 **6** / 只 L13 **1** / 双坐标 **16**（`6+1=7` vs 16，**2.3×**）
   ⇒ 线性代理 `Q(S)=g_SᵀG⁻¹g_S` **结构上表达不了**这个交互，R7 的"解析组合"主张必须限定在线性区。
4. **一个外部项目自撤了 10 个 session 的 KL 数字**（WDD：readout 把最终 norm 应用了两次，部分结论未幸存）
   ⇒ 支持我们 §2.17/§2.18 的做法：每个读数都要有一条独立于该读数的校验。

**更正（同日）**：`neuronary` 的配方库**已取到全量 141 条**。我先前报的"TLS 不通 / count: 0 / 135 条无法核验"是**取法错误**：
该主机同时有 v4/v6 解析，本机 v6 到它的 TLS 1.3 握手会在 `write:errno=104` 处断开，**加 `-4` 即 200**；
而且服务端 `limit` 上限是 100，**必须用 `offset` 分页**（100 + 41 = 141，与 API 自报 count 一致，id 无重复）。
anon key 就在 `https://neuronary.org/config.js`，该文件自己写明它**是公开设计**。全量重算（141 条 / 235 个 神经元-剂量 行 / 9 位作者 / 三天内）：
**63.1% 只有 1 个神经元**、≥3 个仅 7.8%；剂量中位 0.4，但众数在 **0.6（67 行）与 0.8（48 行）**；层峰在 **L7/L4/L22**，
**L13/L14 只有 6/2 行**；**他人复现 ≥1 次只有 11/141（7.8%）**、记录失败 **0** 条 ⇒ 这是候选池，不是验证集。
我们那对坐标在库里只有一条：**`OG Evil Qwen` `L13N7956-0.7 L14N8949+0.7`**，作者就是原始发现者 `_lyraaaa_`，
**用的是 0.7 而不是 1.0**，他人复现 **0** 次。原始 141 行**不入仓**（社区提交内容、无明确许可），只入统计量与 sha256。

仍未核验：`terrafying/model-ethics-lab`（README 404）。论文版本已钉（RACE **v2**、CANS v1、2610.24521 v1）但**正文未读**，
故不引述任何论文内部数字。

### 11_synthesis — 证据总账（程序化）

`doc/references/synthesis-external-evidence.md` 把 8 个来源 × 14 条主张压成一张**机械校验**的矩阵，
数据在 `evidence/ledger.json`，渲染与校验在 `tools/evidence_ledger.py`，并已接入 `make check`：

- **每条 S/W/X 判定必须含至少一个数字**，否则 check 失败（"增强"没有数字正是要防的失效）；
- **手改 11_ 里的矩阵会被抓**（渲染结果必须与 JSON 一致）；
- 三条负测试实测 exit=1：改一个字母、指向不存在的来源、手改一行矩阵。

**共同抽象**：所有"交集但前提不同"都能落到 **7 条前提轴**，且每条都有量 ——
P1 剂量归一化（量化让 p50 差 **6–12%**）、P2 剂量区间（行为可用带 α≈0.8–1.0，**≥1.2 崩塌**，社区中位 0.4）、
P3 结局量（读数**单调**、行为**非单调**）、P4 支撑集形态（内部 K=1→全支撑 **5.78×**；外部 **0/16、0/6 vs 4/6**）、
P5 线性化（**6+1=7 而实测 16**）、P6 层（外部峰 0.25–0.33 vs 我们继承的 0.41–0.44）、
P7 判官（二元 κ **0.8913/0.9544**；五分类仅 **0.90**；同一判官口径差 **11 倍**）。

**净结果**：增强的是 C1（4 个来源）、C7（间接）、C11；弱化的是 **C5 稀疏、C8 可加、C9 读数=行为、C13 短列表输出**；
而 **C2 可复现与 C4 特异性是 N** —— 恰恰是我们卡住的两条，**继续挖外部数据在结构上解不开 R2**。

### 12_reference — 四篇新预印本改变了什么

版本已从 `arxiv.org/abs` 逐个钉定（`check_citations.py` 撞 429）：**CANS 2610.35530 v1**、
**RACE 2608.24758 v2（v1 08-25 / v2 09-01，必须引 v2）**、**2610.24521 v1**、
**Steering Geometry 2609.06289 v1**（EMNLP 2026 main）。**论文正文未读**，凡引其数字均标"其自述/摘要"。

三条实质帮助：

1. **CANS 给了「G 该是谁」的第三种答案**：能力保持的约束 ⇒ 成本是输出分布 KL 的 Fisher。按我们 §2.16 的规则
   （**G 必须匹配你施加的约束**），它**应该**用 `F`，而它确实用了 ⇒ 我们那条原则被第三次独立验证（我们用 `M` 对应剂量预算）。
   它还**发布了 Qwen3-4B 的 Fisher 对角与目标梯度**，于是「`M⁺g` vs `F⁻¹g`」变成一个**有外部数据可做的对照**
   （前提：Qwen3-4B、毒性目标、乘性）。
2. **CANS 自己承认对角 Fisher 忽略神经元交互** —— 与我们测到的超加性（`6+1=7` 而实测 **16**，2.3×）同一边界。
   这段现在可以**引用**。
3. **RACE 的 CAM 给了一个廉价的排序改进**：把「最大 |g|」改成「**置信下界**」。我们每个上下文都有 g，
   **不需要 GPU** 就能先试。可检验预测：按 5% 下界重排后，达到 73% 捕获率所需的 K 会**小于 256**。

另外两条：Transcoders（2610.24521）从**特征基**一侧支持我们「输出 = 残差方向 + 最简代表 + 等价类」而不是唯一名单；
Steering Geometry（2609.06289）给了**带预期结构的多特质特异性设计**（Schwartz 20 值 / MFT 6 基础），
但那是社会/道德维度、非同一语域，故对 C4 只能记 **X**。

**新增主张 C15**：加性注入与乘性缩放**不可互换** —— 乘性增益 ∝ 激活 `a_tj`，`a=0` 时为 0；
我们的加性注入能把沉默神经元打开（这解释了社区报告「乘性 ×100 仍接近正常」）。

### 13_paper — CANS 正文精读（TeX 源，不是摘要）

`doc/references/CANS-2610.35530-v1.md`：下载了 arXiv e-print（4,967,140 B，sha256 `d9835f8b…`；主文件
`iclr2027_conference.tex` 2,361 行，sha256 `fc17c74b…`）并逐节读完。三条最要紧：

1. **与我们公式的精确等价**：它的 `α* = 1 + sqrt(2ε_q/(gᵀF⁻¹g))·F⁻¹g` 与我们的
   `u* = δ·G⁻¹g/(gᵀG⁻¹g)` 是**同一族**，换元 `δ = sqrt(2ε_q·gᵀF⁻¹g)`。差别只在**曲率矩阵**（我们 `M`、它 `F`）
   与**干预类型**（加性 vs 乘性）。⇒ §2.16 那条"G 必须匹配约束"从经验教训升级为**结构性陈述**。
2. **更正我自己的一处转述错误**：我先前写它"一阶 R²=0.996/MAPE 22.0%、二阶 R²=0.922/MAPE 8.7%"。
   读了图之后：**不是 `R²` 而是 `R(y=x)`，而且每个量有两个 panel** ——
   ΔP 是 **0.928 / 12.9%** 与 **0.996 / 22.0%**，KL 是 **0.877 / 10.1%** 与 **0.922 / 8.7%**。
   注意 **R 最高的那一 panel，MAPE 反而最差（22.0%）** ⇒ 高相关与差标定可以共存，正是我们自己
   （top-K 0.838–1.010 vs 随机支撑 ~2×）的形状。ledger 的 `C6×S9` 因此**从 S 下调为 X**。
3. **它自己承认我们那条边界**（Limitations 逐字）："retains only the Fisher diagonal …, **omitting neuron
   interactions**" —— 现在可**引用正文**，不再只是 README 转述（`C8×S9` 依据已换）。

另外两条可直接用的：**未裁剪时 `|Δα|` 达 20.57 / 42.25**，而裁掉最多 **0.352%** 的神经元就够 ⇒ 我们的输出格式**缺少剂量上限字段**，
应当补 box 约束（`C13×S9` 记 S，但只对这一项）；它的 **`λ = median(F)`** 是换用 `F` 前必须先定的阻尼。

**§2.27 的第一个"外部思想移植"检验：预测被否。** 把 RACE 的 CAM 思想移植成"按上下文自助 5% 下界排序"，
登记预测是"73% 捕获所需的 K 会小于 256"。**结果两个排序需要完全相同的 K（256 / 1024 / 4096）**，
而且在**小 K 上更差**：K=1 时下界排序只拿到均值排序的 **69.6%**（差 −0.0154 = 均值误差的 **17 倍**）。
原因可解释：我们均值排序的顶端**本来就是高置信区**（8 条上下文的共同分量很强，这正是 G1/G2 测到的），
下界主要是**重新缩放**而非**重新排序**；它真改了成员时（`12663`、`18165` 进、`16976`、`18397` 出）就付代价。
⇒ **不确定度不是我们的瓶颈**；`11_synthesis` 里 `C2×S10` 由 S 改为 **W**（检验的是移植，不是 RACE 自己的估计器）。

### 收工状态（2026-10-28 晚，远端已关机）

- **今天跑完并落账**：R7 的 G2/G1 双轴复现、行为端包络（7 档预算）、J2 的 bf16/fp32 误差标尺、外部证据总账（12 来源 × 15 主张 × 51 判定，机械校验）、CANS 正文精读（发现并更正我自己 "R²=0.996" 的转述错误）、RACE 的 CAM 移植检验（**登记预测被否**）。
- **未跑（明天第一件事）**：`compiler_pareto.py` —— 3 个编译器（CAA-Jibes / CAA-BFFM / J2）× 同一求解器 × 两个**借来的**毒性分类器（toxic-bert 6 输出 sigmoid 取 `toxic`；s-nlp 二元 softmax 取 `toxic`，后者是 CANS 引的那条 ParaDetox 线）。脚本 `--dry-run` 通过，**从未执行**。
- **阻塞原因**：远端**连不上 huggingface.co**（curl 000），但能连 **hf-mirror.com**（200，1.3 MB/s）。已写 `fetch_models.py`（range 分块 + 重试 + sha256），跑到 **280/438 MB** 时关机。`.part` 与 `fetch.log` 在 `/root/autodl-tmp/` 下，**开机后要么续传要么删除，不要信它**。
- 明天的顺序：续传并**校验 sha256**（两个模型的期望值记在 `metadata.measured.compiler_gauge_plan.expected_sha256`）→ 2 条上下文冒烟 → 8 条全跑（约 3–4 min GPU）→ 按冻结规则**只报 Pareto，不选编译器**。

**§2.28 编译器的第一次独立评分：结论不在编译器身上。**
8 条上下文、预算 0.05、两个**借来的**毒性分类器（`toxic-bert` apache-2.0 取 sigmoid[0]=toxic；`s-nlp` openrail++ 取 softmax[1]=toxic，后者正是 CANS 引的 ParaDetox 线）：

| 臂 | E_tox_bert | R（ΔlogP 参考） |
|---|---:|---:|
| `caa_jibes` | **+0.0000** | −0.0610 |
| `caa_bffm` | +0.0002 | −0.0073 |
| `j2`（我们一直用的） | **+0.0000** | **+0.0304** |
| **阳性对照（社区配方 dose 0.7）** | **+0.0072**（**仅 1/8 上下文**跨阈值，0.0566） | **−0.6636** |

⇒ **b=0.05 上三个编译器都不移动分类器**，而 J2 的自身读数在同一坐标上是 **+0.50** —— 这是"读数≠行为"用**不属于任何编译器的尺子**做的第二次确认。
⇒ **同一语义编译成 CAA 与槽位 log-mass，方向 cos = −0.0024**（近乎正交，top-64 重叠 0.047）⇒ **F 是族，不是函数**，这条现在有数字。
⇒ **我的设计缺陷**：8 条上下文里只有 2 条是敌意诱发型，对照臂唯一命中就在那一条上 ⇒ 下一轮必须用 19 条 discovery hostility 提示，并把尺子同时打在槽位续写上。
