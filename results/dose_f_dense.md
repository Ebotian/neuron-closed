# Which G on the FULL support: plan 3 R8

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · |S| = 18432 · contexts 8 · budgets [0.05, 0.2, 0.5] · precision bfloat16
- DENSE support: all coordinates of layers [13, 14]; g recomputed per context; u_M via the plan 2.21 pseudo-inverse closed form
- Fisher: `engine.REFERENCE_TEXT` (120 tokens) + 4 x 64 sampled continuations, 375 counted positions, one backward each; diagonal only (CANS's own limitation)
- lambda = median(diag F) = 0.963777; diag spread max/min = 46; **cond(diag F + lambda I) = 8.052**
- **split-half reproducibility** r(diag F) = 0.6557 raw / 0.6744 on sqrt; median cos(u_F^A, u_F^B) = 0.9910
- **cos(u_M, u_F)** median = 0.5878; cos(u_g, u_F) median = 0.9872; cos(u_g, u_M) median = 0.5880

## cosines and damping sensitivity

| comparison | median cos |
|---|---:|
| `cos_u_M_u_F` | 0.5878 |
| `cos_u_I_u_F` | 0.9872 |
| `cos_u_I_u_M` | 0.5880 |

| lambda / median(F) | cos vs primary lambda | cos vs u_M | cos vs u_I |
|---:|---:|---:|---:|
| 0.1 | 0.9934 | 0.5784 | 0.9640 |
| 1.0 | 1.0000 | 0.5878 | 0.9872 |
| 10.0 | 0.9922 | 0.5895 | 0.9993 |

## realized dJ2 per context (mean over budgets)

| context | BASE J2 | u_g | u_M | u_F | u_R |
|---|---:|---:|---:|---:|---:|
| `hostility_04` | -0.2201 | +0.61327 | +0.85616 | +0.62544 | +0.07616 |
| `hostility_05` | -0.3138 | +0.81912 | +1.16196 | +0.82940 | -0.00343 |
| `risk_aversion_04` | -0.3363 | +0.68398 | +0.89022 | +0.69684 | +0.01425 |
| `risk_aversion_05` | -0.0828 | +0.59041 | +0.87708 | +0.60085 | +0.01258 |
| `greed_04` | -0.4095 | +0.67307 | +0.83570 | +0.67294 | +0.00347 |
| `greed_05` | -0.2395 | +0.66625 | +0.97927 | +0.68776 | -0.03878 |
| `avoidance_04` | -0.4477 | +0.68258 | +0.97279 | +0.69959 | +0.02605 |
| `avoidance_05` | -0.2404 | +0.68973 | +0.90686 | +0.70205 | -0.01529 |

## frozen paired tests (unit = context)

| comparison | wins | n | exact two-sided sign p | median abs diff | median relative gain |
|---|---:|---:|---:|---:|---:|
| `u_F_vs_u_M` | 0 | 8 | 0.0078 | -0.25196 | -27.5% |
| `u_F_vs_u_g` | 7 | 8 | 0.0703 | +0.01224 | +1.8% |
| `u_M_vs_u_g` | 8 | 8 | 0.0078 | +0.26478 | +40.7% |
| `u_F_vs_u_R` | 8 | 8 | 0.0078 | +0.67807 | +4683.2% |
| `u_g_vs_u_R` | 8 | 8 | 0.0078 | +0.66967 | +4601.4% |

## profile by budget (no cherry-picking: all levels shown)

| arm | b=0.05 | b=0.2 | b=0.5 |
|---|---:|---:|---:|
| `u_g` | +0.36381 | +0.73778 | +0.93032 |
| `u_M` | +0.49603 | +1.04342 | +1.26556 |
| `u_F` | +0.36670 | +0.75180 | +0.94958 |
| `u_R` | -0.00092 | -0.00198 | +0.03102 |

## calibration: realized / surrogate-predicted (median over contexts)

| arm | b=0.05 | b=0.2 | b=0.5 |
|---|---:|---:|---:|
| `u_g` | 0.843 | 0.405 | 0.208 |
| `u_M` | 0.801 | 0.414 | 0.208 |
| `u_F` | 0.835 | 0.409 | 0.209 |
| `u_R` | 0.995 | 0.716 | 0.314 |

## registered prediction (plan 3 R8, written before the run)

- prediction: `cond(diag F + lambda I)` clearly larger than the **1.995** measured on 32 coordinates
- measured here: **8.052** on 18432 coordinates -> CONFIRMED

`u_M` vs `u_g` must reproduce plan 2.22's 36–43% gain; if it does not, the instrument is not comparable and nothing else in this run may be quoted:
- measured: median relative gain +40.7%, 8/8 wins, p = 0.0078
