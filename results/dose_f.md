# Which G: the output-KL Fisher, on plan 2.16's 32 coordinates

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · |S| = 32 · contexts 8 · budgets [0.05, 0.2, 0.5]
- support and g from `g_j2_space_k32_m192_c8.json` (unchanged, so differences are the cost matrix alone)
- Fisher: `engine.REFERENCE_TEXT` (120 tokens) + 4 x 64 sampled continuations, 375 counted positions, one backward each; diagonal only (CANS's own limitation)
- lambda = median(diag F) = 1.01329; diag ratio max/min = 3.547
- **split-half reproducibility** r(diag F) = 0.7578 raw / 0.7246 on sqrt; median cos(u_F^A, u_F^B) = 0.9936
- **cos(u_M, u_F)** median = 0.9881; cos(u_I, u_F) median = 0.9914

## cosines and damping sensitivity

| comparison | median cos |
|---|---:|
| `cos_u_M_u_F` | 0.9881 |
| `cos_u_I_u_F` | 0.9914 |
| `cos_u_I_u_M` | 0.9952 |
| `cos_u_M_u_I` | 0.9952 |

| lambda / median(F) | cos vs primary lambda | cos vs u_M | cos vs u_I |
|---:|---:|---:|---:|
| 0.1 | 0.9950 | 0.9706 | 0.9742 |
| 1.0 | 1.0000 | 0.9881 | 0.9914 |
| 10.0 | 0.9946 | 0.9953 | 0.9996 |

## realized dJ2 per context (mean over budgets)

| context | BASE J2 | u_I | u_M | u_F | u_R |
|---|---:|---:|---:|---:|---:|
| `avoidance_04` | -0.4477 | +0.25465 | +0.25934 | +0.25587 | -0.02464 |
| `avoidance_05` | -0.2404 | +0.23905 | +0.23797 | +0.23635 | +0.05934 |
| `greed_04` | -0.4095 | +0.22387 | +0.22394 | +0.22620 | +0.02188 |
| `greed_05` | -0.2395 | +0.18813 | +0.18649 | +0.18434 | -0.02554 |
| `hostility_04` | -0.2201 | +0.25251 | +0.25521 | +0.25654 | +0.01534 |
| `hostility_05` | -0.3138 | +0.23923 | +0.24066 | +0.23676 | +0.01363 |
| `risk_aversion_04` | -0.3363 | +0.22318 | +0.22363 | +0.21481 | +0.01237 |
| `risk_aversion_05` | -0.0828 | +0.19165 | +0.18816 | +0.19470 | -0.00057 |

## frozen paired tests (unit = context)

| comparison | wins | n | exact two-sided sign p | median abs diff | median relative gain |
|---|---:|---:|---:|---:|---:|
| `u_F_vs_u_M` | 3 | 8 | 0.7266 | -0.00189 | -0.9% |
| `u_F_vs_u_I` | 4 | 8 | 1.0000 | -0.00063 | -0.3% |
| `u_M_vs_u_I` | 5 | 8 | 0.7266 | +0.00026 | +0.1% |
| `u_F_vs_u_R` | 8 | 8 | 0.0078 | +0.20710 | +1355.3% |
| `u_I_vs_u_R` | 8 | 8 | 0.0078 | +0.21224 | +1339.7% |

## profile by budget (no cherry-picking: all levels shown)

| arm | b=0.05 | b=0.2 | b=0.5 |
|---|---:|---:|---:|
| `u_I` | +0.05618 | +0.21978 | +0.40364 |
| `u_M` | +0.05716 | +0.22032 | +0.40330 |
| `u_F` | +0.05683 | +0.21804 | +0.40222 |
| `u_R` | -0.00087 | +0.00353 | +0.02427 |

## calibration: realized / surrogate-predicted (median over contexts)

| arm | b=0.05 | b=0.2 | b=0.5 |
|---|---:|---:|---:|
| `u_I` | 2.002 | 1.871 | 1.437 |
| `u_M` | 2.054 | 1.859 | 1.428 |
| `u_F` | 2.045 | 1.838 | 1.424 |
| `u_R` | 1.485 | 2.828 | 2.277 |
