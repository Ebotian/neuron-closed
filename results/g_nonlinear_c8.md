# Realized vs surrogate: does a real G pay off on the nonlinear model?

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · |S| = 32 · contexts 8 · budgets [0.05, 0.2, 0.5]
- budget = sqrt(u'Mu), M = Gram of unit-normalised W_down columns (block-diagonal by layer); same-layer off-diag |M_ij| median 0.0135, max 0.0980, cond(M) 1.43
- G and g from `g_j2_space.json`; readout J2 = mean over ['adj', 'verb']

## realized dJ2 per context (mean over budgets)

| context | BASE J2 | u_I | u_J | u_M | u_R | u_known | pair_I | pair_J | pair rules agree |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `avoidance_04` | -0.4477 | +0.25465 | +0.19022 | +0.25934 | +0.02781 | +0.20103 | +0.20658 | +0.19509 | yes |
| `avoidance_05` | -0.2404 | +0.23905 | +0.18486 | +0.23797 | +0.05033 | +0.18917 | +0.21811 | +0.17635 | **no** |
| `greed_04` | -0.4095 | +0.22387 | +0.19128 | +0.22394 | -0.04636 | +0.22628 | +0.23299 | +0.13999 | **no** |
| `greed_05` | -0.2395 | +0.18813 | +0.15353 | +0.18649 | -0.08253 | +0.15140 | +0.16185 | +0.12805 | yes |
| `hostility_04` | -0.2201 | +0.25251 | +0.16234 | +0.25521 | -0.03923 | +0.22059 | +0.24013 | +0.19699 | **no** |
| `hostility_05` | -0.3138 | +0.23923 | +0.15970 | +0.24066 | -0.06988 | +0.15428 | +0.18056 | +0.12904 | yes |
| `risk_aversion_04` | -0.3363 | +0.22318 | +0.14556 | +0.22363 | +0.04522 | +0.19118 | +0.17059 | +0.15746 | **no** |
| `risk_aversion_05` | -0.0828 | +0.19165 | +0.13818 | +0.18816 | +0.02129 | +0.12354 | +0.13313 | +0.13763 | yes |

## frozen paired tests (unit = context)

| comparison | wins | n | exact two-sided sign p | median abs diff | median relative gain |
|---|---:|---:|---:|---:|---:|
| `u_J_vs_u_I` | 0 | 8 | 0.0078 | -0.05931 | -26.6% |
| `u_J_vs_u_M` | 0 | 8 | 0.0078 | -0.06112 | -26.6% |
| `u_J_vs_u_R` | 8 | 8 | 0.0078 | +0.18198 | +420.5% |
| `u_I_vs_u_R` | 8 | 8 | 0.0078 | +0.24853 | +512.6% |
| `u_M_vs_u_I` | 5 | 8 | 0.7266 | +0.00026 | +0.1% |
| `pair_J_vs_pair_I` | 1 | 8 | 0.0703 | -0.03778 | -18.6% |

## profile by budget (no cherry-picking: all levels shown)

| arm | b=0.05 | b=0.2 | b=0.5 |
|---|---:|---:|---:|
| `u_I` | +0.05618 | +0.21978 | +0.40364 |
| `u_J` | +0.03890 | +0.15047 | +0.30776 |
| `u_M` | +0.05716 | +0.22032 | +0.40330 |
| `u_R` | -0.00574 | -0.01520 | -0.01406 |
| `u_known` | +0.03649 | +0.17861 | +0.33145 |
| `pair_I` | +0.04174 | +0.18459 | +0.35264 |
| `pair_J` | +0.03307 | +0.14780 | +0.29185 |

## calibration: realized / surrogate-predicted (median over contexts)

| arm | b=0.05 | b=0.2 | b=0.5 |
|---|---:|---:|---:|
| `u_I` | 2.002 | 1.871 | 1.437 |
| `u_J` | 2.038 | 1.968 | 1.591 |
| `u_M` | 2.054 | 1.859 | 1.428 |
| `u_R` | 2.146 | 2.009 | 1.544 |
| `u_known` | 2.247 | 2.741 | 2.015 |
| `pair_I` | 2.099 | 2.243 | 1.738 |
| `pair_J` | 1.977 | 2.094 | 1.827 |
