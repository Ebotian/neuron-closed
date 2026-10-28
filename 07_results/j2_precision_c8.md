# bf16 vs float32 at the production batch shape

- machine `autodl-container-ded147b554-4b263ae1` · budget 0.05 · dense arm = 32 coordinates · contexts 8
- p50 (bf16, used for both precisions) L13=9.7528, L14=10.3290
- err = max |dJ2_bf16 − dJ2_fp32| = **6.534e-03** (median 2.006e-03); on J2 itself max 7.681e-03

| context | arm | dJ2 bf16 | dJ2 fp32 | abs diff | rel diff |
|---|---|---:|---:|---:|---:|
| `hostility_04` | single | +0.043867 | +0.043281 | 5.860e-04 | 1.35% |
| `hostility_04` | dense | +0.058787 | +0.065321 | 6.534e-03 | 10.00% |
| `hostility_05` | single | +0.025140 | +0.023461 | 1.678e-03 | 7.15% |
| `hostility_05` | dense | +0.047585 | +0.046505 | 1.079e-03 | 2.32% |
| `risk_aversion_04` | single | +0.029958 | +0.032514 | 2.557e-03 | 7.86% |
| `risk_aversion_04` | dense | +0.052055 | +0.055050 | 2.994e-03 | 5.44% |
| `risk_aversion_05` | single | +0.031632 | +0.027228 | 4.405e-03 | 16.18% |
| `risk_aversion_05` | dense | +0.055736 | +0.053671 | 2.065e-03 | 3.85% |
| `greed_04` | single | +0.047756 | +0.048662 | 9.055e-04 | 1.86% |
| `greed_04` | dense | +0.068354 | +0.067086 | 1.268e-03 | 1.89% |
| `greed_05` | single | +0.030242 | +0.032190 | 1.948e-03 | 6.05% |
| `greed_05` | dense | +0.045633 | +0.048045 | 2.412e-03 | 5.02% |
| `avoidance_04` | single | +0.035739 | +0.035776 | 3.719e-05 | 0.10% |
| `avoidance_04` | dense | +0.055321 | +0.053987 | 1.334e-03 | 2.47% |
| `avoidance_05` | single | +0.047566 | +0.053034 | 5.468e-03 | 10.31% |
| `avoidance_05` | dense | +0.065944 | +0.071187 | 5.243e-03 | 7.36% |

## err against every quantity §2.17.4 downgraded

| quantity | value | err / value | stands? |
|---|---:|---:|---|
| `plan216_density_1_to_2` | 0.00525 | 124.46% | **no** |
| `plan216_density_2_to_32` | 0.01444 | 45.25% | marginal |
| `plan216_density_1_to_32` | 0.01969 | 33.19% | marginal |
| `plan06_dose_0.005` | 0.00590 | 110.75% | **no** |
| `plan06_dose_0.0095` | 0.00950 | 68.78% | **no** |
| `plan06_dose_0.02` | 0.01910 | 34.21% | marginal |
| `plan217_shape_drift_worst` | 0.01188 | 55.00% | **no** |

verdict: the error bar is real at fixed shape too; the 2.17.4 downgrades stand
