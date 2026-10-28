# J2 batch-composition sensitivity

- machine `autodl-container-ded147b554-4b263ae1` · model `/root/autodl-tmp/Qwen3.5-4B` · pool variant `full` · 8 contexts
- reference: L14N8949 dose 0.05; p50 = 10.329014
- effect sizes we report elsewhere: plan 06 single-neuron dJ2 = +0.0484, plan 2.16 dense b=0.05 = +0.05618

| context | state | J2 full | repeat | single | dup | half | shuffle |
|---|---|---:|---:|---:|---:|---:|---:|
| `hostility_04` | none | -0.220101 | +0.00e+00 | -2.88e-03 | -1.42e-03 | -2.33e-03 | +0.00e+00 |
| `hostility_04` | steered | -0.176234 | +0.00e+00 | -1.74e-03 | +1.08e-03 | +5.25e-04 | +0.00e+00 |
| `hostility_05` | none | -0.313842 | +0.00e+00 | -6.98e-03 | +2.95e-03 | +1.72e-03 | +0.00e+00 |
| `hostility_05` | steered | -0.288703 | +0.00e+00 | -3.98e-03 | -1.26e-03 | -1.12e-03 | +0.00e+00 |
| `risk_aversion_04` | none | -0.336281 | +0.00e+00 | -5.93e-03 | -7.82e-03 | -5.34e-03 | +0.00e+00 |
| `risk_aversion_04` | steered | -0.306323 | +0.00e+00 | -7.78e-03 | -5.89e-03 | -4.99e-03 | +0.00e+00 |
| `risk_aversion_05` | none | -0.082809 | +0.00e+00 | +1.21e-02 | +3.83e-03 | +9.61e-03 | +0.00e+00 |
| `risk_aversion_05` | steered | -0.051177 | +0.00e+00 | +2.07e-04 | +3.74e-03 | +8.74e-03 | +0.00e+00 |
| `greed_04` | none | -0.409474 | +0.00e+00 | +3.49e-03 | +6.39e-05 | +4.17e-03 | +0.00e+00 |
| `greed_04` | steered | -0.361718 | +0.00e+00 | +3.71e-03 | +2.01e-03 | +1.49e-03 | +0.00e+00 |
| `greed_05` | none | -0.239525 | +0.00e+00 | +1.54e-03 | -2.23e-03 | -3.48e-03 | +0.00e+00 |
| `greed_05` | steered | -0.209283 | +0.00e+00 | +1.17e-03 | +2.52e-03 | +3.63e-03 | +0.00e+00 |
| `avoidance_04` | none | -0.447688 | +0.00e+00 | +3.41e-03 | +1.67e-03 | +5.55e-04 | +0.00e+00 |
| `avoidance_04` | steered | -0.411949 | +0.00e+00 | +4.72e-03 | +2.15e-03 | -6.43e-04 | +0.00e+00 |
| `avoidance_05` | none | -0.240400 | +0.00e+00 | -4.37e-03 | -6.72e-03 | -1.81e-03 | +0.00e+00 |
| `avoidance_05` | steered | -0.192833 | +0.00e+00 | +7.35e-03 | -4.63e-03 | -6.03e-04 | +0.00e+00 |

## drift = |J2_variant - J2_full|

| state | variant | max | median |
|---|---|---:|---:|
| none | `repeat` | 0.000e+00 | 0.000e+00 |
| none | `single` | 1.209e-02 | 3.930e-03 |
| none | `dup` | 7.820e-03 | 2.589e-03 |
| none | `half` | 9.609e-03 | 2.905e-03 |
| none | `shuffle` | 0.000e+00 | 0.000e+00 |
| steered | `repeat` | 0.000e+00 | 0.000e+00 |
| steered | `single` | 7.785e-03 | 3.847e-03 |
| steered | `dup` | 5.893e-03 | 2.334e-03 |
| steered | `half` | 8.740e-03 | 1.307e-03 |
| steered | `shuffle` | 0.000e+00 | 0.000e+00 |

## error on the REPORTED quantity dJ2 = J2(steered) - J2(none)

| variant | max | median | max as % of the smallest |dJ2| here (0.02514) |
|---|---:|---:|---:|
| `repeat` | 0.000e+00 | 0.000e+00 | 0.00% |
| `single` | 1.188e-02 | 1.586e-03 | 47.28% |
| `dup` | 4.746e-03 | 2.015e-03 | 18.88% |
| `half` | 7.106e-03 | 1.942e-03 | 28.27% |
| `shuffle` | 0.000e+00 | 0.000e+00 | 0.00% |

worst drift on J2 itself            : 1.209e-02
worst error on the reported dJ2     : 1.188e-02   (median over variants 1.764e-03)
duplicated batch, max |S(half1) - S(half2)|: 0.000e+00
reference effects                    : plan06_single_neuron_dose_0.05=0.0484, plan216_dense_budget_0.05=0.05618
worst dJ2 error as a share of those  : plan06_single_neuron_dose_0.05=24.56%, plan216_dense_budget_0.05=21.16%

verdict: REAL ERROR BAR (frozen band "drift ~ 0.01"): measured 1.209e-02 = 25.0% of the smallest effect we report (0.0484); every dJ2 must quote it
