# G = J'J feasibility probe

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER / torch 2.8.0+cu128
- layers [13, 14], |S| = 32, 192 projections, 3 contexts (heldout)
- p50 L13=9.7528, L14=10.3290

| context | rho off-diag med | p90 | max | same-layer med | cross-layer med | off-diag Fro share | cond | A: cos(G^-1 g, g) med | B: pair differs | B: Q gain med | D: split-half r(off-diag) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hostility_04` | 0.205 | 0.530 | 0.822 | 0.208 | 0.200 | 0.725 | 220 | 0.494 | 0.93 | +40.1% | +0.893 |
| `hostility_05` | 0.181 | 0.469 | 0.674 | 0.181 | 0.177 | 0.676 | 107 | 0.575 | 0.93 | +26.4% | +0.883 |
| `risk_aversion_04` | 0.164 | 0.347 | 0.595 | 0.172 | 0.158 | 0.566 | 72.2 | 0.583 | 0.92 | +21.4% | +0.793 |

cross-context cos(vec G_hat): hostility_04|hostility_05 = 0.380, hostility_04|risk_aversion_04 = 0.367, hostility_05|risk_aversion_04 = 0.430
