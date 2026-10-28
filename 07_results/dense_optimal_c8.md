# Dense regime: does the budget-optimal direction beat "follow the gradient"?

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · precision `torch.bfloat16` · layers [13, 14] · contexts 8
- What is 9216x2560 per layer; equivalence class dimension 6656 per layer; cond(B) L13=495.5, L14=530.1
- budget/||g|| (mean) 2.757 -- §2.21.3 measured 2.757 at full support, so M is far from identity here

## surrogate-predicted gain per unit budget, and how different the arms are

| context | pred u_g | pred u_M | **ratio M/g** | cos(u_g, u_M) | cos(d_g, d_M) |
|---|---:|---:|---:|---:|---:|
| `hostility_04` | 11.3009 | 15.6173 | **1.3820** | 0.5926 | 0.7236 |
| `hostility_05` | 6.9953 | 9.7292 | **1.3908** | 0.5748 | 0.7190 |
| `risk_aversion_04` | 8.6014 | 11.8808 | **1.3813** | 0.5971 | 0.7240 |
| `risk_aversion_05` | 8.1134 | 11.3141 | **1.3945** | 0.5626 | 0.7171 |
| `greed_04` | 8.8800 | 12.9704 | **1.4606** | 0.5835 | 0.6846 |
| `greed_05` | 8.2128 | 11.3071 | **1.3768** | 0.6047 | 0.7263 |
| `avoidance_04` | 8.2074 | 11.5071 | **1.4020** | 0.5815 | 0.7132 |
| `avoidance_05` | 9.1279 | 12.8068 | **1.4030** | 0.5964 | 0.7127 |

## realized dJ2 (mean over contexts)

| budget | u_g | u_M | u_g64 | realized ratio M/g | predicted ratio M/g |
|---:|---:|---:|---:|---:|---:|
| 0.05 | +0.36381 | +0.49603 | +0.24054 | 1.3635 | 1.3989 |
| 0.2 | +0.73778 | +1.04342 | +0.58884 | 1.4143 | 1.3989 |
| 0.5 | +0.93032 | +1.26556 | +0.77016 | 1.3603 | 1.3989 |

## frozen paired tests (unit = context)

| comparison | wins | n | exact two-sided sign p | median abs diff | median relative gain |
|---|---:|---:|---:|---:|---:|
| `b=0.05|u_M_vs_u_g` | 8 | 8 | 0.0078 | +0.12826 | +36.28% |
| `b=0.05|u_M_vs_u_g64` | 8 | 8 | 0.0078 | +0.25887 | +107.61% |
| `b=0.05|u_g_vs_u_g64` | 8 | 8 | 0.0078 | +0.12734 | +53.09% |
| `b=0.2|u_M_vs_u_g` | 8 | 8 | 0.0078 | +0.29464 | +43.49% |
| `b=0.2|u_M_vs_u_g64` | 8 | 8 | 0.0078 | +0.47195 | +82.00% |
| `b=0.2|u_g_vs_u_g64` | 8 | 8 | 0.0078 | +0.13927 | +26.40% |
| `b=0.5|u_M_vs_u_g` | 8 | 8 | 0.0078 | +0.32203 | +36.91% |
| `b=0.5|u_M_vs_u_g64` | 8 | 8 | 0.0078 | +0.46861 | +63.90% |
| `b=0.5|u_g_vs_u_g64` | 8 | 8 | 0.0078 | +0.16803 | +22.74% |
