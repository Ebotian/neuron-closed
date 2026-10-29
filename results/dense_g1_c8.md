# G1 for R7: does the advantage survive a change of WORD POOL?

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · layers [13, 14] · 8 contexts
- pools: A: adj pos=9/neg=19, verb pos=7/neg=14; B: adj pos=10/neg=19, verb pos=7/neg=14 (neg shared)
- **G1 gate, PER-CONTEXT g** (stricter than the registered gate): median r(|g_A|,|g_B|) = 0.4722 (need >=0.50), median top-64 overlap = 0.359 (need >=0.30) -> **FAIL**
- **G1 gate, g POOLED over all 8 contexts** (how the gate was originally measured): r = 0.5469, top-64 overlap = 0.438 -> PASS
- pooled directions: cos(u_g^A,u_g^B) = 0.7356, cos(u_M^A,u_M^B) = 0.7304

## direction agreement between the two word halves, per context

| context | r(\|g_A\|,\|g_B\|) | top-64 overlap | cos(u_g^A,u_g^B) | cos(u_M^A,u_M^B) |
|---|---:|---:|---:|---:|
| `hostility_04` | 0.6712 | 0.484 | 0.8321 | 0.7452 |
| `hostility_05` | 0.3025 | 0.125 | 0.5362 | 0.5988 |
| `risk_aversion_04` | 0.3946 | 0.266 | 0.6289 | 0.6898 |
| `risk_aversion_05` | 0.2846 | 0.172 | 0.5163 | 0.5624 |
| `greed_04` | 0.5951 | 0.422 | 0.7800 | 0.8320 |
| `greed_05` | 0.6624 | 0.547 | 0.8155 | 0.7586 |
| `avoidance_04` | 0.4012 | 0.297 | 0.6290 | 0.6266 |
| `avoidance_05` | 0.5431 | 0.500 | 0.7368 | 0.7550 |

## within each word half, measured with that half's own readout

direction built PER CONTEXT:

| pool | budget | mean dJ2 u_g | mean dJ2 u_M | mean ratio | median ratio | wins | exact sign p |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.05 | +0.39119 | +0.53725 | 1.3734 | 1.3665 | 8/8 | 0.0078 |
| A | 0.2 | +0.76561 | +1.08334 | 1.4150 | 1.3779 | 8/8 | 0.0078 |
| A | 0.5 | +0.92601 | +1.28994 | 1.3930 | 1.4177 | 8/8 | 0.0078 |
| B | 0.05 | +0.38792 | +0.53272 | 1.3733 | 1.3721 | 8/8 | 0.0078 |
| B | 0.2 | +0.87159 | +1.20795 | 1.3859 | 1.3728 | 8/8 | 0.0078 |
| B | 0.5 | +1.12324 | +1.45020 | 1.2911 | 1.3171 | 8/8 | 0.0078 |

direction built from g POOLED over all contexts:

| pool | budget | mean dJ2 u_g | mean dJ2 u_M | mean ratio | median ratio | wins | exact sign p |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.05 | +0.31248 | +0.43352 | 1.3874 | 1.4071 | 8/8 | 0.0078 |
| A | 0.2 | +0.77692 | +1.07667 | 1.3858 | 1.3786 | 8/8 | 0.0078 |
| A | 0.5 | +1.09711 | +1.49956 | 1.3668 | 1.3478 | 8/8 | 0.0078 |
| B | 0.05 | +0.29986 | +0.42475 | 1.4165 | 1.4249 | 8/8 | 0.0078 |
| B | 0.2 | +0.82366 | +1.23651 | 1.5012 | 1.4807 | 8/8 | 0.0078 |
| B | 0.5 | +1.36009 | +1.88012 | 1.3823 | 1.3833 | 8/8 | 0.0078 |
