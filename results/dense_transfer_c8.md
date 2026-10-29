# Does the R7 direction TRANSFER? (direction built on one half, measured on the other)

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · layers [13, 14] · budgets [0.05, 0.2, 0.5]
- half A = ['hostility_04', 'risk_aversion_04', 'greed_04', 'avoidance_04']
- half B = ['hostility_05', 'risk_aversion_05', 'greed_05', 'avoidance_05']
- **G2 gate**: r(|g_A|,|g_B|) = 0.5652 (need >= 0.50), top-64 overlap = 0.422 (need >= 0.30) -> PASS
- cos(u_g(A), u_g(B)) = 0.7503   cos(u_M(A), u_M(B)) = 0.8467

## realized mean dJ2, in-sample vs out-of-sample

| budget | pool | sample | n | u_g | u_M | mean ratio M/g | wins | sign p |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.05 | A | in | 4 | +0.29841 | +0.43264 | 1.4536 | 4/4 | 0.1250 |
| 0.05 | A | out | 4 | +0.21188 | +0.32121 | 1.5171 | 4/4 | 0.1250 |
| 0.05 | B | in | 4 | +0.30021 | +0.40896 | 1.3660 | 4/4 | 0.1250 |
| 0.05 | B | out | 4 | +0.26304 | +0.38243 | 1.4622 | 4/4 | 0.1250 |
| 0.05 | ALL | in | 8 | +0.28641 | +0.40427 | 1.4134 | 8/8 | 0.0078 |
| 0.2 | A | in | 4 | +0.63650 | +1.00802 | 1.5829 | 4/4 | 0.1250 |
| 0.2 | A | out | 4 | +0.54238 | +0.86303 | 1.6208 | 4/4 | 0.1250 |
| 0.2 | B | in | 4 | +0.75959 | +1.08206 | 1.4247 | 4/4 | 0.1250 |
| 0.2 | B | out | 4 | +0.75060 | +1.07562 | 1.4329 | 4/4 | 0.1250 |
| 0.2 | ALL | in | 8 | +0.69270 | +1.03031 | 1.4874 | 8/8 | 0.0078 |
| 0.5 | A | in | 4 | +0.89212 | +1.42326 | 1.6008 | 4/4 | 0.1250 |
| 0.5 | A | out | 4 | +0.79899 | +1.22325 | 1.5396 | 4/4 | 0.1250 |
| 0.5 | B | in | 4 | +1.13895 | +1.48184 | 1.3023 | 4/4 | 0.1250 |
| 0.5 | B | out | 4 | +1.16018 | +1.61143 | 1.3937 | 4/4 | 0.1250 |
| 0.5 | ALL | in | 8 | +1.05499 | +1.49958 | 1.4193 | 8/8 | 0.0078 |

**PRIMARY (out-of-sample, pooled over both halves and all budgets)**: u_M > u_g in 24/24, exact two-sided sign p = 0.0000, median abs diff +0.30884
