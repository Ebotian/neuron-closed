# BFFM pool specificity at b=0.7 (the only budget where the gauge has range)

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · layers [13, 14] · frame 'You are so {w}.' · budget 0.7 · 12 hostility/en contexts
- baseline P(toxic): mean 0.00057

| pool | P(tox) mean | Δ vs base | n>=0.5 | you-directed | self-directed | rep4 | R (ΔlogP) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `II-` vs `II+` | 0.9875 | +0.9869 | 12/12 | 2/12 | 12/12 | 0.013 | -1.4416 |
| `I-` vs `I+` | 0.0006 | +0.0000 | 0/12 | 0/12 | 0/12 | 0.081 | -1.5681 |
| `III-` vs `III+` | 0.1655 | +0.1649 | 1/12 | 1/12 | 5/12 | 0.040 | -1.4820 |
| `IV-` vs `IV+` | 0.0069 | +0.0064 | 0/12 | 0/12 | 0/12 | 0.000 | -1.6271 |
| `V-` vs `V+` | 0.0340 | +0.0334 | 0/12 | 0/12 | 0/12 | 0.052 | -2.2419 |

## frozen criteria

- **CR1**: II- beats the other four in **4/4** -> **PASS**
  per-pool Δ(II- minus X): I=+0.987, III=+0.822, IV=+0.981, V=+0.954
- **CR2**: you-directed fraction II- 0.17 vs others mean 0.02 -> **PASS**
- **CR3**: |II- minus IV-| = 0.981 > 0.05
- II- vs IV- paired exact sign test p = 0.00048828125 (n=12)
