# Semantic compilers under an independent gauge: hostility prompts, English only

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · layers [13, 14] · budgets [0.05, 0.7] · 12 contexts (trait hostility, lang en) · greedy <= 96 -> 64 tokens
- compilers: ['caa_jibes', 'caa_bffm', 'j2']; positive control: community recipe L14N8949+0.7 / L13N7956-0.7
- caa_bffm words: 5 antagonism vs 7 warmth, frame 'You are so {w}.'
- baseline P(toxic): max 0.00066, mean 0.00057

| arm | budget | E_tox_bert | E_tox_snlp | E_own | R (dlogP ref) | n_pos/12 | max P(toxic) |
|---|---|---:|---:|---:|---:|---:|---:|
| `caa_jibes` | 0.05 | +0.0000 | -0.0000 | +0.08 | -0.0964 | 5/12 | 0.0007 |
| `caa_jibes` | 0.7 | +0.6280 | +0.6818 | +1.50 | -2.3022 | 12/12 | 0.9858 |
| `caa_bffm` | 0.05 | +0.0001 | +0.0000 | +0.17 | -0.0275 | 8/12 | 0.0012 |
| `caa_bffm` | 0.7 | +0.9869 | +0.9912 | +0.92 | -1.4416 | 12/12 | 0.9936 |
| `j2` | 0.05 | +0.0003 | +0.0000 | +0.00 | +0.0035 | 8/12 | 0.0031 |
| `j2` | 0.7 | +0.1022 | +0.3384 | +1.17 | -1.2547 | 12/12 | 0.5444 |
| `known_recipe` | 0.05 | +0.0801 | +0.0860 | +0.00 | -0.6215 | 10/12 | 0.5553 |
| `known_recipe` | 0.7 | +0.0801 | +0.0860 | +0.00 | -0.6215 | 10/12 | 0.5553 |

## direction agreement between compilers (unit M-norm, budget independent)

| pair | cos | top-64 overlap |
|---|---:|---:|
| `caa_jibes|caa_bffm` | 0.1791 | 0.031 |
| `caa_jibes|j2` | 0.1536 | 0.031 |
| `caa_bffm|j2` | 0.5812 | 0.281 |

## gauge agreement over all texts

- Pearson(tox_bert, tox_snlp) = 0.8888773653891914
- Pearson(tox_bert, E_own) = 0.5153346998860503
- Pearson(tox_snlp, E_own) = 0.6740134106038529

## registered predictions

- P1: **HELD**
- P2: **HELD**
- P3: **HELD**

_rendered from compiler_pareto_hostility12.json (the run crashed in the markdown step after writing the JSON; the renderer has since been fixed)._
