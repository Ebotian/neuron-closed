# Semantic compilers under an independent gauge (compare, do not choose)

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · layers [13, 14] · budget 0.05 · 8 contexts · greedy <= 96 tokens
- compilers: ['caa_jibes', 'caa_bffm', 'j2']
- caa_bffm words: antagonism 5, warmth 7, frame 'You are so {w}.'

## Pareto table (all deltas are steer minus base, mean over contexts)

| compiler | E_tox_bert | E_tox_snlp | E_own (cross-check) | R (dlogP ref) |
|---|---:|---:|---:|---:|
| `caa_jibes` | +0.0000 | -0.0000 | +0.125 | -0.06096 |
| `caa_bffm` | +0.0002 | -0.0000 | -0.125 | -0.00731 |
| `j2` | +0.0000 | -0.0000 | +0.125 | +0.03038 |
| `known_recipe` | +0.0072 | +0.0001 | +1.000 | -0.66360 |

## direction agreement (does the compiler change the vector?)

| pair | cos | top-64 overlap |
|---|---:|---:|
| `caa_jibes|caa_bffm` | 0.1767 | 0.094 |
| `caa_jibes|j2` | -0.0024 | 0.047 |
| `caa_bffm|j2` | 0.4902 | 0.281 |

## gauge agreement on the same texts

- Pearson(tox_bert, tox_snlp) = 0.9955415959164539
- kappa(tox_bert, tox_snlp) at 0.5 = nan (agreement 1.000, n=64)
- Pearson(tox_bert, own) = 0.3317677745983257
- Pearson(tox_snlp, own) = 0.3746995502984139

## registered predictions

- P1: **HELD**  (cos(caa_jibes, j2) < 0.8)
- P2: **HELD**  (the two borrowed gauges agree better with each other than with own)
- P3: **HELD**  (E_tox_bert(j2) > E_tox_bert(caa_jibes))
