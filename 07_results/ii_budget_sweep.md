# Budget sweep of the specific hostility direction (plan 2.30.4 items 1-3)

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · layers [13, 14] · frame 'You are so {w}.' · 12 hostility/en contexts
- II- words: ['cold', 'demanding', 'harsh', 'rude', 'selfish']
- II+ words: ['cooperative', 'generous', 'helpful', 'kind', 'pleasant', 'sympathetic', 'warm']
- baseline P(toxic) mean 0.00057

| arm | P(tox) mean | n>=0.5 | dlogP ref | rep4 | you-dir | self-dir |
|---|---:|---:|---:|---:|---:|---:|
| `II-@b=0.05` | 0.0006 | 0/12 | -0.0275 | 0.005 | 0/12 | 0/12 |
| `II-@b=0.2` | 0.1802 | 2/12 | -0.3746 | 0.000 | 1/12 | 1/12 |
| `II-@b=0.3` | 0.5818 | 7/12 | -0.5867 | 0.000 | 4/12 | 3/12 |
| `II-@b=0.4` | 0.6075 | 8/12 | -0.7454 | 0.001 | 4/12 | 4/12 |
| `II-@b=0.5` | 0.7602 | 9/12 | -0.9269 | 0.000 | 3/12 | 7/12 |
| `II-@b=0.7` | 0.9875 | 12/12 | -1.4416 | 0.013 | 2/12 | 12/12 |
| `known@0.7` | 0.0806 | 1/12 | -0.6215 | 0.003 | 0/12 | 0/12 |

## the frozen rule

a usable point exists iff P(toxic) > 0.5 and |dlogP| < 0.3 nat

- points that satisfy it: **NONE**

## how far is the famous coordinate from this direction?

- cos(u_II-, u_known) = **+0.0299**
- u_known has 2 nonzero coordinates; the II- top-64 contains 1 of them
- both arms measured on the same contexts; compare the `known@0.7` row with `II-@b=0.7`
