# G on J2's own output space vs the mismatched probe

- machine `autodl-container-ded147b554-4b263ae1` · model `/root/autodl-tmp/Qwen3.5-4B` · layers [13, 14] · |S| = 32 · m = 192
- readout: J2 = mean over families ['adj', 'verb']; pool variant `full`; lexicon sha256 `af6487149e9ffc75…`
- matched output space: adj = 38 seqs × 5 pos × 2560, verb = 28 seqs × 6 pos × 2560
- mismatched: last prompt position of 1 sequence (from `g_feasibility.json`)

| context | cos(u*_match, u*_mism) | cos(u*_match, u*_I) | cos(u*_mism, u*_I) | r(off-diag G) | Spearman(diag G) | best pair agrees | split-half r | median |rho| | cond |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `hostility_04` | 0.643 | 0.542 | 0.624 | 0.792 | 0.893 | yes | +0.812 | 0.155 | 97.4 |
| `hostility_05` | 0.716 | 0.703 | 0.683 | 0.699 | 0.760 | yes | +0.826 | 0.146 | 52.9 |
| `risk_aversion_04` | 0.718 | 0.640 | 0.641 | 0.692 | 0.896 | yes | +0.753 | 0.117 | 35 |
| `risk_aversion_05` | n/a | 0.682 | n/a | n/a | n/a | n/a | +0.875 | 0.192 | 82.8 |
| `greed_04` | n/a | 0.655 | n/a | n/a | n/a | n/a | +0.769 | 0.149 | 56.4 |
| `greed_05` | n/a | 0.772 | n/a | n/a | n/a | n/a | +0.642 | 0.102 | 27.9 |
| `avoidance_04` | n/a | 0.811 | n/a | n/a | n/a | n/a | +0.708 | 0.118 | 29 |
| `avoidance_05` | n/a | 0.562 | n/a | n/a | n/a | n/a | +0.700 | 0.108 | 31 |

## best pair per context (same real g)

| context | under matched G | under mismatched G | under G=I (two largest abs g) |
|---|---|---|---|
| `hostility_04` | L14N8949, L13N619 | L14N8949, L13N619 | L14N8949, L14N8213 |
| `hostility_05` | L14N8949, L13N1130 | L14N8949, L13N1130 | L14N8949, L13N1130 |
| `risk_aversion_04` | L14N8949, L13N1130 | L14N8949, L13N1130 | L14N8949, L13N7956 |

## joint-pair Q under the matched metric, by selection rule

Q(S) = g_S' G_SS^-1 g_S for a 2-coordinate JOINT control (not the sum of two singles).

| context | Q at matched-G pair | Q at G=I pair | gain of the matched rule |
|---|---:|---:|---:|
| `hostility_04` | 1.466e-08 | 1.08e-08 | +35.8% |
| `hostility_05` | 1.445e-08 | 1.445e-08 | +0.0% |
| `risk_aversion_04` | 1.519e-08 | 1.292e-08 | +17.6% |
