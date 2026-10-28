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

## best pair per context (same real g)

| context | under matched G | under mismatched G | under G=I (two largest abs g) |
|---|---|---|---|
| `hostility_04` | L14N8949, L13N619 | L14N8949, L13N619 | L14N8949, L14N8213 |
| `hostility_05` | L14N8949, L13N1130 | L14N8949, L13N1130 | L14N8949, L13N1130 |
| `risk_aversion_04` | L14N8949, L13N1130 | L14N8949, L13N1130 | L14N8949, L13N7956 |
