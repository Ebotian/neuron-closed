# Lower-confidence-bound ranking vs mean |g| ranking

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · layers [13, 14] · budget 0.05 · 8 contexts · bootstrap 2000
- Spearman(LCB, |mean g|) = **0.6724**, top-64 overlap **0.797** (so the two rankings really differ)
- full-support dJ2 (mean ranking) = +0.28641

| K | mean-rank dJ2 | lcb-rank dJ2 | mean-rank capture | lcb-rank capture |
|---:|---:|---:|---:|---:|
| 1 | +0.05059 | +0.03521 | 17.7% | 12.3% |
| 2 | +0.06743 | +0.05998 | 23.5% | 20.9% |
| 4 | +0.08725 | +0.08725 | 30.5% | 30.5% |
| 8 | +0.11554 | +0.11191 | 40.3% | 39.1% |
| 16 | +0.13293 | +0.13293 | 46.4% | 46.4% |
| 32 | +0.15199 | +0.14869 | 53.1% | 51.9% |
| 64 | +0.17295 | +0.17143 | 60.4% | 59.9% |
| 128 | +0.19311 | +0.19344 | 67.4% | 67.5% |
| 256 | +0.21931 | +0.21954 | 76.6% | 76.7% |
| 512 | +0.24104 | +0.24145 | 84.2% | 84.3% |
| 1024 | +0.25919 | +0.26122 | 90.5% | 91.2% |
| 2048 | +0.27397 | +0.27285 | 95.7% | 95.3% |
| 4096 | +0.28160 | +0.28048 | 98.3% | 97.9% |
| 8192 | +0.28523 | +0.28364 | 99.6% | 99.0% |
| 18432 | +0.28641 | +0.28641 | 100.0% | 100.0% |

## K needed to reach a share of the full-support effect (same denominator)

| share | mean ranking | LCB ranking |
|---|---:|---:|
| 73% | 256 | 256 |
| 89% | 1024 | 1024 |
| 97% | 4096 | 4096 |

**registered prediction**: K@73%(lcb) < 256 = K@73%(mean) -> FALSIFIED
