# The closed form over the whole model (32 layers, 294,912 coordinates)

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · model `/root/autodl-tmp/Qwen3.5-4B` · **no readout measured**
- contexts 8: hostility_04, hostility_05, risk_aversion_04, risk_aversion_05, greed_04, greed_05, avoidance_04, avoidance_05
- selection: greedy max of `Q(S)=g_SᵀM_SS⁻¹g_S`, M block-diagonal by layer (exact budget metric); M_ii = 1 ⇒ Q is additive across layers
- doses normalised to `b=0.05` on the reported support

## layer histogram of the top-K positions

| K | layers used | top layers (count) | median Q share |
|---:|---:|---|---:|
| 64 | 24 | L31:11, L25:6, L30:5, L5:4, L23:4, L13:3 | 0.0130 |
| 256 | 32 | L31:23, L26:15, L22:14, L27:14, L30:14, L5:12 | 0.0030 |
| 1024 | 32 | L31:77, L5:47, L25:40, L26:40, L28:37, L30:37 | 0.0007 |

## stability of the per-context lists

| K | median pairwise Jaccard | min | max |
|---:|---:|---:|---:|
| 64 | 0.143 | 0.016 | 0.333 |
| 256 | 0.098 | 0.018 | 0.193 |
| 1024 | 0.098 | 0.018 | 0.193 |

## the community library (public API snapshot, outside git)

- 165 unique coordinates / 251 neuron-dose rows
- their layer histogram (top 6): L7:17, L22:16, L21:14, L6:11, L23:11, L17:10

| K | community coordinates found | random null | dose ratio (ours/theirs) |
|---:|---:|---:|---|
| 64 | **0** | 0.04 | — |
| 256 | **1** | 0.14 | 0.0076 |
| 1024 | **4** | 0.57 | 0.0058, 0.0023, -0.0039, 0.0023 |

| K | layer | neuron | our rank | our Q share | our dose | their dose | ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 16 | 8927 | 231 | 0.0020 | +0.00229 | +0.30 | 0.0076 |
| 1024 | 16 | 8927 | 231 | 0.0010 | +0.00174 | +0.30 | 0.0058 |
| 1024 | 14 | 8949 | 290 | 0.0009 | +0.00159 | +0.70 | 0.0023 |
| 1024 | 20 | 1950 | 384 | 0.0008 | -0.00156 | +0.40 | -0.0039 |
| 1024 | 17 | 6015 | 412 | 0.0008 | -0.00140 | -0.60 | 0.0023 |
