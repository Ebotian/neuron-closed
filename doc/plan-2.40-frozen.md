# Plan 2.40 — 冻结判据

| 项 | 冻结值 |
|---|---|
| φ | 两个借来的毒性判官（`unitary_toxic-bert`、`s-nlp/roberta_toxicity_classifier`）在**生成文本**上的 P(toxic) 均值 |
| C0（仪器） | `0 < mean φ < 0.5` 且 `frac(φ>0.5) < 0.5`，否则 STOP |
| C1 | score-function 梯度跨样本的可复现性（split-half）；不足即"不可估"，不回调 |
| 可靠性 | ICC + F 区间（来自 Gram）；`r(K)=K/(K+2κ)`，样本需求是 K=2κ |
| 记录 | 逐上下文 φ、判官分歧、ICC、区间；判官不一致必须并存，不许平均掉 |
