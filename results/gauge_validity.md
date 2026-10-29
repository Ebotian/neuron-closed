# Stage 0: can the borrowed toxicity gauge see our construct?

- machine `autodl-container-ded147b554-4b263ae1` · trait `hostility` · 25 probe rows (4 Chinese) · gauges {'tox_bert': 'sigmoid0', 'tox_snlp': 'softmax1'}
- P(toxic) per text; AUC is threshold-free and counts ties as 0.5

| block | gauge | mean P(pos) | mean P(neg) | AUC(pos>neg) | #pos>=0.5 | #neg>=0.5 | n |
|---|---|---:|---:|---:|---:|---:|---:|
| `probe_continuations_all` | tox_bert | 0.5362 | 0.0019 | 0.973 | 14 | 0 | 25 |
| `probe_continuations_all` | tox_snlp | 0.6814 | 0.0000 | 0.957 | 17 | 0 | 25 |
| `probe_continuations_zh` | tox_bert | 0.0130 | 0.0077 | 0.688 | 0 | 0 | 4 |
| `probe_continuations_zh` | tox_snlp | 0.0000 | 0.0001 | 0.438 | 0 | 0 | 4 |
| `probe_continuations_en` | tox_bert | 0.6359 | 0.0007 | 0.986 | 14 | 0 | 21 |
| `probe_continuations_en` | tox_snlp | 0.8112 | 0.0000 | 1.000 | 17 | 0 | 21 |
| `slot_words_adj` | tox_bert | 0.7151 | 0.0411 | 0.983 | 14 | 0 | 19 |
| `slot_words_adj` | tox_snlp | 0.5205 | 0.0001 | 0.939 | 10 | 0 | 19 |
| `slot_words_verb` | tox_bert | 0.0613 | 0.0179 | 0.643 | 0 | 0 | 14 |
| `slot_words_verb` | tox_snlp | 0.0001 | 0.0002 | 0.587 | 0 | 0 | 14 |
