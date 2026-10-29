# persona-matrix gate report

- started: 2026-10-26 16:14:17
- finished: 2026-10-26 16:15:39
- model: /root/autodl-tmp/Qwen3.5-4B
- gpu: NVIDIA GeForce RTX 4080 SUPER / torch 2.8.0+cu128
- dose: 1.0  max_new_tokens: 128

## Verdict

| gate | result |
|---|---|
| gate0_p50 | PASS |
| gate2_behaviour | PASS |
| gate2b_randdir | PASS |

## Gate 0 - p50 identity

| layer | got | expected | abs diff | pass |
|---|---|---|---|---|
| L13 | 9.752792358398438 | 9.752792358398438 | 0 | PASS |
| L14 | 10.32901382446289 | 10.32901382446289 | 0 | PASS |

## Gate 2 - evil_A+evil_B @ dose 1.0 vs recorded target_pair

pair terms: `[(14, 8949, 1.0), (13, 7956, -1.0)]`

exact 8/8, whitespace-normalised 8/8, baseline exact 8/8

| prompt | cond | tokens (got/exp) | match | similarity |
|---|---|---|---|---|
| p01 | baseline | 42/42 | EXACT | 1.0 |
| p01 | target_pair | 32/32 | EXACT | 1.0 |
| p02 | baseline | 128/128 | EXACT | 1.0 |
| p02 | target_pair | 128/128 | EXACT | 1.0 |
| p03 | baseline | 128/128 | EXACT | 1.0 |
| p03 | target_pair | 128/128 | EXACT | 1.0 |
| p04 | baseline | 100/100 | EXACT | 1.0 |
| p04 | target_pair | 128/128 | EXACT | 1.0 |
| p05 | baseline | 128/128 | EXACT | 1.0 |
| p05 | target_pair | 128/128 | EXACT | 1.0 |
| p06 | baseline | 74/74 | EXACT | 1.0 |
| p06 | target_pair | 63/63 | EXACT | 1.0 |
| p07 | baseline | 128/128 | EXACT | 1.0 |
| p07 | target_pair | 128/128 | EXACT | 1.0 |
| p08 | baseline | 128/128 | EXACT | 1.0 |
| p08 | target_pair | 128/128 | EXACT | 1.0 |

## Gate 2b - random-direction control fires

- hook fired: True
- output changed: True
- hook_calls: `{'14:randdir': 128}`
- randdir: `[{'layer': 14, 'seed': 0, 'dose': 1.0, 'residual_norm': 10.329014}]`

## What to do next

All gates passed. The package shares a dose scale with qwen-evil-repro and reproduces the published pair behaviour verbatim; the random-direction hook works. Safe to spend the long runs on this package.
