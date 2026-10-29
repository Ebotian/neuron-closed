# Is the R7 gain degeneration? Judge-free behavioural check

- machine `autodl-container-ded147b554-4b263ae1` / NVIDIA GeForce RTX 4080 SUPER · layers [13, 14] · max_new 96 · contexts 8
- vectorised hook vs frozen `apply_steering`: max |dS| = 0.0e+00 (bitwise equal)
- capability proxy: teacher-forced mean logP of `engine.REFERENCE_TEXT` (sha256 afb86522116b9ccd…)

## absolute levels (mean over contexts)

| budget | arm | stop=eos | rep4 | longest imm. repeat | distinct-2 | dlogP ref |
|---:|---|---:|---:|---:|---:|---:|
| 0.05 | `none` | 2/8 | 0.0067 | 0.0 | 0.9553 | +0.00000 |
| 0.05 | `u_g` | 2/8 | 0.0040 | 0.0 | 0.9661 | -0.00714 |
| 0.05 | `u_M` | 3/8 | 0.0013 | 0.0 | 0.9763 | +0.01125 |
| 0.5 | `none` | 2/8 | 0.0067 | 0.0 | 0.9553 | +0.00000 |
| 0.5 | `u_g` | 4/8 | 0.0115 | 0.0 | 0.9664 | -0.53810 |
| 0.5 | `u_M` | 1/8 | 0.0551 | 0.0 | 0.8868 | -0.64223 |

## frozen paired tests: is u_M worse than u_g? (positive = u_M better)

| budget | metric | u_M better | ties | u_M worse | n non-tied | exact sign p | mean u_g | mean u_M | mean none |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | `rep4` | 1 | 7 | 0 | 1 | 1.0000 | +0.00403 | +0.00134 | +0.00672 |
| 0.05 | `longest_immediate_repeat` | 0 | 8 | 0 | 0 | n/a (all ties) | +0.00000 | +0.00000 | +0.00000 |
| 0.05 | `distinct2` | 3 | 2 | 3 | 6 | 1.0000 | +0.96612 | +0.97632 | +0.95526 |
| 0.05 | `dlogp_ref` | 8 | 0 | 0 | 8 | 0.0078 | -0.00714 | +0.01125 | +0.00000 |
| 0.5 | `rep4` | 1 | 5 | 2 | 3 | 1.0000 | +0.01149 | +0.05511 | +0.00672 |
| 0.5 | `longest_immediate_repeat` | 0 | 8 | 0 | 0 | n/a (all ties) | +0.00000 | +0.00000 | +0.00000 |
| 0.5 | `distinct2` | 3 | 1 | 4 | 7 | 1.0000 | +0.96640 | +0.88684 | +0.95526 |
| 0.5 | `dlogp_ref` | 2 | 0 | 6 | 8 | 0.2891 | -0.53810 | -0.64223 | +0.00000 |

## PRIMARY verdict

- b=0.05: rep4 not-worse in 8/8 (worse in 0, ties 7), dlogP_ref not-worse in 8/8 (worse in 0, ties 0); need >=7/8 -> **PASS**
- b=0.5: rep4 not-worse in 6/8 (worse in 2, ties 5), dlogP_ref not-worse in 2/8 (worse in 6, ties 0); need >=7/8 -> **FAIL**
