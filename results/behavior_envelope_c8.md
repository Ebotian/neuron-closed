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
| 0.1 | `none` | 2/8 | 0.0067 | 0.0 | 0.9553 | +0.00000 |
| 0.1 | `u_g` | 2/8 | 0.0081 | 0.0 | 0.9632 | -0.05274 |
| 0.1 | `u_M` | 2/8 | 0.0054 | 0.0 | 0.9605 | -0.00840 |
| 0.15 | `none` | 2/8 | 0.0067 | 0.0 | 0.9553 | +0.00000 |
| 0.15 | `u_g` | 1/8 | 0.0108 | 0.0 | 0.9658 | -0.12263 |
| 0.15 | `u_M` | 1/8 | 0.0161 | 0.0 | 0.9316 | -0.05599 |
| 0.2 | `none` | 2/8 | 0.0067 | 0.0 | 0.9553 | +0.00000 |
| 0.2 | `u_g` | 3/8 | 0.0030 | 0.0 | 0.9731 | -0.19283 |
| 0.2 | `u_M` | 2/8 | 0.0094 | 0.0 | 0.9539 | -0.12645 |
| 0.3 | `none` | 2/8 | 0.0067 | 0.0 | 0.9553 | +0.00000 |
| 0.3 | `u_g` | 4/8 | 0.0067 | 0.0 | 0.9640 | -0.30763 |
| 0.3 | `u_M` | 3/8 | 0.0081 | 0.0 | 0.9469 | -0.27647 |
| 0.4 | `none` | 2/8 | 0.0067 | 0.0 | 0.9553 | +0.00000 |
| 0.4 | `u_g` | 4/8 | 0.0013 | 0.0 | 0.9721 | -0.42060 |
| 0.4 | `u_M` | 1/8 | 0.0161 | 0.0 | 0.9237 | -0.44203 |
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
| 0.1 | `rep4` | 2 | 5 | 1 | 3 | 1.0000 | +0.00806 | +0.00538 | +0.00672 |
| 0.1 | `longest_immediate_repeat` | 0 | 8 | 0 | 0 | n/a (all ties) | +0.00000 | +0.00000 | +0.00000 |
| 0.1 | `distinct2` | 4 | 2 | 2 | 6 | 0.6875 | +0.96316 | +0.96053 | +0.95526 |
| 0.1 | `dlogp_ref` | 7 | 0 | 1 | 8 | 0.0703 | -0.05274 | -0.00840 | +0.00000 |
| 0.15 | `rep4` | 2 | 3 | 3 | 5 | 1.0000 | +0.01075 | +0.01613 | +0.00672 |
| 0.15 | `longest_immediate_repeat` | 0 | 8 | 0 | 0 | n/a (all ties) | +0.00000 | +0.00000 | +0.00000 |
| 0.15 | `distinct2` | 1 | 1 | 6 | 7 | 0.1250 | +0.96579 | +0.93158 | +0.95526 |
| 0.15 | `dlogp_ref` | 7 | 0 | 1 | 8 | 0.0703 | -0.12263 | -0.05599 | +0.00000 |
| 0.2 | `rep4` | 2 | 3 | 3 | 5 | 1.0000 | +0.00303 | +0.00941 | +0.00672 |
| 0.2 | `longest_immediate_repeat` | 0 | 8 | 0 | 0 | n/a (all ties) | +0.00000 | +0.00000 | +0.00000 |
| 0.2 | `distinct2` | 3 | 1 | 4 | 7 | 1.0000 | +0.97307 | +0.95395 | +0.95526 |
| 0.2 | `dlogp_ref` | 7 | 0 | 1 | 8 | 0.0703 | -0.19283 | -0.12645 | +0.00000 |
| 0.3 | `rep4` | 1 | 5 | 2 | 3 | 1.0000 | +0.00672 | +0.00806 | +0.00672 |
| 0.3 | `longest_immediate_repeat` | 0 | 8 | 0 | 0 | n/a (all ties) | +0.00000 | +0.00000 | +0.00000 |
| 0.3 | `distinct2` | 3 | 2 | 3 | 6 | 1.0000 | +0.96404 | +0.94689 | +0.95526 |
| 0.3 | `dlogp_ref` | 5 | 0 | 3 | 8 | 0.7266 | -0.30763 | -0.27647 | +0.00000 |
| 0.4 | `rep4` | 0 | 5 | 3 | 3 | 0.2500 | +0.00134 | +0.01613 | +0.00672 |
| 0.4 | `longest_immediate_repeat` | 0 | 8 | 0 | 0 | n/a (all ties) | +0.00000 | +0.00000 | +0.00000 |
| 0.4 | `distinct2` | 2 | 1 | 5 | 7 | 0.4531 | +0.97211 | +0.92368 | +0.95526 |
| 0.4 | `dlogp_ref` | 4 | 0 | 4 | 8 | 1.0000 | -0.42060 | -0.44203 | +0.00000 |
| 0.5 | `rep4` | 1 | 5 | 2 | 3 | 1.0000 | +0.01149 | +0.05511 | +0.00672 |
| 0.5 | `longest_immediate_repeat` | 0 | 8 | 0 | 0 | n/a (all ties) | +0.00000 | +0.00000 | +0.00000 |
| 0.5 | `distinct2` | 3 | 1 | 4 | 7 | 1.0000 | +0.96640 | +0.88684 | +0.95526 |
| 0.5 | `dlogp_ref` | 2 | 0 | 6 | 8 | 0.2891 | -0.53810 | -0.64223 | +0.00000 |

## PRIMARY verdict

- b=0.05: rep4 not-worse in 8/8 (worse in 0, ties 7), dlogP_ref not-worse in 8/8 (worse in 0, ties 0); need >=7/8 -> **PASS**
- b=0.1: rep4 not-worse in 7/8 (worse in 1, ties 5), dlogP_ref not-worse in 7/8 (worse in 1, ties 0); need >=7/8 -> **PASS**
- b=0.15: rep4 not-worse in 5/8 (worse in 3, ties 3), dlogP_ref not-worse in 7/8 (worse in 1, ties 0); need >=7/8 -> **FAIL**
- b=0.2: rep4 not-worse in 5/8 (worse in 3, ties 3), dlogP_ref not-worse in 7/8 (worse in 1, ties 0); need >=7/8 -> **FAIL**
- b=0.3: rep4 not-worse in 6/8 (worse in 2, ties 5), dlogP_ref not-worse in 5/8 (worse in 3, ties 0); need >=7/8 -> **FAIL**
- b=0.4: rep4 not-worse in 5/8 (worse in 3, ties 5), dlogP_ref not-worse in 4/8 (worse in 4, ties 0); need >=7/8 -> **FAIL**
- b=0.5: rep4 not-worse in 6/8 (worse in 2, ties 5), dlogP_ref not-worse in 2/8 (worse in 6, ties 0); need >=7/8 -> **FAIL**
