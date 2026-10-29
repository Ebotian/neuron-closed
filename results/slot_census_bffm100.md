# BFFM-100 slot census

- lexicon `bffm100_factors.json` sha256 `70e8705f1fff09e2…`
- tokenizer `/root/autodl-tmp/Qwen3.5-4B/tokenizer.json` (vocab 248070)
- filter `len(Tokenizer.encode(' ' + w).tokens) == 1`
- **56/100 adjectives are single tokens**; 44 dropped

| pool | source | survivors | halves | usable for G1 |
|---|---:|---:|---:|---|
| `I+` | 10 | 6 | 3 | yes |
| `I-` | 10 | 6 | 3 | yes |
| `II+` | 10 | 7 | 3 | yes |
| `II-` | 10 | 5 | 2 | **no** |
| `III+` | 10 | 9 | 4 | yes |
| `III-` | 10 | 5 | 2 | **no** |
| `IV+` | 6 | 1 | 0 | **no** |
| `IV-` | 14 | 6 | 3 | yes |
| `V+` | 10 | 9 | 4 | yes |
| `V-` | 10 | 2 | 1 | **no** |

usable pools: I+, I-, II+, III+, IV-, V+
thin pools (<6): II-, III-, IV+, V-

## dropped terms (multi-token)

- `agreeable` -> 2 pieces ['Ġagree', 'able']
- `assertive` -> 2 pieces ['Ġassert', 'ive']
- `bashful` -> 2 pieces ['Ġbash', 'ful']
- `conscientious` -> 2 pieces ['Ġconscient', 'ious']
- `considerate` -> 2 pieces ['Ġconsider', 'ate']
- `disorganized` -> 2 pieces ['Ġdis', 'organized']
- `distrustful` -> 2 pieces ['Ġdistrust', 'ful']
- `envious` -> 2 pieces ['Ġen', 'vious']
- `extraverted` -> 3 pieces ['Ġextrav', 'ert', 'ed']
- `fretful` -> 2 pieces ['Ġfret', 'ful']
- `haphazard` -> 3 pieces ['Ġh', 'aph', 'azard']
- `high-strung` -> 3 pieces ['Ġhigh', '-str', 'ung']
- `imperceptive` -> 2 pieces ['Ġimper', 'ceptive']
- `imperturbable` -> 4 pieces ['Ġimp', 'ert', 'urb', 'able']
- `impractical` -> 2 pieces ['Ġimpr', 'actical']
- `introspective` -> 2 pieces ['Ġintros', 'pective']
- `introverted` -> 2 pieces ['Ġintro', 'verted']
- `irritable` -> 2 pieces ['Ġir', 'ritable']
- `moody` -> 2 pieces ['Ġmo', 'ody']
- `self-pitying` -> 4 pieces ['Ġself', '-p', 'ity', 'ing']
- `talkative` -> 2 pieces ['Ġtalk', 'ative']
- `temperamental` -> 2 pieces ['Ġtemper', 'amental']
- `touchy` -> 2 pieces ['Ġtouch', 'y']
- `trustful` -> 2 pieces ['Ġtrust', 'ful']
- `unadventurous` -> 4 pieces ['Ġun', 'ad', 'vent', 'urous']
- `uncharitable` -> 3 pieces ['Ġun', 'char', 'itable']
- `uncooperative` -> 3 pieces ['Ġun', 'co', 'operative']
- `uncreative` -> 2 pieces ['Ġun', 'creative']
- `undemanding` -> 3 pieces ['Ġund', 'emand', 'ing']
- `undependable` -> 3 pieces ['Ġund', 'epend', 'able']
- `unemotional` -> 3 pieces ['Ġun', 'em', 'otional']
- `unenvious` -> 3 pieces ['Ġun', 'en', 'vious']
- `unexcitable` -> 3 pieces ['Ġun', 'exc', 'itable']
- `unimaginative` -> 2 pieces ['Ġunimagin', 'ative']
- `uninquisitive` -> 3 pieces ['Ġunin', 'quis', 'itive']
- `unintellectual` -> 3 pieces ['Ġunint', 'el', 'lectual']
- `unintelligent` -> 2 pieces ['Ġunint', 'elligent']
- `unkind` -> 2 pieces ['Ġun', 'kind']
- `unreflective` -> 3 pieces ['Ġun', 'reflect', 'ive']
- `unrestrained` -> 2 pieces ['Ġunre', 'strained']
- `unsophisticated` -> 4 pieces ['Ġuns', 'oph', 'istic', 'ated']
- `unsympathetic` -> 4 pieces ['Ġuns', 'ymp', 'ath', 'etic']
- `unsystematic` -> 3 pieces ['Ġuns', 'ystem', 'atic']
- `untalkative` -> 3 pieces ['Ġunt', 'alk', 'ative']

## familiarity of the surviving terms (TDA `prop`, CC0)

- median 0.87, min 0.39, max 0.98 (n=100 with a value)
