"""The judge-based behaviour functional - the ONE implementation of the plan-2.40 phi.

DIFFERENT CONSTRUCT FROM estimators/phi.py, kept separate on purpose. `phi.py` reads the hostility-vs-praise slot
log-mass along the model's own continuation (nats, real dynamic range). This one reads two borrowed toxicity
classifiers' P(toxic) on the generated text: it is the construct plan 2.40 actually measured, it saturates near 0 on
polite text (mean 0.04 +- 0.01), and that saturation is why its gradient turned out to be unmeasurable. Recording the
two side by side is the point - the difference between them is not a bug in either.

`s-nlp` returns exactly 0.0000 on polite replies, so phi is dominated by `unitary`; both are returned per text so the
disagreement stays visible instead of being averaged away.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


def load_gauges(root, dev):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    out = {}
    for sub, name in [('unitary_toxic-bert', 'unitary'), ('s-nlp_roberta_toxicity_classifier', 'snlp')]:
        d = Path(root) / sub
        if not d.exists():
            continue
        tk = AutoTokenizer.from_pretrained(d, local_files_only=True)
        m = AutoModelForSequenceClassification.from_pretrained(d, local_files_only=True).to(dev).eval()
        out[name] = (tk, m)
    return out


def p_toxic(gauges, texts, dev, batch=8):
    res = {}
    for name, (tk, m) in gauges.items():
        vals = []
        for i in range(0, len(texts), batch):
            enc = tk(texts[i:i + batch], return_tensors='pt', padding=True, truncation=True,
                     max_length=256).to(dev)
            with torch.inference_mode():
                lg = m(**enc).logits.float()
            p = torch.sigmoid(lg[:, 0]) if lg.shape[-1] == 1 else F.softmax(lg, -1)[:, 1]
            vals.extend(float(x) for x in p)
        res[name] = vals
    return res


def phi_judge(gauges, texts, dev):
    """Mean of the gauges' P(toxic) per text - the plan-2.40 construct, one line of aggregation."""
    ph = p_toxic(gauges, texts, dev)
    n = len(texts)
    return [sum(d[i] for d in ph.values()) / len(ph) for i in range(n)], ph
