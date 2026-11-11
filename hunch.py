"""Ask Hunch typed questions, get typed answers back."""
import json

import torch
from tokenizers import Tokenizer

from model import Hunch
from tokenizer import collate


def to_options(q):
    """Every question type becomes a list of candidate answers."""
    c = q.get("criteria")
    if q["type"] == "choice":                  # {"billing": "Payment issues"}
        return list(c), [f"{k}: {v}" if v else k for k, v in c.items()]
    if q["type"] == "score":                   # ["calm", "frustrated", ...]
        return list(range(len(c))), list(c)
    c = c or {"true": "yes", "false": "no"}    # noul
    return [True, False], [c["true"], c["false"]]


def confidence(kind, p):
    """TypeSafe's published formulas: 0 = even spread, 1 = certain."""
    n = len(p)
    if n < 2:
        return 1.0
    top = max(range(n), key=p.__getitem__)
    if kind == "choice":                     # top answer above an even guess
        return (p[top] - 1 / n) / (1 - 1 / n)
    away = sum(x * abs(i - top) for i, x in enumerate(p))     # score
    even = sum(abs(i - (n - 1) / 2) for i in range(n)) / n
    return max(0.0, 1 - away / even)


def answer(kind, keys, p):
    if kind == "noul":
        return {"type": "noul", "noul": round(p[0], 3)}
    out = {"type": kind,
           "probabilities": {str(k): round(x, 3) for k, x in zip(keys, p)},
           "confidence": round(confidence(kind, p), 3)}
    if kind == "choice":
        out["choice"] = keys[max(range(len(p)), key=p.__getitem__)]
    else:                                      # score = expected level
        out["score"] = round(sum(i * x for i, x in enumerate(p)), 2)
    return out


class Model:
    def __init__(self, path="hunch.pt", temperature=1.0, device="cpu"):
        self.tok = Tokenizer.from_file("tokenizer.json")
        self.net = Hunch(vocab=self.tok.get_vocab_size()).to(device)
        self.net.load_state_dict(torch.load(path, map_location=device))
        self.net.eval()
        self.T, self.device = temperature, device

    @torch.inference_mode()
    def ask(self, state, questions):
        if not isinstance(state, str):
            state = json.dumps(state)
        names = list(questions)
        keys, qs = [], []
        for name in names:
            q = questions[name]
            k, texts = to_options(q)
            keys.append(k)
            qs.append({"type": q["type"], "options": texts,
                       "instructions": q["instructions"]})
        x, _ = collate(self.tok, [{"state": state, "questions": qs}])
        x = {k: v.to(self.device) for k, v in x.items()}
        logits = self.net(**x)[0]                  # one state, all questions
        out = {}
        for i, name in enumerate(names):
            z = logits[i, :len(keys[i])] / self.T
            p = torch.softmax(z, -1).tolist()
            out[name] = answer(questions[name]["type"], keys[i], p)
        return out
