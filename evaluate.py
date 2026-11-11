"""Accuracy, calibration error, and the one number that fixes it."""
import json

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Hunch
from train import batches, device, load, run


@torch.no_grad()
def collect(model, tok, rows):
    """(source, type, logits, right answer) for every question."""
    model.eval()
    items = []
    for chunk in batches(rows, 64):
        logits, y = run(model, tok, chunk)
        K = logits.size(-1)
        logits, y = logits.view(len(chunk), -1, K).cpu(), y.view(len(chunk), -1)
        for b, row in enumerate(chunk):
            for q, question in enumerate(row["questions"]):
                k = len(question["options"])
                items.append((row["source"], question["type"],
                              logits[b, q, :k], y[b, q].item()))
    return items


def ece(conf, correct, bins=10):
    """Average gap between how sure the model was and how often it was right."""
    err = 0.0
    for i in range(bins):
        idx = [j for j, c in enumerate(conf) if i / bins < c <= (i + 1) / bins]
        if idx:
            acc = sum(correct[j] for j in idx) / len(idx)
            avg = sum(conf[j] for j in idx) / len(idx)
            err += len(idx) / len(conf) * abs(acc - avg)
    return err


def report(items, T=1.0):
    for group in sorted({(s, t) for s, t, _, _ in items}):
        sub = [(l, y) for s, t, l, y in items if (s, t) == group]
        probs = [torch.softmax(l / T, -1) for l, _ in sub]
        conf = [p.max().item() for p in probs]
        right = [int(p.argmax().item() == y) for p, (_, y) in zip(probs, sub)]
        print(f"T={T:.2f} {group[0]:5s} {group[1]:6s} n={len(sub):5d} "
              f"acc={sum(right) / len(right):.3f} ECE={ece(conf, right):.3f}")
    oos = [l.argmax().item() == 1 for s, t, l, y in items
           if s == "clinc" and t == "noul" and y == 1]
    print(f"out-of-scope recall {sum(oos) / len(oos):.3f}")


def fit_temperature(items):
    """Find T that makes softmax(logits / T) honest (Guo et al., 2017)."""
    K = max(len(l) for _, _, l, _ in items)
    logits = torch.full((len(items), K), -1e4)
    for i, (_, _, l, _) in enumerate(items):
        logits[i, :len(l)] = l
    y = torch.tensor([y for _, _, _, y in items])
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def main():
    tok = Tokenizer.from_file("tokenizer.json")
    model = Hunch(vocab=tok.get_vocab_size()).to(device)
    model.load_state_dict(torch.load("hunch.pt", map_location=device))
    test = collect(model, tok, load("data/test.jsonl"))
    report(test)                                        # before calibration
    T = fit_temperature(collect(model, tok, load("data/val.jsonl")))
    report(test, T)                                     # fit on val, judged on test
    json.dump({"temperature": T}, open("calibration.json", "w"))


if __name__ == "__main__":
    main()
