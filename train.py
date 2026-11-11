"""Train Hunch from random weights."""
import json
import math
import random

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import Hunch
from tokenizer import collate

device = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else "cpu")
random.seed(0)
torch.manual_seed(0)


def load(path):
    return [json.loads(line) for line in open(path)]


def batches(rows, size, shuffle=False):
    if shuffle:
        random.shuffle(rows)
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def run(model, tok, rows):
    x, y = collate(tok, rows)
    logits = model(**{k: v.to(device) for k, v in x.items()})
    return logits.flatten(0, 1), y.flatten().to(device)


@torch.no_grad()
def val_loss(model, tok, rows):
    model.eval()
    total, n = 0.0, 0
    for chunk in batches(rows, 64):
        logits, y = run(model, tok, chunk)
        total += F.cross_entropy(logits, y, ignore_index=-100,
                                 reduction="sum").item()
        n += (y != -100).sum().item()
    model.train()
    return total / n


def main(epochs=4, bs=32, lr=5e-4, warmup=500):
    tok = Tokenizer.from_file("tokenizer.json")
    train, val = load("data/train.jsonl"), load("data/val.jsonl")
    model = Hunch(vocab=tok.get_vocab_size()).to(device)
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * math.ceil(len(train) / bs)

    def lr_at(step):                          # warm up, then cosine down
        return min((step + 1) / warmup,
                   0.5 * (1 + math.cos(math.pi * step / total)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, best = 0, float("inf")
    for epoch in range(epochs):
        for rows in batches(train, bs, shuffle=True):
            logits, y = run(model, tok, rows)
            loss = F.cross_entropy(logits, y, ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"step {step}/{total}  loss {loss.item():.3f}")
        v = val_loss(model, tok, val)
        print(f"epoch {epoch}  val loss {v:.3f}")
        if v < best:                          # keep the best epoch only
            best = v
            torch.save(model.state_dict(), "hunch.pt")


if __name__ == "__main__":
    main()
