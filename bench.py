"""How long does one decision take? Median and 95th-percentile latency."""
import time

import torch

from hunch import Model

STATE = ("Our API integration started returning 500 errors on every "
         "request about 20 minutes ago, and we can't process any "
         "customer orders until this is fixed.")
QUESTIONS = {
    "team": {"type": "choice",
             "instructions": "Which team should handle this request?",
             "criteria": {"billing": "", "technical": "", "sales": ""}},
    "tone": {"type": "score", "instructions": "How positive is this message?",
             "criteria": ["very negative", "negative", "neutral",
                          "positive", "very positive"]},
    "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    "refund": {"type": "noul", "instructions": "Does the user want a refund?"},
}


def timed(fn, runs=200, warmup=20):
    for _ in range(warmup):                  # first calls pay one-time costs
        fn()
    ms = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()                                 # .tolist() inside waits for the GPU
        ms.append((time.perf_counter() - t0) * 1000)
    ms.sort()
    return ms[len(ms) // 2], ms[int(len(ms) * 0.95)]


def main(device="cpu"):
    torch.set_num_threads(4)
    m = Model("hunch.pt", device=device)
    one = {"urgent": QUESTIONS["urgent"]}
    cases = {
        "1 question": lambda: m.ask(STATE, one),
        "4 packed": lambda: m.ask(STATE, QUESTIONS),
        "4 separate": lambda: [m.ask(STATE, {k: q})
                               for k, q in QUESTIONS.items()],
    }
    for name, fn in cases.items():
        p50, p95 = timed(fn)
        print(f"{name:11s} p50 {p50:6.1f} ms   p95 {p95:6.1f} ms")


if __name__ == "__main__":
    main()
