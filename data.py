"""Turn three public datasets into typed questions for Hunch."""
import json
import os
import random
import urllib.request

from datasets import load_dataset

rng = random.Random(0)
DOMAINS = ("https://raw.githubusercontent.com/clinc/oos-eval/"
           "master/data/domains.json")
YES_NO = ["yes", "no"]
SUPPORTED = ["Is this a request our assistant can handle?",
             "Does this message match one of our services?"]
TEAM = ["Which team should handle this request?",
        "Which area is this message about?"]
INTENT = ["What does the user want to do?",
          "Which action matches this request?"]
LEVELS = ["very negative", "negative", "neutral",
          "positive", "very positive"]
TONE = ["How positive is this message?",
        "How does the writer feel about it?"]
UNHAPPY = ["Is the writer unhappy?", "Does this message complain?"]


def nice(name):
    return name.replace("_", " ")          # report_lost_card -> report lost card


def ask(kind, instructions, options, answer):
    return {"type": kind, "instructions": instructions,
            "options": options, "answer": answer}


def shuffled(options, answer):
    order = list(range(len(options)))
    rng.shuffle(order)
    return [options[i] for i in order], order.index(answer)


def clinc(split, domains):
    """Routing: is it in scope, which team, which action."""
    ds = load_dataset("clinc/clinc_oos", "plus", split=split)
    names = ds.features["intent"].names
    home = {i: d for d, intents in domains.items() for i in intents}
    for row in ds:
        intent = names[row["intent"]]
        qs = [ask("noul", rng.choice(SUPPORTED), YES_NO,
                  int(intent == "oos"))]
        if intent != "oos":
            team = home[intent]
            others = [d for d in domains if d != team]
            opts = [team] + rng.sample(others, rng.randint(2, 9))
            opts, a = shuffled([nice(d) for d in opts], 0)
            qs.append(ask("choice", rng.choice(TEAM), opts, a))
            peers = [i for i in domains[team] if i != intent]
            opts = [intent] + rng.sample(peers, rng.randint(3, 14))
            opts, a = shuffled([nice(i) for i in opts], 0)
            qs.append(ask("choice", rng.choice(INTENT), opts, a))
        yield {"source": "clinc", "state": row["text"], "questions": qs}


def sst(split):
    """Triage: how negative is the message, is the writer unhappy."""
    for row in load_dataset("SetFit/sst5", split=split):
        y = row["label"]
        qs = [ask("score", rng.choice(TONE), LEVELS, y)]
        if y != 2:                                     # skip neutral
            qs.append(ask("noul", rng.choice(UNHAPPY), YES_NO, int(y > 2)))
        yield {"source": "sst", "state": row["text"], "questions": qs}


def mnli(split, n=None):
    """Is this statement true, given the state?"""
    ds = load_dataset("nyu-mll/glue", "mnli", split=split)
    if n:
        ds = ds.shuffle(seed=0).select(range(n))
    for row in ds:
        no = int(row["label"] != 0)                    # 0 = entailment
        q = ask("noul", row["hypothesis"], YES_NO, no)
        yield {"source": "mnli", "state": row["premise"], "questions": [q]}


def human():
    """Answers people fixed in the review queue (see review.py)."""
    if not os.path.exists("data/human.jsonl"):
        return []
    return [json.loads(line) for line in open("data/human.jsonl")]


def main():
    domains = json.load(urllib.request.urlopen(DOMAINS))
    splits = {
        "train": [clinc("train", domains), sst("train"),
                  mnli("train", 60_000), human()],
        "val": [clinc("validation", domains), sst("validation"),
                mnli("validation_matched")],
        "test": [clinc("test", domains), sst("test"),
                 mnli("validation_mismatched")],
    }
    os.makedirs("data", exist_ok=True)
    for name, parts in splits.items():
        rows = [row for part in parts for row in part]
        rng.shuffle(rows)
        with open(f"data/{name}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(name, len(rows), "states",
              sum(len(r["questions"]) for r in rows), "questions")


if __name__ == "__main__":
    main()
