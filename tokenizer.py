"""A WordPiece tokenizer, plus the packer that lays out one request."""
import json

import torch
from tokenizers import (Tokenizer, models, normalizers,
                        pre_tokenizers, trainers)

SPECIAL = ["[PAD]", "[UNK]", "[STATE]", "[Q]", "[OPT]",
           "[CHOICE]", "[SCORE]", "[NOUL]"]
PAD, UNK, STATE, Q, OPT = range(5)
TYPE_ID = {"choice": 5, "score": 6, "noul": 7}
FIELDS = ("ids", "pos", "role", "qid", "oid")


def texts(path):
    for line in open(path):
        row = json.loads(line)
        yield row["state"]
        for q in row["questions"]:
            yield q["instructions"]
            yield from q["options"]


def train(path="data/train.jsonl", size=8192):
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=size,
                                        special_tokens=SPECIAL)
    tok.train_from_iterator(texts(path), trainer)
    tok.save("tokenizer.json")
    return tok


def pack(tok, state, questions, max_state=384):
    """One state and all its questions, laid out as one sequence."""
    seq = {k: [] for k in FIELDS}

    def put(tokens, start, role, q, o):
        for i, t in enumerate(tokens):
            for k, v in zip(FIELDS, (t, start + i, role, q, o)):
                seq[k].append(v)
        return start + len(tokens)

    s_end = put([STATE] + tok.encode(state).ids[:max_state], 0, 0, -1, -1)
    marks = []
    for q, question in enumerate(questions):
        head = [Q, TYPE_ID[question["type"]]]
        head += tok.encode(question["instructions"]).ids[:64]
        o_start = put(head, s_end, 1, q, -1)        # every question starts at s_end
        row = []
        for o, text in enumerate(question["options"]):
            row.append(len(seq["ids"]))             # where this [OPT] sits
            put([OPT] + tok.encode(text).ids[:32], o_start, 2, q, o)
        marks.append(row)                           # every option starts at o_start
    return seq, marks


def collate(tok, rows):
    """Pad a list of requests into tensors the model can read."""
    packed = [pack(tok, r["state"], r["questions"]) for r in rows]
    B = len(rows)
    L = max(len(seq["ids"]) for seq, _ in packed)
    Qn = max(len(marks) for _, marks in packed)
    K = max(len(row) for _, marks in packed for row in marks)
    x = {k: torch.full((B, L), -1 if k in ("qid", "oid") else 0)
         for k in FIELDS}
    x["pad"] = torch.ones(B, L, dtype=torch.bool)
    x["opt_at"] = torch.full((B, Qn, K), -1)
    answer = torch.full((B, Qn), -100)
    for b, ((seq, marks), row) in enumerate(zip(packed, rows)):
        n = len(seq["ids"])
        for k in FIELDS:
            x[k][b, :n] = torch.tensor(seq[k])
        x["pad"][b, :n] = False
        for q, opts in enumerate(marks):
            x["opt_at"][b, q, :len(opts)] = torch.tensor(opts)
            answer[b, q] = row["questions"][q].get("answer", -100)
    return x, answer


if __name__ == "__main__":
    tok = train()
    print(tok.get_vocab_size(), "tokens")
    print(tok.encode("My card was charged twice.").tokens)
