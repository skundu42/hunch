# Hunch: a System One decision model from scratch

A ~5.4M-parameter Transformer encoder that reads a state once and answers typed
questions (Choice, Score, Noul) with a probability for every candidate answer.
Built for the Forward Pass video "I Built a Decision Model Like Jev From Scratch".

```bash
pip install -r requirements.txt
python data.py            # data/{train,val,test}.jsonl from CLINC150, SST-5, MultiNLI
python tokenizer.py       # tokenizer.json (WordPiece, 8,192 tokens)
python train.py           # hunch.pt, trained from random weights
python evaluate.py        # accuracy, ECE, temperature -> calibration.json
uvicorn serve:app         # POST /v1/decide on localhost:8000
python review.py          # label the review queue -> data/human.jsonl
python bench.py           # p50 / p95 latency
python -m pytest tests    # failure cases
```

Example request:

```bash
curl localhost:8000/v1/decide -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice for my order. Please refund the duplicate.",
  "questions": {
    "team":   {"type": "choice", "instructions": "Which team should handle this request?",
               "criteria": {"billing": "Payment issues", "technical": "Bugs", "sales": "Pricing"}},
    "refund": {"type": "noul", "instructions": "Does the user want a refund?"}
  }}'
```

Datasets: CLINC150 (clinc/clinc_oos, "plus"), SST-5 (SetFit/sst5), MultiNLI (nyu-mll/glue, mnli).
Not affiliated with TypeSafe; Jev's architecture is not public, so Hunch is our own design.
