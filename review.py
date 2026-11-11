"""Label what the model wasn't sure about. Labels become training data."""
import json
import os
import sqlite3

from hunch import to_options

db = sqlite3.connect("reviews.db")


def label_queue():
    todo = db.execute("SELECT id, state, question, answer FROM reviews"
                      " WHERE label IS NULL").fetchall()
    for rid, state, question, answer in todo:
        q = json.loads(question)
        _, texts = to_options(q)
        print(f"\n{json.loads(state)}\nQ: {q['instructions']}")
        print(f"model: {answer}")
        for i, text in enumerate(texts):
            print(f"  [{i}] {text}")
        pick = input("right answer (enter to skip): ").strip()
        if pick:
            db.execute("UPDATE reviews SET label = ? WHERE id = ?",
                       (int(pick), rid))
            db.commit()


def export():
    os.makedirs("data", exist_ok=True)
    rows = db.execute("SELECT state, question, label FROM reviews"
                      " WHERE label IS NOT NULL").fetchall()
    with open("data/human.jsonl", "w") as f:
        for state, question, label in rows:
            q, s = json.loads(question), json.loads(state)
            _, texts = to_options(q)
            f.write(json.dumps({
                "source": "human",
                "state": s if isinstance(s, str) else json.dumps(s),
                "questions": [{"type": q["type"], "options": texts,
                               "instructions": q["instructions"],
                               "answer": label}]}) + "\n")
    print(len(rows), "human labels -> data/human.jsonl")


if __name__ == "__main__":
    label_queue()
    export()
