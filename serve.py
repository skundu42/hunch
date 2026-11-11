"""A local decision API that sends unsure answers to a person.

Run:  uvicorn serve:app --port 8000
"""
import json
import sqlite3

from fastapi import FastAPI
from pydantic import BaseModel

from hunch import Model

T = json.load(open("calibration.json"))["temperature"]
model = Model("hunch.pt", temperature=T)
db = sqlite3.connect("reviews.db", check_same_thread=False)
db.execute("CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY,"
           " state TEXT, question TEXT, answer TEXT, label INTEGER)")
app = FastAPI()


class Request(BaseModel):
    state: str | dict | list
    questions: dict


def needs_review(a):
    if a["type"] == "noul":
        return 0.3 < a["noul"] < 0.7        # not a clear yes or a clear no
    return a["confidence"] < 0.5            # probability spread out


@app.post("/v1/decide")
def decide(req: Request):
    answers = model.ask(req.state, req.questions)
    unsure = [k for k, a in answers.items() if needs_review(a)]
    for k in unsure:
        db.execute("INSERT INTO reviews (state, question, answer)"
                   " VALUES (?, ?, ?)",
                   (json.dumps(req.state), json.dumps(req.questions[k]),
                    json.dumps(answers[k])))
    db.commit()
    return {"model": "hunch-0.1", "answers": answers, "needs_review": unsure}
