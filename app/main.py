"""Steps 29-31 — FastAPI backend.

    uvicorn app.main:app --port 8000

POST /search        {"query": "...", "k": 10}  -> Stages 2-3 top-k, each with B's validity + summary
GET  /case/{id}     full judgment with role-labelled sentences (for role-coloured display)
GET  /health        which components are loaded, load time
Models load once at startup.
"""
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import b_interface

STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.engine import Engine
    STATE["engine"] = Engine()
    yield
    STATE.clear()


app = FastAPI(title="NyayaSethu — retrieval API (Member A)", version="0.1.0", lifespan=lifespan)

ROLE_COLOURS = {"Facts": "#2563eb", "Ruling by Lower Court": "#7c3aed", "Issue": "#dc2626", "Argument": "#ea580c",
                "Statute": "#0891b2", "Precedent": "#0d9488", "Ratio of the decision": "#16a34a",
                "Ruling by Present Court": "#4b5563"}


class SearchIn(BaseModel):
    query: str
    k: int = 10
    with_b: bool = True


def engine():
    e = STATE.get("engine")
    if e is None:
        raise HTTPException(503, "models not loaded")
    return e


@app.post("/search")
def search(body: SearchIn):
    if not body.query.strip():
        raise HTTPException(400, "empty query")
    t0 = time.time()
    out = engine().search(body.query, min(body.k, 50))
    if body.with_b:
        t1 = time.time()
        for r in out["results"]:
            r["validity"] = b_interface.get_validity(r["case_id"])
            r["summary"] = b_interface.summarize(body.query, r["case_id"])
        out["timings"]["member_b"] = round(time.time() - t1, 3)
    out["timings"]["total"] = round(time.time() - t0, 3)
    out["member_b"] = b_interface.available()
    return out


@app.get("/case/{case_id:path}")
def case(case_id: str):
    d = engine().docs.get(case_id)
    if d is None:
        raise HTTPException(404, "unknown case_id")
    return {"case_id": d.case_id, "title": d.title, "court": d.court, "date": d.date, "bench_size": d.bench_size,
            "source": d.source, "role_colours": ROLE_COLOURS,
            "sentences": [{"idx": s.idx, "text": s.text, "role": s.role, "conf": s.conf,
                           "colour": ROLE_COLOURS.get(s.role or "", "#9ca3af")} for s in d.sentences]}


@app.get("/health")
def health(case_id: Optional[str] = None):
    e = engine()
    return {"judgments": len(e.docs), "components": e.components, "load_seconds": e.load_seconds,
            "member_b": b_interface.available()}
