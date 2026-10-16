"""Step 26 — authority prior. Member B's pagerank.csv when it exists, otherwise the raw citation
in-degree from Stage 0 (data/processed/indegree.csv). Returned on a log scale, min-max normalised
to [0, 1] so it can be mixed with the cross-encoder score."""
from __future__ import annotations

import csv
import math
from typing import Dict, Tuple

from common.config import path


def load_prior() -> Tuple[Dict[str, float], str]:
    pr, src = {}, None
    p = path("pagerank")
    if p.exists():
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cid = r.get("case_id") or r.get("doc_id")
                pr[cid] = float(r["pagerank"])
        src = "pagerank (Member B)"
    else:
        q = path("processed") / "indegree.csv"
        if q.exists():
            with open(q, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    pr[r["case_id"]] = float(r["in_degree"]) + 1.0       # +1: log(1) = 0 for uncited cases
            src = "citation in-degree (fallback until pagerank.csv arrives)"
    if not pr:
        return {}, "none"
    logs = {k: math.log(max(v, 1e-12)) for k, v in pr.items()}
    lo, hi = min(logs.values()), max(logs.values())
    return {k: (v - lo) / (hi - lo) if hi > lo else 0.0 for k, v in logs.items()}, src
