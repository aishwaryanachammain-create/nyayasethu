"""Stage 4 (Member B): doctrinal validity of a precedent from how later judgments treat it.

    from stage4_validity import get_validity
    get_validity("sc:ESCR010000771985")   # -> {case_id, label, confidence, n_citing, evidence, ...}

Reads paths.validity (outputs/validity.json, or the mock with use_mock: true).
"""
from __future__ import annotations

import functools
import json
from typing import Dict


@functools.lru_cache(maxsize=1)
def _table() -> Dict[str, dict]:
    from common.config import path
    p = path("validity")
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return {v["case_id"]: v for v in json.load(f)}


def get_validity(case_id: str) -> Dict:
    t = _table()
    if not t:
        return {"status": "pending", "case_id": case_id, "label": None,
                "note": "validity.json not built (python -m stage4_validity.aggregate)"}
    v = t.get(case_id)
    if v is not None:
        return {"status": "ok", **v}
    return {"status": "ok", "case_id": case_id, "label": "insufficient_evidence", "confidence": None,
            "n_citing": 0, "evidence": [], "note": "no citing judgment in the citation graph"}
