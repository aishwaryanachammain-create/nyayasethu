"""UI logic without Streamlit (tested in tests/test_ui.py): mode setup, search backends, badges.

Backends, chosen in ui/app.py:
  mock    the contract stand-ins in paths.mock (fixed retrieval list, mock validity, mock judgments)
  engine  Member A's retrieval engine in-process (app/engine.py; needs models/index_corpus)
  api     Member A's FastAPI server (POST /search with Member B's validity + summary per hit)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from common.config import ROOT

FIELD_NAMES = {"issue": "Issue", "material_facts": "Material facts", "ratio": "Ratio", "decision": "Decision",
               "relevance_to_query": "Relevance to your query"}

# validity label -> (badge text, Streamlit badge colour, one-line meaning)
BADGES = {
    "followed": ("Followed", "green", "Later judgments mostly follow or apply it"),
    "distinguished": ("Distinguished", "orange", "Later judgments mostly distinguish it on facts or law"),
    "doubted": ("Doubted", "orange", "An overruling we could not confirm (smaller or unknown bench)"),
    "overruled": ("Overruled", "red", "Overruled by a bench at least as large: not good law"),
    "insufficient_evidence": ("Insufficient evidence", "gray", "Fewer than 3 citing judgments with a treatment"),
}


def badge(v: Optional[Dict]):
    if not v or v.get("status") == "pending" or not v.get("label"):
        return "Validity pending", "gray", (v or {}).get("note", "Stage 4 output not available")
    return BADGES.get(v["label"], (v["label"], "gray", ""))


def use_mock_config() -> Path:
    """Point common.config at a copy of config.yaml with use_mock: true (must run before cfg() is first used)."""
    c = yaml.safe_load((ROOT / "config.yaml").read_text())
    c["use_mock"] = True
    p = Path(tempfile.gettempdir()) / "nyayasethu_config_mock.yaml"
    p.write_text(yaml.safe_dump(c))
    os.environ["NYAYASETHU_CONFIG"] = str(p)
    from common import config
    config.cfg.cache_clear()
    return p


def mock_search(query: str, k: int = 10) -> Dict:
    """Mock retrieval is a fixed list (paths.mock/retrieval.json): it ignores the query."""
    from common.config import path
    from stage5_summary.docs import get_judgment
    runs = json.loads(path("retrieval").read_text(encoding="utf-8"))
    results = []
    for r in (runs[0]["results"] if runs else [])[:k]:
        d = get_judgment(r["case_id"])
        results.append({"case_id": r["case_id"], "score": r.get("score"), "title": d.title if d else "",
                        "court": d.court if d else None, "date": d.date if d else None,
                        "bench_size": d.bench_size if d else None, "matched_segments": r.get("matched_segments")})
    return {"query": query, "results": results, "components": {"retrieval": "mock (fixed list, ignores the query)"}}


def add_member_b(out: Dict, query: str) -> Dict:
    """Attach validity + (fast, API-mode) summary to each hit, the way A's /search does."""
    from app import b_interface
    for r in out["results"]:
        r.setdefault("validity", b_interface.get_validity(r["case_id"]))
        r.setdefault("summary", b_interface.summarize(query, r["case_id"]))
    return out


def engine_search(engine, query: str, k: int) -> Dict:
    return add_member_b(engine.search(query, k), query)


def api_search(url: str, query: str, k: int) -> Dict:
    import httpx
    r = httpx.post(f"{url.rstrip('/')}/search", json={"query": query, "k": k, "with_b": True}, timeout=600)
    r.raise_for_status()
    return r.json()


def evidence_rows(v: Optional[Dict], limit: int = 10) -> List[Dict]:
    """Evidence sentences for the expander, strongest first; the cited span is bolded."""
    ev = sorted((v or {}).get("evidence") or [], key=lambda e: -(e.get("weight") or 0) * (e.get("confidence") or 1))
    return [{**e, "sent_md": (e.get("sent") or "").replace("[[", "**").replace("]]", "**")} for e in ev[:limit]]
