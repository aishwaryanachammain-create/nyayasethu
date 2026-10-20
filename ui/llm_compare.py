"""NyayaSethu vs a plain LLM, for the demo (data: eval/llm_only/, built by `python -m baseline_llm.run --engine --nli`).

For the five baseline queries the saved replies and their hand-checked resolution are used. Any other query can
be sent to the LLM live (needs the key in .env, see baseline_llm/client.py); its cases are resolved with the
same Stage 0 resolver, without the hand check.
"""
from __future__ import annotations

import csv
import json
import re
import time
from typing import Dict, List, Optional

from common.config import ROOT

OUT = ROOT / "eval" / "llm_only"
# eval/RESULTS_B.md 4.1: of the 12 resolver matches, these were checked by reading the titles.
HAND_CHECK = {("q2", 1): "mismatch", ("q2", 6): "mismatch", ("q2", 8): "mismatch", ("q4", 7): "uncertain"}
VERDICTS = {
    "confirmed": ("Exists", "green", "Found in our citation graph and checked by hand"),
    "resolved": ("Found (automatic match)", "blue", "The resolver matched it to a case in our graph; not hand-checked"),
    "uncertain": ("Uncertain match", "orange", "Matched on party names only"),
    "mismatch": ("Wrong case", "red", "The resolver's match is a different judgment (hand check)"),
    "not_found": ("Not found", "red", "No such case in our 38k-case citation graph"),
}


def _norm(q: str) -> str:
    return " ".join((q or "").lower().split())


def _title_key(t: str) -> str:
    return re.sub(r"[^a-z]", "", re.sub(r"\bm/s\.?", "", (t or "").lower()))


def _rows(cases: List[Dict], qid: Optional[str]) -> List[Dict]:
    seen, out = set(), []
    for c in cases:
        rank = int(c.get("rank") or len(out) + 1)
        if not c.get("case_id"):
            verdict = "not_found"
        elif qid and (qid, rank) in HAND_CHECK:
            verdict = HAND_CHECK[(qid, rank)]
        elif str(c.get("method")) == "name_only":
            verdict = "uncertain"
        else:
            verdict = "confirmed" if qid else "resolved"
        key = _title_key(c.get("llm_title"))
        out.append({"rank": rank, "title": c.get("llm_title"), "citation": c.get("llm_citation"),
                    "year": c.get("llm_year"), "llm_status": c.get("llm_status") or "unknown",
                    "case_id": c.get("case_id") or None, "matched_title": c.get("matched_title") or None,
                    "our_label": c.get("our_label") or None, "verdict": verdict, "repeated": key in seen,
                    "page_one": bool(re.search(r"\b1\s*$", str(c.get("llm_citation") or "")))})
        seen.add(key)
    return out


def saved(query: str, cache: Optional[Dict] = None) -> Optional[Dict]:
    """The saved comparison for a baseline query (or one stored in the demo cache), else None."""
    hit = ((cache or {}).get("llm") or {}).get(_norm(query))
    if hit:
        return hit
    try:
        replies = [json.loads(x) for x in (OUT / "responses.jsonl").read_text(encoding="utf-8").splitlines() if x]
        with open(OUT / "cases.csv", encoding="utf-8") as f:
            cases = list(csv.DictReader(f))
        model = json.loads((OUT / "results.json").read_text(encoding="utf-8")).get("model")
    except (OSError, ValueError):
        return None
    rep = [r for r in replies if _norm(r["query"]) == _norm(query)]
    if not rep:
        return None
    qid = rep[-1]["query_id"].lower()
    rows = _rows([c for c in cases if c["query_id"].lower() == qid], qid)
    return {"query": query, "model": model, "seconds": rep[-1].get("seconds"), "rows": rows, "hand_checked": True}


def live(query: str, resolver, k: int = 10) -> Dict:
    """Ask the LLM now (same prompt as the baseline) and resolve its cases against the graph."""
    import httpx
    from app import b_interface
    from baseline_llm.client import call_with_retry, settings
    from baseline_llm.run import PROMPT, parse_cases, resolve_case
    provider, model = settings()
    t0 = time.time()
    with httpx.Client(timeout=180) as client:
        for _ in range(3):           # a reasoning model can spend its whole budget and reply empty: re-request,
            raw = call_with_retry(PROMPT.format(query=query, k=k), provider, model, client, max_tokens=6000, tries=2)
            if parse_cases(raw):     # as the baseline run did (eval/RESULTS_B.md 4.1)
                break
    secs = round(time.time() - t0, 1)
    cases = []
    for i, c in enumerate(parse_cases(raw), 1):
        cid, method = resolve_case(c, resolver)
        lab = (b_interface.get_validity(cid) or {}).get("label") if cid else None
        from ui.graph import node_info
        cases.append({"rank": i, "llm_title": c.get("title"), "llm_citation": c.get("citation"),
                      "llm_year": c.get("year"), "llm_status": c.get("status"), "case_id": cid, "method": method,
                      "matched_title": node_info(cid)[0] if cid else None, "our_label": lab})
    return {"query": query, "model": model, "seconds": secs, "rows": _rows(cases, None), "hand_checked": False,
            "raw_empty": not cases}


def headline(cmp: Dict) -> Dict:
    rows = cmp.get("rows") or []
    ok = sum(r["verdict"] in ("confirmed", "resolved") for r in rows)
    return {"named": len(rows), "exist": ok, "repeated": sum(r["repeated"] for r in rows),
            "good_law": sum(r["llm_status"] == "good_law" for r in rows),
            "page_one": sum(r["page_one"] for r in rows),
            "conflicts": [r for r in rows if r["our_label"] in ("overruled", "doubted") and r["llm_status"] == "good_law"]}


def totals() -> Dict:
    """Across the five baseline queries: cases named and cases confirmed to exist after the hand check."""
    try:
        with open(OUT / "cases.csv", encoding="utf-8") as f:
            cases = list(csv.DictReader(f))
    except OSError:
        return {}
    rows = [r for q in sorted({c["query_id"].lower() for c in cases})
            for r in _rows([c for c in cases if c["query_id"].lower() == q], q)]
    return {"named": len(rows), "confirmed": sum(r["verdict"] == "confirmed" for r in rows)}
