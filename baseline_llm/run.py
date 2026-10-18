"""LLM-only baseline: ask an LLM for precedents, summaries and validity for a query, then check the answer
against our data.

    python -m baseline_llm.run                       # the 5 queries in app/latency_test.py
    python -m baseline_llm.run --queries q.json      # {"<query_id>": "<query text>", ...}
    python -m baseline_llm.run --engine --nli        # + overlap with our top-k, + NLI check of LLM summaries
    python -m baseline_llm.run --score-only          # re-score saved responses, no API calls

Checks per case the LLM names:
  exists      resolved to a Supreme Court case in our citation graph (AWS metadata 1950-2025 + corpus), using
              the Stage 0 resolver (reporter key, else party names near the given year). "Not found" is NOT
              proof of hallucination: High Court cases and SCC/AIR-only citations of unknown cases cannot resolve.
  full_text   the judgment text is in paths.judgments (needed for the NLI check)
  validity    LLM status vs our validity.json label, only where ours is decided (not insufficient_evidence)
  in_our_topk with --engine: the case is also in Member A's top-k for that query
  nli         with --nli: share of LLM summary sentences entailed by the judgment (Stage 5 filter, same threshold)

Outputs: eval/llm_only/responses.jsonl (raw, resumable), eval/llm_only/cases.csv, eval/llm_only/results.json.
The API key comes from .env (baseline_llm/client.py). Without it nothing runs and the baseline stays PENDING.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from typing import Dict, List, Optional

from common.config import ROOT, path

OUT = ROOT / "eval" / "llm_only"
FIELDS = ["issue", "material_facts", "ratio", "decision", "relevance_to_query"]
LLM_STATUS = {"good_law": "followed", "followed": "followed", "distinguished": "distinguished",
              "doubted": "doubted", "overruled": "overruled", "unknown": None}
DECIDED = {"followed", "distinguished", "doubted", "overruled"}

PROMPT = """You are an Indian legal research assistant. A user describes their problem:

"{query}"

List up to {k} Indian Supreme Court judgments that are the most relevant precedents, most relevant first.
For each, give:
- title: case name as "Petitioner v. Respondent"
- citation: one reporter citation, e.g. "(1964) 1 SCR 515" or "(2003) 6 SCC 675"
- year: year of decision
- summary: issue, material_facts, ratio, decision, relevance_to_query (one or two sentences each)
- status: how later courts have treated it: "good_law" (followed), "distinguished", "doubted", "overruled", or "unknown"

Answer with a JSON array only, no other text:
[{{"title": "...", "citation": "...", "year": 0000, "summary": {{"issue": "...", "material_facts": "...", "ratio": "...", "decision": "...", "relevance_to_query": "..."}}, "status": "..."}}]"""


def default_queries() -> Dict[str, str]:
    from app.latency_test import QUERIES
    return {f"q{i + 1}": q for i, q in enumerate(QUERIES)}


def parse_cases(text: str) -> List[dict]:
    """The first JSON array in the reply; [] if there is none (the reply is kept raw either way)."""
    i, j = (text or "").find("["), (text or "").rfind("]")
    if i < 0 or j <= i:
        return []
    try:
        out = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return []
    return [c for c in out if isinstance(c, dict) and c.get("title")]


# ---------------------------------------------------------------- resolution
def load_resolver():
    from stage0_graph.build import iter_docs
    from stage0_graph.nodes import merge_corpus, metadata_nodes
    from stage0_graph.resolve import Resolver
    nodes = merge_corpus(metadata_nodes(path("raw") / "sc-judgments" / "metadata"), iter_docs(path("judgments")))
    return Resolver(nodes)


def _year(v) -> Optional[int]:
    m = re.search(r"\b(19|20)\d{2}\b", str(v or ""))
    return int(m.group(0)) if m else None


def resolve_case(case: dict, res) -> tuple:
    """-> (case_id or None, method). Reporter citation first, then the party names near the stated year."""
    from stage0_graph.extract import Mention, case_names, mentions
    title, cite = str(case.get("title") or ""), str(case.get("citation") or "")
    year = _year(case.get("year")) or _year(cite)
    for m in mentions(f"{title}, {cite}"):
        if m.key:
            cid, method, _ = res.resolve(m, None)
            if cid:
                return cid, method
    names = case_names(title)
    name = names[0].name if names else re.sub(r"\s+(?:v\.?|vs\.?|versus)\s+", " v ", title)
    if year:
        cid, _ = res._match_name(name, year - 1, year + 1, res.t_year)
        if cid:
            return cid, "name_year"
    cid, method, _ = res.resolve(Mention("NAME", "", (0, 0), title, None, name), None)
    return cid, method


# ---------------------------------------------------------------- scoring
def score(rows: List[dict]) -> dict:
    n = len(rows)
    ex = [r for r in rows if r["case_id"]]
    st = {"cases_named": n, "queries": len({r["query_id"] for r in rows}),
          "exists_in_graph": len(ex), "exists_rate": round(len(ex) / n, 4) if n else None,
          "full_text": sum(bool(r["full_text"]) for r in rows),
          "resolve_method": dict(Counter(r["method"] for r in ex)),
          "llm_status": dict(Counter(r["llm_status"] or "unparsed" for r in rows)),
          "our_label_for_existing": dict(Counter(r["our_label"] for r in ex))}
    both = [r for r in ex if r["our_label"] in DECIDED and LLM_STATUS.get(r["llm_status"])]
    st["validity_compared"] = len(both)
    st["validity_agreement"] = (round(sum(LLM_STATUS[r["llm_status"]] == r["our_label"] for r in both) / len(both), 4)
                                if both else None)
    st["validity_pairs"] = dict(Counter(f"llm={LLM_STATUS[r['llm_status']]}|ours={r['our_label']}" for r in both))
    tk = [r for r in rows if r.get("in_our_topk") not in (None, "")]
    if tk:
        st["in_our_topk"] = sum(bool(r["in_our_topk"]) for r in tk)
    nli = [r for r in rows if r.get("nli_sentences")]
    if nli:
        tot = sum(r["nli_sentences"] for r in nli)
        st["nli_cases"] = len(nli)
        st["nli_sentences"] = tot
        st["nli_entailed_rate"] = round(sum(r["nli_kept"] for r in nli) / tot, 4)
    return st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--engine", action="store_true", help="also run Member A's engine and compare top-k")
    p.add_argument("--nli", action="store_true", help="NLI-check LLM summaries of cases we have the text of")
    p.add_argument("--score-only", action="store_true")
    p.add_argument("--out", default=str(OUT))
    a = p.parse_args()
    from pathlib import Path
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    queries = json.load(open(a.queries)) if a.queries else default_queries()
    resp_file = out / "responses.jsonl"
    done = {}
    if resp_file.exists():
        for line in open(resp_file, encoding="utf-8"):
            r = json.loads(line)
            done[r["query_id"]] = r

    todo = [q for q in queries if q not in done]
    if todo and not a.score_only:
        import httpx
        from .client import call_with_retry, settings
        provider, model = settings()
        with open(resp_file, "a", encoding="utf-8") as f, httpx.Client(timeout=180) as client:
            for qid in todo:
                t0 = time.time()
                raw = call_with_retry(PROMPT.format(query=queries[qid], k=a.k), provider, model, client, max_tokens=6000)
                rec = {"query_id": qid, "query": queries[qid], "provider": provider, "model": model,
                       "seconds": round(time.time() - t0, 2), "raw": raw}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                done[qid] = rec
                print(f"  {qid}: {len(parse_cases(raw))} cases ({rec['seconds']}s)", flush=True)
    recs = [done[q] for q in queries if q in done]
    if not recs:
        raise SystemExit("No LLM responses yet: the LLM-only baseline stays PENDING.")

    print("loading the citation graph nodes for resolution ...", flush=True)
    res = load_resolver()
    from stage4_validity import get_validity
    from stage5_summary.docs import get_judgment, offsets
    have_text = offsets(path("judgments"))
    engine = scorer = None
    if a.engine:
        from app.engine import Engine
        engine = Engine(load_dense=False)
    if a.nli:
        from stage5_summary.nli import NLIScorer
        scorer = NLIScorer()

    rows = []
    for rec in recs:
        ours = {r["case_id"] for r in engine.search(rec["query"], a.k)["results"]} if engine else None
        for rank, c in enumerate(parse_cases(rec["raw"])[:a.k], 1):
            cid, method = resolve_case(c, res)
            st = str(c.get("status") or "").strip().lower().replace(" ", "_")
            row = {"query_id": rec["query_id"], "rank": rank, "llm_title": c.get("title"),
                   "llm_citation": c.get("citation"), "llm_year": c.get("year"),
                   "llm_status": st if st in LLM_STATUS else "", "case_id": cid or "",
                   "method": method if cid else method or "unresolved", "matched_title": res.nodes[cid].title if cid else "",
                   "full_text": bool(cid and cid in have_text),
                   "our_label": get_validity(cid).get("label") if cid else "",
                   "in_our_topk": (cid in ours) if ours is not None else "",
                   "nli_sentences": 0, "nli_kept": 0}
            if scorer and row["full_text"]:
                from stage0_corpus.splitter import split_sentences
                from stage5_summary.nli import filter_text
                src = [s.text for s in get_judgment(cid).sentences]
                summ = c.get("summary") if isinstance(c.get("summary"), dict) else {}
                for f in FIELDS:
                    if summ.get(f):
                        kept, rej = filter_text(str(summ[f]), src, scorer)
                        n_kept = len(split_sentences(kept)) if kept else 0
                        row["nli_sentences"] += n_kept + len(rej)
                        row["nli_kept"] += n_kept
            rows.append(row)

    with open(out / "cases.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["query_id"])
        w.writeheader()
        w.writerows(rows)
    results = {"provider": recs[0]["provider"], "model": recs[0]["model"], "k": a.k, **score(rows),
               "note": "exists = resolved to a known SC case; unresolved is not proof of hallucination "
                       "(HC cases and SCC/AIR citations of cases the resolver cannot match stay unresolved)"}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
