"""Batch: summaries for the top-k results of each query -> paths.summary (summary.json, CONTRACT schema).

    python -m stage5_summary.run --queries queries.json [--k 5] [--mode extractive]

queries.json: {"<query_id>": "<query text>", ...}. Result lists come from paths.retrieval
([{query_id, results: [{case_id, ...}]}]); a query without results there is skipped.
"""
from __future__ import annotations

import argparse
import json
import time

from common.config import path

from . import summarize
from .fields import FIELDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True)
    p.add_argument("--retrieval", default=str(path("retrieval")))
    p.add_argument("--out", default=str(path("summary")))
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--mode")
    a = p.parse_args()
    queries = json.load(open(a.queries))
    out, t0 = [], time.time()
    for r in json.load(open(a.retrieval)):
        q = queries.get(r["query_id"])
        if not q:
            continue
        for hit in r["results"][:a.k]:
            s = summarize(q, hit["case_id"], mode=a.mode)
            if s["status"] != "ok":
                continue
            out.append({"query_id": r["query_id"], "case_id": hit["case_id"], **{f: s[f] for f in FIELDS},
                        "rejected": s["rejected"], "roles_source": s["roles_source"], "generator": s["generator"]})
            print(f"  {r['query_id']} {hit['case_id']} ({time.time() - t0:.0f}s)", flush=True)
    path("summary").parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"{len(out)} summaries -> {a.out}")


if __name__ == "__main__":
    main()
