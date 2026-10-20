"""Hand check of the NLI faithfulness filter (§3.3 of eval/RESULTS_B.md).

A blinded sample of query-focused summary sentences the filter kept and rejected, shuffled together. A person
marks each sentence `supported` yes/no (or blank if they cannot tell) from the judgment, without seeing whether
the filter kept it. The scorer then reports:
  * rejection precision: share of rejected sentences that are really unsupported (rejections that were right);
  * kept precision: share of kept sentences that are really supported (what the filter lets through is faithful).

    python -m stage5_summary.nli_check --make     # -> eval/nli_check_sample.csv (40 kept + 40 rejected)
    streamlit run eval/check_tool.py              # fill `supported` (or edit the CSV by hand)
    python -m stage5_summary.nli_check --score --provenance "<who checked>"   # -> eval/nli_check_results.json
"""
from __future__ import annotations

import argparse
import json
import random

import pandas as pd

from common.config import ROOT
from stage0_graph.precision import NO, YES, wilson

from .eval_pairs import FIELDS, PAIRS
from .nli import top_premise

SAMPLE = ROOT / "eval" / "nli_check_sample.csv"
OUT = ROOT / "eval" / "nli_check_results.json"
EVIDENCE_K = 5


def make(n_each: int, seed: int):
    from stage0_corpus.splitter import split_sentences
    from .docs import get_judgment
    kept, rej = [], []
    for r in map(json.loads, open(PAIRS, encoding="utf-8")):
        base = {"query_id": r["query_id"], "case_id": r["case_id"], "title": r["title"]}
        for fl in FIELDS:
            for s in split_sentences(r["query_focused"][fl]) if r["query_focused"][fl] else []:
                kept.append(base | {"field": fl, "sentence": s, "filter": "kept", "p_entail": ""})
        for x in r["qf_rejected"]:
            rej.append(base | {"field": x["field"], "sentence": x["sentence"], "filter": "rejected",
                               "p_entail": x["p_entail"]})
    rng = random.Random(seed)
    rows = rng.sample(kept, min(n_each, len(kept))) + rng.sample(rej, min(n_each, len(rej)))
    rng.shuffle(rows)
    docs = {}
    for i, row in enumerate(rows):
        if row["case_id"] not in docs:
            docs[row["case_id"]] = [s.text for s in get_judgment(row["case_id"]).sentences]
        row["row_id"] = f"n{i:03d}"
        row["evidence"] = top_premise(row["sentence"], docs[row["case_id"]], EVIDENCE_K)
        row["supported"] = ""
        row["note"] = ""
    cols = ["row_id", "query_id", "case_id", "title", "field", "sentence", "evidence", "supported", "note",
            "filter", "p_entail"]
    pd.DataFrame(rows)[cols].to_csv(SAMPLE, index=False)
    print(f"{len(rows)} rows ({len(kept)} kept, {len(rej)} rejected in the pool) -> {SAMPLE}")


def score(provenance: str) -> dict:
    df = pd.read_csv(SAMPLE, keep_default_na=False)
    v = df.supported.astype(str).str.strip().str.lower()
    out = {"provenance": provenance, "sample": SAMPLE.name, "rows": len(df)}
    for name, filt, good in (("rejection_precision", "rejected", NO), ("kept_precision", "kept", YES)):
        s = v[df["filter"] == filt]
        n, k = int(s.isin(YES | NO).sum()), int(s.isin(good).sum())
        out[name] = ({"checked": n, "of": len(s), "correct": k, "precision": round(k / n, 4), "ci95": wilson(k, n)}
                     if n else f"PENDING: 0 of {len(s)} rows checked")
    out["note"] = ("rejection_precision = rejected sentences judged unsupported; kept_precision = kept sentences "
                   "judged supported; blank (cannot tell) rows are skipped")
    OUT.write_text(json.dumps(out, indent=2))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--make", action="store_true")
    p.add_argument("--score", action="store_true")
    p.add_argument("--n", type=int, default=40, help="sentences per group (kept, rejected)")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--provenance", default="not recorded: state who checked the sample")
    a = p.parse_args()
    if a.make:
        if SAMPLE.exists():
            raise SystemExit(f"{SAMPLE} exists; delete it first to resample (it may hold checks).")
        make(a.n, a.seed)
    if a.score:
        print(json.dumps(score(a.provenance), indent=2))


if __name__ == "__main__":
    main()
