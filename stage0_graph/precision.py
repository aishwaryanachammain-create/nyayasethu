"""Citation-graph precision from the hand-checked sample (eval/citation_precision_sample.csv).

A person fills `extraction_correct` (extraction rows) and `resolution_correct` (resolution rows) with yes/no.
`assistant_correct` is an AI pre-label for the reviewer and is never scored.

    python -m stage0_graph.precision        # -> eval/graph_precision.json (PENDING until rows are filled)
"""
from __future__ import annotations

import json
import math

import pandas as pd

from common.config import ROOT

SAMPLE = ROOT / "eval" / "citation_precision_sample.csv"
OUT = ROOT / "eval" / "graph_precision.json"
YES, NO = {"yes", "y", "1", "true", "correct"}, {"no", "n", "0", "false", "incorrect"}


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def score(df: pd.DataFrame) -> dict:
    out = {}
    for check, col in (("extraction", "extraction_correct"), ("resolution", "resolution_correct")):
        v = df[df.check == check][col].astype(str).str.strip().str.lower()
        k, n = int(v.isin(YES).sum()), int(v.isin(YES | NO).sum())
        if not n:
            out[check] = f"PENDING: 0 of {len(v)} rows checked"
            continue
        r = {"checked": n, "of": len(v), "correct": k, "precision": round(k / n, 4), "ci95": wilson(k, n)}
        if check == "extraction":
            sub = df[df.check == check].assign(v=v)
            for name, part in (("reporter_citations", sub[sub.kind != "NAME"]), ("bare_case_names", sub[sub.kind == "NAME"])):
                kk, nn = int(part.v.isin(YES).sum()), int(part.v.isin(YES | NO).sum())
                r[name] = {"checked": nn, "precision": round(kk / nn, 4) if nn else None}
        out[check] = r
    return out


def main():
    res = score(pd.read_csv(SAMPLE, keep_default_na=False, dtype=str))
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
