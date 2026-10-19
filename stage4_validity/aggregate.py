"""Step 3 of Stage 4: per-mention treatments -> one validity verdict per cited case (validity.json).

Per citing judgment j of cited case c (mentions of c in j are merged first):
    w_bench(j) = min(bench(j) / bench(c), bench_cap)           unknown bench -> default_bench
    w_time(j)  = exp(-(ref_year - year(j)) / recency_tau)       ref_year = latest citing year in the graph
    w(j)       = w_bench * w_time

Rules (the project brief):
  * overruled  if some j overrules c with confidence > overrule_conf and bench(j) >= bench(c);
  * doubted    if the only such overrulings come from a smaller bench, or from a bench we cannot verify
               (unknown size) and the source is the classifier rather than the SCR editors;
  * otherwise  if fewer than min_citing judgments cite c -> insufficient_evidence;
               else weighted majority of followed vs distinguished; if no judgment followed or
               distinguished c (only plain references) -> insufficient_evidence.
Confidence = winning weight share (followed vs distinguished), or the overruling mention's confidence.

Treatment of a mention: the SCR editors' label when present (use_editorial), else the classifier.

    python -m stage4_validity.aggregate                 # -> outputs/validity.json, logs/stage4_aggregate.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional

import pandas as pd

from common.config import cfg, path

from .dataset import out_paths


def mention_treatments(preds: pd.DataFrame, use_editorial: bool) -> pd.DataFrame:
    p = preds.copy()
    p["treatment"], p["t_conf"], p["source"] = p.pred, p.conf, "classifier"
    if use_editorial:
        m = p.editorial.notna()
        p.loc[m, "treatment"], p.loc[m, "t_conf"], p.loc[m, "source"] = p.editorial[m], 1.0, "editorial"
    return p


def judgment_treatment(rows: List[dict], thr: float) -> dict:
    """Merge one citing judgment's mentions of the same cited case."""
    over = [r for r in rows if r["treatment"] == "overruled" and r["t_conf"] > thr]
    if over:
        return max(over, key=lambda r: r["t_conf"])
    score = Counter()
    for r in rows:
        if r["treatment"] in ("followed", "distinguished"):
            score[r["treatment"]] += r["t_conf"]
    if score:
        lab = score.most_common(1)[0][0]
        return max((r for r in rows if r["treatment"] == lab), key=lambda r: r["t_conf"])
    return max(rows, key=lambda r: r["t_conf"]) | {"treatment": "neutral"}


def _num(x) -> Optional[float]:
    """NaN / None / 0 -> None (parquet stores missing ints as NaN)."""
    return x if x and x == x else None


def verdict(case_id: str, citing: Dict[str, List[dict]], nodes: Dict[str, dict], s4: dict, ref_year: int) -> dict:
    cited_bench = _num(nodes.get(case_id, {}).get("bench_size"))
    ev, votes, over_ok, over_small = [], Counter(), [], []
    for j, rows in citing.items():
        t = judgment_treatment(rows, s4["overrule_conf"])
        jb = _num(nodes.get(j, {}).get("bench_size"))
        b_ratio = (jb or s4["default_bench"]) / (cited_bench or s4["default_bench"])
        y = _num(nodes.get(j, {}).get("year"))
        age = max(0, ref_year - int(y)) if y else 0
        w = min(b_ratio, s4["bench_cap"]) * math.exp(-age / s4["recency_tau"])
        e = {"citing_id": j, "sent": t["sentence"], "treatment": t["treatment"], "weight": round(w, 4),
             "confidence": round(float(t["t_conf"]), 4), "source": t["source"]}
        ev.append(e)
        if t["treatment"] == "overruled" and t["t_conf"] > s4["overrule_conf"]:
            verified = jb is not None and cited_bench is not None
            if (verified and jb >= cited_bench) or (not verified and t["source"] == "editorial"):
                over_ok.append(e)
            else:
                over_small.append(e)
        elif t["treatment"] in ("followed", "distinguished"):
            votes[t["treatment"]] += w
    n = len(citing)
    ev.sort(key=lambda e: (e["treatment"] == "neutral", -e["weight"]))
    out = {"case_id": case_id, "n_citing": n, "votes": {k: round(v, 4) for k, v in votes.items()},
           "cited_bench": cited_bench}
    if over_ok:
        return out | {"label": "overruled", "confidence": max(e["confidence"] for e in over_ok), "evidence": ev}
    if over_small:
        return out | {"label": "doubted", "confidence": max(e["confidence"] for e in over_small), "evidence": ev}
    if n < s4["min_citing"] or not votes:
        return out | {"label": "insufficient_evidence", "confidence": None, "evidence": ev}
    lab, w = votes.most_common(1)[0]
    return out | {"label": lab, "confidence": round(w / sum(votes.values()), 4), "evidence": ev}


def aggregate(preds: pd.DataFrame, nodes: pd.DataFrame, s4: dict) -> List[dict]:
    p = mention_treatments(preds, s4.get("use_editorial", True))
    nd = nodes.set_index("case_id")[["year", "bench_size", "title"]].to_dict("index")
    ref_year = int(max(y for j in p.citing_id.unique() if (y := _num(nd.get(j, {}).get("year")))))
    by_case: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in p[["cited_id", "citing_id", "sentence", "treatment", "t_conf", "source"]].to_dict("records"):
        by_case[r["cited_id"]][r["citing_id"]].append(r)
    out = []
    for c, citing in by_case.items():
        v = verdict(c, citing, nd, s4, ref_year)
        v["title"] = nd.get(c, {}).get("title")
        out.append(v)
    return out


def main():
    s4 = cfg()["stage4"]
    p = argparse.ArgumentParser()
    p.add_argument("--preds", default=str(out_paths()[0].parent / "validity_preds.parquet"))
    p.add_argument("--out", default=str(path("validity")))
    a = p.parse_args()
    v = aggregate(pd.read_parquet(a.preds), pd.read_parquet(out_paths()[1]), s4)
    path("validity").parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(v, f, ensure_ascii=False)
    lab = Counter(x["label"] for x in v)
    src = Counter(e["source"] for x in v for e in x["evidence"])
    st = {"cases": len(v), "by_label": dict(lab),
          "cases_with_>=min_citing": sum(x["n_citing"] >= s4["min_citing"] for x in v),
          "coverage_decided": round(sum(x["label"] != "insufficient_evidence" for x in v) / max(len(v), 1), 4),
          "evidence_by_source": dict(src), "settings": s4}
    (path("logs") / "stage4_aggregate.json").write_text(json.dumps(st, indent=2))
    print(json.dumps(st, indent=2))
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
