"""Standard IR metrics as used on AILA and IL-PCR so results compare directly to published work:
MAP, nDCG@10, P@10, MRR, Recall@100."""
from __future__ import annotations

import math
from typing import Dict, List, Sequence


def precision_at_k(ranked: Sequence[str], rel: Dict[str, int], k: int = 10) -> float:
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / k


def recall_at_k(ranked: Sequence[str], rel: Dict[str, int], k: int = 100) -> float:
    R = sum(1 for v in rel.values() if v > 0)
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / R if R else 0.0


def average_precision(ranked: Sequence[str], rel: Dict[str, int]) -> float:
    R = sum(1 for v in rel.values() if v > 0)
    if not R:
        return 0.0
    hits, s = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if rel.get(d, 0) > 0:
            hits += 1
            s += hits / i
    return s / R


def reciprocal_rank(ranked: Sequence[str], rel: Dict[str, int]) -> float:
    for i, d in enumerate(ranked, 1):
        if rel.get(d, 0) > 0:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], rel: Dict[str, int], k: int = 10) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1))
    ideal = sorted((v for v in rel.values() if v > 0), reverse=True)[:k]
    idcg = sum((2 ** v - 1) / math.log2(i + 1) for i, v in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def evaluate_run(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    qs = [q for q in qrels if q in run]
    if not qs:
        return {}
    agg = {"MAP": 0.0, "nDCG@10": 0.0, "P@10": 0.0, "MRR": 0.0, "Recall@100": 0.0}
    for q in qs:
        r, rel = run[q], qrels[q]
        agg["MAP"] += average_precision(r, rel)
        agg["nDCG@10"] += ndcg_at_k(r, rel, 10)
        agg["P@10"] += precision_at_k(r, rel, 10)
        agg["MRR"] += reciprocal_rank(r, rel)
        agg["Recall@100"] += recall_at_k(r, rel, 100)
    out = {k: v / len(qs) for k, v in agg.items()}
    out["queries"] = len(qs)
    return out


def score(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """MAP, nDCG@10, P@10, MRR, Recall@100 — with ranx when installed (step 10), else the
    implementations above. They agree exactly on binary relevance; on graded qrels ranx's nDCG uses
    linear gain while ndcg_at_k above uses 2^rel - 1."""
    import warnings
    try:
        from ranx import Qrels, Run, evaluate
        warnings.filterwarnings("ignore", module="ranx")
        warnings.filterwarnings("ignore", message=".*unsafe cast.*")
    except ImportError:
        return evaluate_run(run, qrels)
    qs = [q for q in qrels if q in run and any(v > 0 for v in qrels[q].values())]
    if not qs:
        return {}
    R = Run({q: {d: float(len(run[q]) - i) for i, d in enumerate(run[q])} for q in qs})
    Q = Qrels({q: {d: int(v) for d, v in qrels[q].items() if v > 0} for q in qs})
    m = evaluate(Q, R, ["map", "ndcg@10", "precision@10", "mrr", "recall@100"], make_comparable=True)
    return {"MAP": float(m["map"]), "nDCG@10": float(m["ndcg@10"]), "P@10": float(m["precision@10"]),
            "MRR": float(m["mrr"]), "Recall@100": float(m["recall@100"]), "queries": len(qs)}
