"""Steps 26-27 — rerank the hybrid top 100 and keep the top 10.

    final = alpha * minmax(ce_score) + (1 - alpha) * log_prior        (log_prior already in [0, 1])

alpha is tuned on the DEV split (SC-PCR dev, or AILA Q1-Q20) and then fixed for test.

    python -m stage3_rerank.rerank --bench sc-pcr            # tune alpha on dev, report P@10 / MRR / nDCG@10 on test
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional

from common.config import cfg, path


def _minmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 1.0 for x in xs]


class Reranker:
    def __init__(self, cross_encoder, prior: Dict[str, float], alpha: float = 0.8, top_k: int = 10):
        self.ce, self.prior, self.alpha, self.top_k = cross_encoder, prior, alpha, top_k

    def ce_scores(self, facts: str, cand_ids: List[str], texts: Dict[str, str]) -> List[float]:
        if self.ce is None:
            return [0.0] * len(cand_ids)
        return self.ce.score(facts, [texts.get(d, "") for d in cand_ids])

    def combine(self, cand_ids: List[str], ce: List[float], alpha: Optional[float] = None, k: Optional[int] = None):
        a = self.alpha if alpha is None else alpha
        ce_n = _minmax(ce) if self.ce is not None else [0.0] * len(ce)
        rows = [{"case_id": d, "ce": round(c, 4), "prior": round(self.prior.get(d, 0.0), 4),
                 "score": a * c + (1 - a) * self.prior.get(d, 0.0)} for d, c in zip(cand_ids, ce_n)]
        rows.sort(key=lambda r: -r["score"])
        return rows[: (k or self.top_k)]


def rerank_run(retriever, reranker, queries: Dict[str, str], labeller=None, depth: int = 100):
    """Returns {qid: (cand_ids, raw_ce_scores)} so alpha can be swept without re-scoring."""
    from stage2_retrieval.query import split_query
    texts = retriever.texts.get("reasoning") or retriever.texts["full"]
    out = {}
    for qid, q in queries.items():
        facts, _ = split_query(q, labeller)
        fused, _ = retriever.search(q, facts=facts, k=depth)
        ids = [d for d, _ in fused]
        out[qid] = (ids, reranker.ce_scores(facts, ids, texts))
    return out


def main():
    from eval.benchmarks import load_benchmark
    from eval.metrics import score
    from stage2_retrieval.build_index import encoder_path
    from stage2_retrieval.dense import load_encoder
    from stage2_retrieval.hybrid import HybridRetriever
    from stage3_rerank.authority import load_prior
    from stage3_rerank.cross_encoder import CrossEncoder

    c = cfg()
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="sc-pcr")
    p.add_argument("--no-dense", action="store_true")
    a = p.parse_args()
    labeller = None
    try:
        from stage1_roles.label_corpus import Labeller
        labeller = Labeller()
    except FileNotFoundError:
        pass
    enc = None if a.no_dense else load_encoder(encoder_path())
    docs = load_benchmark(a.bench, "test")[0]
    retr = HybridRetriever(enc, use_dense=enc is not None, labeller=labeller).build(docs)
    ce_dir = path("models") / "cross_encoder"
    ce = CrossEncoder(str(ce_dir), c["stage3"]["max_len"]) if ce_dir.exists() else None
    prior, src = load_prior()
    rr = Reranker(ce, prior, top_k=c["stage3"]["top_k"])
    print(f"cross-encoder: {'trained' if ce else 'none (prior only)'}; prior: {src}")

    _, qd, qrd = load_benchmark(a.bench, "dev")
    dev = rerank_run(retr, rr, qd, labeller)
    best = max(c["stage3"]["alpha_grid"], key=lambda al: score(
        {q: [r["case_id"] for r in rr.combine(ids, s, al, 100)] for q, (ids, s) in dev.items()}, qrd).get("MAP", 0))
    rr.alpha = best
    _, qt, qrt = load_benchmark(a.bench, "test")
    test = rerank_run(retr, rr, qt, labeller)
    m = score({q: [r["case_id"] for r in rr.combine(ids, s, best, 100)] for q, (ids, s) in test.items()}, qrt)
    print(f"alpha (tuned on dev) = {best}")
    print("test: " + "  ".join(f"{k} {v:.4f}" for k, v in m.items() if k != "queries"))
    out = path("logs") / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"rerank_{a.bench}.json").write_text(json.dumps({"alpha": best, "prior": src, "metrics": m}, indent=2))
    (path("models") / "stage3_alpha.json").write_text(json.dumps({"alpha": best, "bench": a.bench}))


if __name__ == "__main__":
    main()
