"""Steps 10-11 — BM25 baseline (full text) + query-document lexical overlap, on AILA and SC-PCR.

    python -m stage2_retrieval.bm25_baseline                 # both benchmarks that are present
    python -m stage2_retrieval.bm25_baseline --bench aila

Writes logs/results/bm25_<bench>.json. The overlap table shows mean Jaccard(query, doc) for relevant
vs non-relevant documents: a small gap means lexical matching cannot separate them, which is the
headroom for dense retrieval.
"""
import argparse
import json
import random

from common.config import cfg, path
from common.text import tokenize
from eval.benchmarks import load_benchmark
from eval.metrics import score
from stage2_retrieval.bm25 import BM25


def lexical_overlap(docs, queries, qrels, seed=13, neg_per_query=50):
    rng = random.Random(seed)
    toks = {d.case_id: set(tokenize(d.text)) for d in docs}
    ids = list(toks)
    jac = lambda a, b: len(a & b) / max(len(a | b), 1)
    rel, non = [], []
    for q, text in queries.items():
        qt = set(tokenize(text))
        r = {d for d, v in qrels.get(q, {}).items() if v > 0 and d in toks}
        rel += [jac(qt, toks[d]) for d in r]
        non += [jac(qt, toks[d]) for d in rng.sample(ids, min(neg_per_query, len(ids))) if d not in r]
    m = lambda x: sum(x) / max(len(x), 1)
    return {"jaccard_relevant": round(m(rel), 4), "jaccard_non_relevant": round(m(non), 4),
            "ratio": round(m(rel) / max(m(non), 1e-9), 3), "pairs": [len(rel), len(non)]}


def run_bm25(bench: str, split: str = "test"):
    c = cfg()["stage2"]
    docs, queries, qrels = load_benchmark(bench, split)
    bm = BM25(c["bm25_k1"], c["bm25_b"]).fit([d.case_id for d in docs], [d.text for d in docs])
    run = {q: [d for d, _ in bm.search(t, c["depth"])] for q, t in queries.items()}
    return score(run, qrels), lexical_overlap(docs, queries, qrels), run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", nargs="*", default=["aila", "sc-pcr"])
    p.add_argument("--split", default="test")
    a = p.parse_args()
    out = path("logs") / "results"
    out.mkdir(parents=True, exist_ok=True)
    for b in a.bench:
        try:
            m, ov, _ = run_bm25(b, a.split)
        except FileNotFoundError as e:
            print(f"{b}: not available ({e.filename}) — download/build it first")
            continue
        if not m:
            print(f"{b}: no queries with relevance judgments in split '{a.split}'")
            continue
        print(f"\n== BM25 baseline on {b} ({a.split}) ==")
        print("  " + "  ".join(f"{k} {v:.4f}" for k, v in m.items() if k != "queries") + f"  (queries {m.get('queries')})")
        print(f"  lexical overlap: relevant {ov['jaccard_relevant']} vs non-relevant {ov['jaccard_non_relevant']} "
              f"(x{ov['ratio']})")
        (out / f"bm25_{b}.json").write_text(json.dumps({"metrics": m, "overlap": ov, "split": a.split}, indent=2))


if __name__ == "__main__":
    main()
