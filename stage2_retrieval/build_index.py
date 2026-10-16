"""Steps 19 + 21 — build the role-partitioned BM25 + FAISS indexes for a benchmark's candidates
(or the whole corpus, for the API).

    python -m stage2_retrieval.build_index --bench sc-pcr          # -> models/index_sc-pcr/
    python -m stage2_retrieval.build_index --bench aila
    python -m stage2_retrieval.build_index --corpus                # whole judgments.jsonl -> models/index_corpus/
    python -m stage2_retrieval.build_index --corpus --no-dense     # BM25 only (no GPU / model)
"""
import argparse
from pathlib import Path

from common.config import cfg, path
from common.io import attach_segments, load_judgments
from eval.benchmarks import load_benchmark
from stage2_retrieval.hybrid import HybridRetriever


def encoder_path() -> str:
    ft = path("models") / "biencoder"
    return str(ft) if (ft / "modules.json").exists() else cfg()["models"]["encoder"]


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--bench")
    g.add_argument("--corpus", action="store_true")
    p.add_argument("--no-dense", action="store_true")
    p.add_argument("--encoder", default=None)
    a = p.parse_args()
    if a.corpus:
        docs = load_judgments(path("judgments"))
        if path("segments").exists():
            attach_segments(docs, path("segments"))
        out = path("models") / "index_corpus"
    else:
        docs = load_benchmark(a.bench, "test")[0]
        out = path("models") / f"index_{a.bench}"
    enc = None
    if not a.no_dense:
        from stage2_retrieval.dense import load_encoder
        name = a.encoder or encoder_path()
        print(f"encoder: {name}")
        enc = load_encoder(name)
    r = HybridRetriever(enc, use_dense=enc is not None).build(docs)
    r.save(out)
    print(f"{len(docs)} documents indexed (facts + reasoning partitions, {'BM25 + dense' if enc else 'BM25'}) -> {out}")


if __name__ == "__main__":
    main()
