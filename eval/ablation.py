"""Steps 24 + 33 — ablation table on AILA and SC-PCR:

    BM25 (full text)
    + dense                (BM25 + dense over full text, RRF)
    + role partitioning    (facts -> Facts index, full query -> Reasoning index; 4 lists, RRF)
    + rerank               (cross-encoder only, alpha = 1)
    + PageRank             (alpha tuned on dev)

    python -m eval.ablation                       # both benchmarks, whatever components exist
    python -m eval.ablation --bench aila --no-dense
Writes logs/results/ablation.json and logs/results/ablation.md (paste into the report).
"""
import argparse
import json

from common.config import cfg, path
from eval.benchmarks import load_benchmark
from eval.metrics import score
from stage2_retrieval.bm25 import BM25
from stage2_retrieval.hybrid import HybridRetriever

COLS = ["MAP", "nDCG@10", "P@10", "MRR", "Recall@100"]


def ablate(bench: str, enc, labeller, ce, prior, alpha_grid):
    from stage3_rerank.rerank import Reranker, rerank_run
    docs, q, qr = load_benchmark(bench, "test")
    rows = {}
    bm = BM25().fit([d.case_id for d in docs], [d.text for d in docs])
    rows["BM25 (full text)"] = score({k: [d for d, _ in bm.search(t, 100)] for k, t in q.items()}, qr)
    if enc is not None:
        h = HybridRetriever(enc, use_roles=False).build(docs)
        rows["+ dense"] = score({k: [d for d, _ in h.search(t)[0]] for k, t in q.items()}, qr)
    roles = HybridRetriever(enc, use_dense=enc is not None, use_roles=True, labeller=labeller).build(docs)
    rows["+ role partitioning"] = score({k: [d for d, _ in roles.search(t)[0]] for k, t in q.items()}, qr)
    if ce is not None or prior:
        rr = Reranker(ce, prior)
        test = rerank_run(roles, rr, q, labeller)
        if ce is not None:
            rows["+ rerank"] = score({k: [r["case_id"] for r in rr.combine(i, s, 1.0, 100)] for k, (i, s) in test.items()}, qr)
        _, qd, qrd = load_benchmark(bench, "dev")
        dev = rerank_run(roles, rr, qd, labeller)
        grid = alpha_grid if ce is not None else [0.0]
        best = max(grid, key=lambda a: score({k: [r["case_id"] for r in rr.combine(i, s, a, 100)]
                                              for k, (i, s) in dev.items()}, qrd).get("MAP", 0))
        rows[f"+ PageRank (alpha={best})"] = score({k: [r["case_id"] for r in rr.combine(i, s, best, 100)]
                                                     for k, (i, s) in test.items()}, qr)
    return rows


def to_md(all_rows):
    lines = []
    for bench, rows in all_rows.items():
        lines += [f"\n**{bench}** (test queries: {next(iter(rows.values())).get('queries', '?')})\n",
                  "| System | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
        for name, m in rows.items():
            lines.append(f"| {name} | " + " | ".join(f"{m.get(c, 0):.4f}" for c in COLS) + " |")
    return "\n".join(lines)


def main():
    c = cfg()
    p = argparse.ArgumentParser()
    p.add_argument("--bench", nargs="*", default=["aila", "sc-pcr"])
    p.add_argument("--no-dense", action="store_true")
    a = p.parse_args()
    enc = None
    if not a.no_dense:
        from stage2_retrieval.build_index import encoder_path
        from stage2_retrieval.dense import load_encoder
        enc = load_encoder(encoder_path())
    labeller = None
    try:
        from stage1_roles.label_corpus import Labeller
        labeller = Labeller()
    except FileNotFoundError:
        pass
    ce = None
    if (path("models") / "cross_encoder").exists():
        from stage3_rerank.cross_encoder import CrossEncoder
        ce = CrossEncoder(str(path("models") / "cross_encoder"), c["stage3"]["max_len"])
    from stage3_rerank.authority import load_prior
    prior, src = load_prior()
    res = {}
    for b in a.bench:
        try:
            res[b] = ablate(b, enc, labeller, ce, prior, c["stage3"]["alpha_grid"])
        except FileNotFoundError as e:
            print(f"{b}: not available ({e.filename})")
    if not res:
        raise SystemExit("No benchmark available.")
    out = path("logs") / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps({"prior": src, "results": res}, indent=2))
    md = to_md(res)
    (out / "ablation.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
