"""Step 34 — pick the 5 worst test queries of the full system and diagnose each.

For every relevant document of a failed query it records: whether it is in the index at all,
its rank in each ranked list (BM25/dense x Facts/Reasoning) and after fusion, the lexical overlap
with the query vs the top wrong hit, and how much of it Stage 1 labelled as Facts / Reasoning.
A first-guess cause is assigned (you confirm it by reading the documents):
  missing document     relevant doc not in the candidate set
  wrong role labels    found by full-text BM25 but not by either role partition (partition empty/mislabelled)
  vocabulary gap       relevant doc shares fewer words with the query than the wrong top hit
  ranking              retrieved within the top 100 but ranked too low

    python -m eval.error_analysis --bench sc-pcr --no-dense
-> logs/results/error_analysis_<bench>.md
"""
import argparse

from common.config import cfg, path
from common.text import tokenize
from eval.benchmarks import load_benchmark
from eval.metrics import average_precision
from stage2_retrieval.bm25 import BM25
from stage2_retrieval.hybrid import HybridRetriever


def main():
    c = cfg()["stage2"]
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="sc-pcr")
    p.add_argument("--n", type=int, default=5)
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
    docs, q, qr = load_benchmark(a.bench, "test")
    D = {d.case_id: d for d in docs}
    full = BM25().fit(list(D), [d.text for d in docs])
    hyb = HybridRetriever(enc, use_dense=enc is not None, labeller=labeller).build(docs)
    toks = {k: set(tokenize(d.text)) for k, d in D.items()}
    jac = lambda x, y: len(x & y) / max(len(x | y), 1)
    rows = []
    for qid, text in q.items():
        fused, runs = hyb.search(text)
        ranked = [d for d, _ in fused]
        rows.append((average_precision(ranked, qr.get(qid, {})), qid, text, ranked, runs))
    rows.sort(key=lambda r: r[0])
    md = [f"# Error analysis — {a.bench}, {a.n} worst test queries (hybrid retriever)\n"]
    for ap, qid, text, ranked, runs in rows[: a.n]:
        qt = set(tokenize(text))
        top_wrong = next((d for d in ranked if qr.get(qid, {}).get(d, 0) == 0), None)
        full_rank = {d: i + 1 for i, (d, _) in enumerate(full.search(text, 100))}
        md += [f"## {qid} — AP {ap:.3f}", f"> {text[:400].strip()}…\n",
               f"Top wrong hit: `{top_wrong}` (Jaccard with query {jac(qt, toks.get(top_wrong, set())):.3f})\n",
               "| relevant doc | in index | fused rank | full-text BM25 rank | " + " | ".join(runs) + " | Jaccard | Facts / Reasoning sentences | first-guess cause |",
               "|---|---|---|---|" + "---|" * len(runs) + "---|---|---|"]
        for d in [x for x, v in qr.get(qid, {}).items() if v > 0]:
            if d not in D:
                md.append(f"| {d} | no | – | – | " + " | ".join("–" for _ in runs) + " | – | – | missing document |")
                continue
            fr = ranked.index(d) + 1 if d in ranked else None
            rr = {n: next((i + 1 for i, (x, _) in enumerate(r) if x == d), None) for n, r in runs.items()}
            nf = sum(1 for s in D[d].sentences if s.role in c["facts_roles"])
            nr = sum(1 for s in D[d].sentences if s.role in c["reasoning_roles"])
            j = jac(qt, toks[d])
            if full_rank.get(d) and all(v is None for v in rr.values()):
                cause = "wrong role labels"
            elif top_wrong and j < jac(qt, toks[top_wrong]):
                cause = "vocabulary gap"
            else:
                cause = "ranking"
            md.append(f"| {d} | yes | {fr or '>100'} | {full_rank.get(d, '>100')} | "
                      + " | ".join(str(v or ">100") for v in rr.values())
                      + f" | {j:.3f} | {nf} / {nr} | {cause} |")
        md.append("\n**Explanation (write after reading the documents):** …\n")
    out = path("logs") / "results" / f"error_analysis_{a.bench}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:40]))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
