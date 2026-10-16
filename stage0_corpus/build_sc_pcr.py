"""Open replacement for IL-PCR (which is a gated download): build "SC-PCR" from the CC BY 4.0
Supreme Court judgments in data/judgments.jsonl.

  * QUERY = a judgment that no other corpus judgment cites, with its citations masked
    (reporter strings -> <CITATION>, cited case names -> <CASE>);
  * RELEVANT = the earlier corpus judgments it actually cites;
  * queries and candidates are disjoint, so a query can never retrieve itself;
  * splits are by decision year: train < dev < test (no leakage from the future).

    python -m stage0_corpus.build_sc_pcr               # -> data/processed/sc-pcr/
Also logs step 11 (lexical overlap): mean Jaccard of query vs relevant vs random candidates.
NOTE for Member B: query texts here are MASKED — build the citation graph from judgments.jsonl.
"""
import argparse
import json
import random
import re
from pathlib import Path

from common.config import path
from common.io import load_judgments
from common.text import tokenize
from stage0_corpus.citations import CitationResolver, extract_citations
from stage0_corpus.graph import CitationGraph


def name_patterns(doc):
    pats = []
    if doc.title:
        first = re.split(r"\s+v(?:s)?\.?\s+", doc.title, maxsplit=1)[0].split()
        pats.append(re.compile(re.escape(doc.title.split(",")[0]).replace(r"\ ", r"\s+"), re.I))
        if len(first) >= 2:
            pats.append(re.compile(r"\b" + r"\s+".join(map(re.escape, first[:2])) + r"\b", re.I))
    for a in doc.aliases:
        pats.append(re.compile(re.escape(a).replace(r"\ ", r"\s*"), re.I))
    return pats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judgments", default=str(path("judgments")))
    p.add_argument("--sources", default="AWS-SC", help="comma list of Judgment.source prefixes to use")
    p.add_argument("--min-cited", type=int, default=2)
    p.add_argument("--dev-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--out", default=str(path("processed") / "sc-pcr"))
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()

    keep = tuple(s.strip() for s in a.sources.split(","))
    docs = [d for d in load_judgments(a.judgments) if d.source.startswith(keep)]
    D = {d.case_id: d for d in docs}
    res = CitationResolver()
    for d in docs:
        res.register(d.case_id, d.aliases, d.title)
    for d in docs:
        for s in d.sentences:
            s.citations = [x for x in res.resolve_sentence(s.text) if x != d.case_id]
    g = CitationGraph.build(docs)
    print(f"{len(docs)} judgments; graph {g.stats()}")
    indeg = path("processed") / "indegree.csv"
    if not indeg.exists():                       # authority prior fallback for Stage 3
        indeg.parent.mkdir(parents=True, exist_ok=True)
        indeg.write_text("case_id,in_degree\n" + "".join(f"{k},{v}\n" for k, v in g.g.in_degree()))

    def mask(doc, cited):
        text = doc.text
        for c in sorted(extract_citations(text), key=lambda c: -c.span[0]):
            text = text[:c.span[0]] + "<CITATION>" + text[c.span[1]:]
        for cid in cited:
            for pat in name_patterns(D[cid]):
                text = pat.sub("<CASE>", text)
        return re.sub(r"(<CITATION>[\s,;]*)+", "<CITATION> ", text)

    queries = []
    for d in docs:
        if g.g.in_degree(d.case_id) > 0:
            continue
        cited = sorted({c for c in g.g.successors(d.case_id) if not (d.year and D[c].year and D[c].year > d.year)})
        if len(cited) >= a.min_cited:
            queries.append({"id": d.case_id, "year": d.year or 0, "text": mask(d, cited), "relevant_candidates": cited})
    if not queries:
        raise SystemExit("No judgment cites >= --min-cited corpus judgments: add more years (metadata titles/"
                         "citations are what make citations resolvable).")
    queries.sort(key=lambda q: (q["year"], q["id"]))
    n = len(queries)
    nt, nd = int(n * a.test_frac), int(n * a.dev_frac)
    splits = {"train": queries[: n - nt - nd], "dev": queries[n - nt - nd: n - nt], "test": queries[n - nt:]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, qs in splits.items():
        with open(out / f"{name}_queries.jsonl", "w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps({k: q[k] for k in ("id", "text", "relevant_candidates")}, ensure_ascii=False) + "\n")
    qids = {q["id"] for q in queries}
    with open(out / "candidates.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            if d.case_id not in qids:
                f.write(json.dumps({"id": d.case_id, "text": d.text}, ensure_ascii=False) + "\n")

    rng = random.Random(a.seed)
    toks = {d.case_id: set(tokenize(d.text)) for d in docs if d.case_id not in qids}
    pool = list(toks)
    jac = lambda x, y: len(x & y) / max(len(x | y), 1)
    rel, rnd = [], []
    for q in queries[:500]:
        qt = set(tokenize(q["text"]))
        rel += [jac(qt, toks[c]) for c in q["relevant_candidates"]]
        rnd += [jac(qt, toks[rng.choice(pool)]) for _ in q["relevant_candidates"]]
    stats = {"queries": {k: len(v) for k, v in splits.items()}, "candidates": len(pool),
             "mean_relevant_per_query": sum(len(q["relevant_candidates"]) for q in queries) / n,
             "year_ranges": {k: [v[0]["year"], v[-1]["year"]] for k, v in splits.items() if v},
             "jaccard_query_vs_relevant": round(sum(rel) / max(len(rel), 1), 4),
             "jaccard_query_vs_random": round(sum(rnd) / max(len(rnd), 1), 4)}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
