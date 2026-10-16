"""Steps 7-9 — merge the corpora into data/judgments.jsonl (the Week-2 hand-off to Member B).

    python -m stage0_corpus.build_corpus                      # all sources found under data/raw
    python -m stage0_corpus.build_corpus --sc-limit 2000      # quick run

Per judgment: {case_id, source, court, date, bench_size, title, aliases, text, sentences}.
Dedup: first on case_id, then on normalised title + date (same judgment reported twice).
Also writes data/processed/indegree.csv (raw citation in-degree), the authority prior that
Stage 3 uses until Member B's pagerank.csv arrives, and logs metadata coverage.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

from common.config import cfg, path
from common.io import save_judgments
from common.text import normalise
from stage0_corpus.citations import CitationResolver
from stage0_corpus.graph import CitationGraph
from stage0_corpus.loaders import load_aila, load_bhattacharya, load_sc_judgments
from stage0_corpus.metadata import coverage, fill_metadata
from stage0_corpus.splitter import split_sentences
from common.schema import Sentence


def _title_key(d):
    t = re.sub(r"[^a-z0-9 ]+", " ", (d.title or "").lower())
    t = re.sub(r"\b(the|ors?|anr|others|another|and|v|vs|versus)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return (t, d.date) if t and d.date else None


def load_all(raw: Path, sc_limit=None, min_chars=500):
    sources = {}
    b = raw / "bhattacharya2019" / "semantic-segmentation.zip"
    if b.exists():
        sources["bhattacharya2019"] = load_bhattacharya(b)
    a = raw / "aila2019" / "AILA_2019_dataset.zip"
    if a.exists():
        sources["aila2019"] = load_aila(a)[0]
    sc = raw / "sc-judgments" / "pdf"
    if sc.exists():
        sources["sc-judgments"] = load_sc_judgments(sc, raw / "sc-judgments" / "metadata", sc_limit, min_chars)
    return sources


def main():
    c = cfg()
    p = argparse.ArgumentParser()
    p.add_argument("--sc-limit", type=int)
    p.add_argument("--out", default=str(path("judgments")))
    a = p.parse_args()

    sources = load_all(path("raw"), a.sc_limit, c["stage0"]["min_chars"])
    if not sources:
        raise SystemExit("Nothing under data/raw. Run: python -m stage0_corpus.download")
    by_id, by_title, dup_id, dup_title = {}, {}, 0, 0
    reports = []
    for name, docs in sources.items():
        for d in docs:
            if d.case_id in by_id:
                dup_id += 1
                continue
            d.text = normalise(d.text)
            if not d.sentences:                       # Bhattacharya is already split; keep its gold split
                d.sentences = [Sentence(i, s) for i, s in enumerate(split_sentences(d.text))]
            reports.append(fill_metadata(d))
            k = _title_key(d)
            if k and k in by_title:
                dup_title += 1
                keep = by_id[by_title[k]]
                keep.aliases = sorted(set(keep.aliases) | set(d.aliases))
                continue
            by_id[d.case_id] = d
            if k:
                by_title[k] = d.case_id
    docs = list(by_id.values())

    # citations -> in-degree prior (Member B owns the full graph; this is A's stand-in)
    resolver = CitationResolver()
    for d in docs:
        resolver.register(d.case_id, d.aliases, d.title)
    for d in docs:
        for s in d.sentences:
            s.citations = [x for x in resolver.resolve_sentence(s.text) if x != d.case_id]
    g = CitationGraph.build(docs)
    path("processed").mkdir(parents=True, exist_ok=True)
    with open(path("processed") / "indegree.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "in_degree"])
        for cid, deg in sorted(g.g.in_degree(), key=lambda x: -x[1]):
            w.writerow([cid, deg])

    n = save_judgments(docs, a.out, with_roles=True)   # gold roles kept where a dataset has them (Bhattacharya)
    cov = coverage(reports)
    summary = {"judgments": n, "by_source": dict(Counter(d.source.split("/")[0] for d in docs)),
               "duplicates_removed": {"case_id": dup_id, "title+date": dup_title},
               "sentences": sum(len(d.sentences) for d in docs), "metadata_coverage": cov,
               "citation_graph": g.stats()}
    path("logs").mkdir(parents=True, exist_ok=True)
    (path("logs") / "corpus_build.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n{n} judgments -> {a.out}  (hand this to Member B)")


if __name__ == "__main__":
    main()
