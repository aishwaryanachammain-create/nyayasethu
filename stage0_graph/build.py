"""Stage 0 (Member B) — citation graph over Member A's corpus, PageRank, graph statistics.

    python -m stage0_graph.build                     # full corpus from paths.judgments
    python -m stage0_graph.build --limit 300         # quick run

Reads   paths.judgments (Member A's corpus; case_ids used as-is), AWS metadata under paths.raw
Writes  paths.edges      {citing_id, cited_id, citing_sent_idx, context, cite_text, method, score}
        paths.pagerank   {case_id, pagerank, in_degree, in_corpus}
        logs/graph_build.json, eval/figures/indegree_distribution.png,
        eval/citation_precision_sample.csv (blank columns for manual checking)
Only the judgment body is scanned: SCR front matter (headnote, Case Law Reference, counsel) is skipped.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import networkx as nx

from common.config import ROOT, path
from common.io import read_jsonl
from common.schema import Judgment

from .extract import mentions
from .nodes import merge_corpus, metadata_nodes
from .resolve import Resolver

_BODY = [re.compile(r"(?:judgment|order|opinion)s? of the court (?:was|were) (?:delivered|passed|pronounced)", re.I),
         re.compile(r"\bthe following (?:judgment|order)", re.I),
         re.compile(r"\badvs?\.? for the (?:appearing )?(?:parties|appellants?|respondents?|petitioners?)", re.I)]
_MARGIN = re.compile(r"(?:\b[A-H]\b\s+){3,}(?:[A-H]\b)?")
# SCR page furniture that the PDF text layer puts inside sentences:
#   even pages  "SUPREME COURT REPORTS [2018] 10 S.C.R."
#   odd pages   "M/S. GOEL GANGA DEVELOPERS v. U.O.I.THR. SECY. MINISTRY OF ENVIRONMENT [DEEPAK GUPTA, J.]"
#   first page  "[2018] 10 S.C.R. 1005" (the case's own citation; resolves to itself and is dropped)
_RUNNING = [re.compile(r"SUPREME COURT REPORTS\s*\[\d{4}\]\s*(?:SUPP\.?\s*)?\d*\s*S\.\s?C\.\s?R\.?"),
            # possessive tokens and bounded counts: unbounded nesting backtracked for hours on ALL-CAPS text
            re.compile(r"(?:[A-Z0-9][A-Z0-9.,&'()/\-]*+(?:\s+(?:[A-Z0-9][A-Z0-9.,&'()/\-]*+|v\.|&)){0,30}\s+)?"
                       r"\[(?:[A-Z][A-Z.'\-]*+\s?){1,12}(?:,\s*)?(?:J|JJ|CJI)\.\]")]


def body_start(doc: Judgment) -> int:
    """First sentence of the judgment proper. 0 when no SCR front-matter marker is found."""
    limit = max(len(doc.sentences) // 2, 1)
    for pat in _BODY:
        for s in doc.sentences[:limit]:
            if pat.search(s.text):
                return s.idx
    return 0


def clean(s: str) -> str:
    for pat in _RUNNING:
        s = pat.sub(" ", s)
    return re.sub(r"\s+", " ", _MARGIN.sub(" ", s)).strip()


def context(doc: Judgment, i: int, w: int = 2) -> str:
    return " ".join(clean(s.text) for s in doc.sentences[max(0, i - w): i + w + 1])


def iter_docs(p, limit=None):
    for i, r in enumerate(read_jsonl(p)):
        if limit and i >= limit:
            break
        yield Judgment.from_dict(r)


def build(judgments: Path, meta_dir: Path, limit=None, seed: int = 13):
    nodes = metadata_nodes(meta_dir)
    n_meta = len(nodes)
    merge_corpus(nodes, iter_docs(judgments, limit))
    res = Resolver(nodes)
    edges, sample_pool, sample_edges = [], [], []
    stats = Counter()
    by_kind = defaultdict(Counter)
    citing_with_edges = set()
    rng = random.Random(seed)
    for d in iter_docs(judgments, limit):
        start = body_start(d)
        stats["docs"] += 1
        if stats["docs"] % 500 == 0:
            print(f"  {stats['docs']} docs, {stats['mentions']} mentions, {len(edges)} edges", flush=True)
        stats["sentences_skipped_front_matter"] += start
        seen = set()
        for s in d.sentences[start:]:
            for m in mentions(clean(s.text)):
                cid, method, score = res.resolve(m, nodes[d.case_id].year)
                kind = m.kind if m.key else "NAME"
                by_kind[kind]["mentions"] += 1
                by_kind[kind][method if cid != d.case_id else "self_citation_dropped"] += 1
                stats["mentions"] += 1
                if rng.random() < 0.02:
                    sample_pool.append((d.case_id, s.idx, clean(s.text), m))
                if not cid or cid == d.case_id:
                    continue
                stats["resolved"] += 1
                if (d.case_id, cid, s.idx) in seen:
                    continue
                seen.add((d.case_id, cid, s.idx))
                row = {"citing_id": d.case_id, "cited_id": cid, "citing_sent_idx": s.idx,
                       "context": context(d, s.idx), "cite_text": m.raw, "method": method, "score": round(score, 1)}
                edges.append(row)
                citing_with_edges.add(d.case_id)
                if rng.random() < 0.05:
                    sample_edges.append((row, m.name))
    return nodes, n_meta, edges, stats, by_kind, citing_with_edges, sample_pool, sample_edges


def graph_stats(nodes, edges, min_citing=3):
    g = nx.DiGraph()
    g.add_nodes_from(nodes)
    for e in edges:
        g.add_edge(e["citing_id"], e["cited_id"])           # judgment level: one edge per (citing, cited)
    indeg = dict(g.in_degree())
    cited = [v for v in indeg.values() if v > 0]
    corpus = [c for c, n in nodes.items() if n.in_corpus]
    st = {"nodes_total": g.number_of_nodes(), "nodes_in_corpus": len(corpus),
          "edges_judgment_level": g.number_of_edges(), "edges_mention_level": len(edges),
          "cited_nodes": len(cited),
          "share_of_cited_with_>=3_citing": round(sum(v >= min_citing for v in cited) / max(len(cited), 1), 4),
          "nodes_with_>=3_citing": sum(v >= min_citing for v in cited),
          "share_of_corpus_with_>=3_citing": round(sum(indeg[c] >= min_citing for c in corpus) / max(len(corpus), 1), 4),
          "max_in_degree": max(cited) if cited else 0,
          "mean_in_degree_of_cited": round(sum(cited) / max(len(cited), 1), 3),
          "in_degree_histogram": dict(sorted(Counter(min(v, 50) for v in cited).items()))}
    return g, st


def write_pagerank(g, nodes, out: Path):
    pr = nx.pagerank(g, alpha=0.85) if g.number_of_edges() else {n: 1 / max(len(g), 1) for n in g}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "pagerank", "in_degree", "in_corpus"])      # A's authority.py reads `pagerank`
        for cid, v in sorted(pr.items(), key=lambda x: -x[1]):
            w.writerow([cid, f"{v:.8g}", g.in_degree(cid), int(nodes[cid].in_corpus)])


def plot_indegree(g, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    deg = Counter(v for _, v in g.in_degree() if v > 0)
    if not deg:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    xs = sorted(deg)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, [deg[x] for x in xs], s=12, color="#0d9488")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("in-degree (citing judgments)"); ax.set_ylabel("number of cases")
    ax.axvline(3, color="#9ca3af", ls="--", lw=1)
    ax.set_title("Citation in-degree distribution (cited cases)")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def write_samples(nodes, pool, sample_edges, out: Path, seed=13, n=150):
    rng = random.Random(seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():                                    # never overwrite hand labels
        with open(out, newline="", encoding="utf-8") as f:
            if any(r.get("extraction_correct") or r.get("resolution_correct") for r in csv.DictReader(f)):
                out = out.with_name(out.stem + "_new.csv")
                print(f"  {out.name}: existing sample has labels, kept it")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["check", "citing_id", "sent_idx", "sentence_or_context", "extracted", "kind", "party_name",
                    "resolved_to", "resolved_title", "resolved_date", "method", "score",
                    "extraction_correct", "resolution_correct"])
        for cid, idx, sent, m in rng.sample(pool, min(n, len(pool))):
            w.writerow(["extraction", cid, idx, sent, m.raw, m.kind if m.key else "NAME", m.name, "", "", "", "", "", "", ""])
        for row, name in rng.sample(sample_edges, min(n, len(sample_edges))):
            nd = nodes[row["cited_id"]]
            w.writerow(["resolution", row["citing_id"], row["citing_sent_idx"], row["context"], row["cite_text"], "",
                        name, row["cited_id"], nd.title, nd.date, row["method"], row["score"], "", ""])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judgments", default=str(path("judgments")))
    p.add_argument("--meta", default=str(path("raw") / "sc-judgments" / "metadata"))
    p.add_argument("--limit", type=int)
    p.add_argument("--edges", default=str(path("edges")))
    p.add_argument("--pagerank", default=str(path("pagerank")))
    a = p.parse_args()

    nodes, n_meta, edges, stats, by_kind, citing, pool, samp = build(Path(a.judgments), Path(a.meta), a.limit)
    out = Path(a.edges)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["citing_id", "cited_id", "citing_sent_idx", "context", "cite_text", "method", "score"])
        w.writeheader()
        w.writerows(edges)
    g, gst = graph_stats(nodes, edges)
    write_pagerank(g, nodes, Path(a.pagerank))
    plot_indegree(g, ROOT / "eval" / "figures" / "indegree_distribution.png")
    write_samples(nodes, pool, samp, ROOT / "eval" / "citation_precision_sample.csv")

    rk = {k: {**dict(v), "resolution_rate": round(sum(x for m, x in v.items() if m in ("key", "key_range", "name_year", "name_only"))
                                                   / max(v["mentions"], 1), 4)} for k, v in by_kind.items()}
    reporter = [v for k, v in by_kind.items() if k != "NAME"]
    rep_m = sum(v["mentions"] for v in reporter)
    rep_r = sum(v.get("key", 0) + v.get("key_range", 0) + v.get("name_year", 0) for v in reporter)
    summary = {"metadata_nodes": n_meta, "corpus_docs": stats["docs"],
               "citing_docs_with_edges": len(citing),
               "front_matter_sentences_skipped": stats["sentences_skipped_front_matter"],
               "mentions": stats["mentions"], "mentions_resolved": stats["resolved"],
               "resolution_rate_all_mentions": round(stats["resolved"] / max(stats["mentions"], 1), 4),
               "resolution_rate_reporter_citations": round(rep_r / max(rep_m, 1), 4),
               "by_kind": rk, "graph": gst}
    path("logs").mkdir(parents=True, exist_ok=True)
    (path("logs") / "graph_build.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_kind"}, indent=2))
    print(f"{len(edges)} edges -> {out}; pagerank -> {a.pagerank}")


if __name__ == "__main__":
    main()
