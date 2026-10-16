"""Step 1 of Stage 4: one row per citing mention (graph edge), with weak treatment labels.

For every edge in edges.csv:
  * `sentence`: the citing sentence (page furniture removed), with the citation wrapped in [[ ]];
  * `context`: the edge's +-2 sentence window;
  * `editorial`: the SCR editors' label for this (citing, cited) pair, when the citing judgment's
    Case Law Reference block lists the cited case. When the block gives paragraph numbers, only mentions
    inside those paragraphs get the label;
  * `cue`: the brief's keyword rules on the citing sentence plus the next one;
  * `weak_label`: editorial if present, else cue (None = dropped as ambiguous or unlabelled);
  * `split`: train/dev/test by citing judgment (80/10/10, hashed so it is stable).

Also writes a compact node table (date, bench) for aggregation, and logs/stage4_weak_labels.json.

    python -m stage4_validity.dataset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from common.config import path
from common.io import read_jsonl
from common.schema import Judgment
from stage0_graph.build import body_start, clean
from stage0_graph.extract import mentions, reporter_mentions
from stage0_graph.nodes import merge_corpus, metadata_nodes
from stage0_graph.resolve import Resolver

from .editorial import entries
from .labels import LABELS, cue_label

_PARA = re.compile(r"^\s*(\d{1,3})\s*\.\s+\S")


def out_paths():
    d = path("edges").parent
    return d / "validity_mentions.parquet", d / "validity_nodes.parquet"


def split_of(case_id: str) -> str:
    h = int(hashlib.md5(case_id.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "dev" if h == 1 else "train"


def para_spans(doc: Judgment, start: int):
    """Paragraph number -> (first sentence, first sentence of the next numbered paragraph)."""
    starts = []
    for s in doc.sentences[start:]:
        m = _PARA.match(s.text)
        if m:
            starts.append((int(m.group(1)), s.idx))
    spans = {}
    for i, (n, idx) in enumerate(starts):
        if n not in spans:
            spans[n] = (idx, starts[i + 1][1] if i + 1 < len(starts) else len(doc.sentences))
    return spans


def mark(sentence: str, cite: str) -> str:
    i = sentence.find(cite)
    return sentence if i < 0 else f"{sentence[:i]}[[{cite}]]{sentence[i + len(cite):]}"


def edge_key(cite_text: str) -> str:
    ms = mentions(cite_text)
    return ms[0].key if ms else ""


def editorial_for_doc(doc, doc_edges, res: Resolver, year):
    """Editorial label per edge row index of this doc; None where the block does not list the case."""
    ents = entries(doc)
    if not ents:
        return {}, 0, False
    spans = para_spans(doc, body_start(doc))
    out, n_ent = {}, 0
    for e in ents:
        cids = set()
        for m in reporter_mentions(e.cite_text):
            cid, _, _ = res.resolve(m, year)          # no party name here, so key / page range only
            if cid:
                cids.add(cid)
        hits = [r for r in doc_edges if r.cited_id in cids or r.key in e.keys]
        if not hits:
            continue
        n_ent += 1
        inside = [r for r in hits if any(a <= r.citing_sent_idx < b for p in e.paras for a, b in [spans.get(p, (-1, -1))])]
        for r in (inside or hits):
            prev = out.get(r.Index, "unset")
            out[r.Index] = e.label if prev in ("unset", e.label) else None   # conflicting entries: drop
    return {k: v for k, v in out.items() if v != "unset"}, n_ent, True


def build(judgments: Path, meta_dir: Path, edges_csv: Path):
    nodes = merge_corpus(metadata_nodes(meta_dir), (Judgment.from_dict(r) for r in read_jsonl(judgments)))
    res = Resolver(nodes)
    ed = pd.read_csv(edges_csv)
    ed["key"] = ed.cite_text.map(edge_key)
    by_doc = defaultdict(list)
    for r in ed.itertuples():
        by_doc[r.citing_id].append(r)

    sentence, nxt, editorial = {}, {}, {}
    stats = Counter()
    for raw in read_jsonl(judgments):
        rows = by_doc.get(raw["case_id"])
        if not rows:
            continue
        d = Judgment.from_dict(raw)
        for r in rows:
            i = r.citing_sent_idx
            sentence[r.Index] = mark(clean(d.sentences[i].text), r.cite_text)
            nxt[r.Index] = clean(d.sentences[i + 1].text) if i + 1 < len(d.sentences) else ""
        lab, n_ent, has_block = editorial_for_doc(d, rows, res, nodes[d.case_id].year)
        editorial.update(lab)
        stats["citing_docs_with_editorial_block"] += has_block
        stats["editorial_entries_joined_to_edges"] += n_ent

    ed["sentence"] = ed.index.map(sentence)
    ed["editorial"] = ed.index.map(lambda i: editorial.get(i))
    ed["cue"] = [cue_label(f"{s} {n}") for s, n in zip(ed.sentence, ed.index.map(nxt))]
    ed["weak_label"] = ed.editorial.where(ed.editorial.notna(), ed.cue)
    ed["weak_source"] = ed.editorial.notna().map({True: "editorial", False: "cue"}).where(ed.weak_label.notna())
    ed["split"] = ed.citing_id.map(split_of)
    ed["mention_id"] = ed.citing_id + "|" + ed.citing_sent_idx.astype(str) + "|" + ed.cited_id
    ed = ed.drop(columns=["key"])

    nt = pd.DataFrame([{"case_id": n.case_id, "title": n.title, "date": n.date, "year": n.year,
                        "bench_size": n.bench_size, "in_corpus": n.in_corpus} for n in nodes.values()])
    return ed, nt, stats


def report(ed: pd.DataFrame, stats: Counter):
    vc = lambda s: {k: int(v) for k, v in s.value_counts().reindex(LABELS).fillna(0).items()}
    both = ed[ed.editorial.notna() & ed.cue.notna()]
    return {
        "mentions": len(ed), **{k: int(v) for k, v in stats.items()},
        "editorial_labelled": int(ed.editorial.notna().sum()), "editorial_by_class": vc(ed.editorial),
        "cue_labelled": int(ed.cue.notna().sum()), "cue_by_class": vc(ed.cue),
        "weak_labelled": int(ed.weak_label.notna().sum()), "weak_by_class": vc(ed.weak_label),
        "weak_by_source": {k: int(v) for k, v in ed.weak_source.value_counts().items()},
        "dropped_ambiguous_or_unlabelled": int(ed.weak_label.isna().sum()),
        "cue_agrees_with_editorial_when_both": round(float((both.cue == both.editorial).mean()), 4) if len(both) else None,
        "n_both": len(both),
        "weak_by_split": {s: vc(g.weak_label) for s, g in ed.groupby("split")},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judgments", default=str(path("judgments")))
    p.add_argument("--meta", default=str(path("raw") / "sc-judgments" / "metadata"))
    p.add_argument("--edges", default=str(path("edges")))
    a = p.parse_args()
    ed, nt, stats = build(Path(a.judgments), Path(a.meta), Path(a.edges))
    mp, np_ = out_paths()
    ed.to_parquet(mp, index=False)
    nt.to_parquet(np_, index=False)
    rep = report(ed, stats)
    path("logs").mkdir(parents=True, exist_ok=True)
    (path("logs") / "stage4_weak_labels.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print(f"-> {mp}, {np_}")


if __name__ == "__main__":
    main()
