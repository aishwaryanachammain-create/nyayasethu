"""Step 4 — inspect each downloaded dataset: schema, counts, 3 sample records -> data/README.md.

    python -m stage0_corpus.inspect_datasets
"""
import json
from collections import Counter
from datetime import date

from common.config import path
from stage0_corpus.loaders import load_aila, load_bhattacharya, load_sc_judgments

MARK = "<!-- AUTO-INSPECTION -->"


def _sample(docs, k=3, chars=300):
    return [{"case_id": d.case_id, "court": d.court, "date": d.date, "bench_size": d.bench_size,
             "title": d.title[:80], "n_sentences": len(d.sentences), "text": d.text[:chars].replace("\n", " ") + "…"}
            for d in docs[:k]]


def main():
    raw = path("raw")
    out = [MARK, f"\n## Dataset inspection (generated {date.today()})\n"]

    b = raw / "bhattacharya2019" / "semantic-segmentation.zip"
    if b.exists():
        docs = load_bhattacharya(b)
        roles = Counter(s.role for d in docs for s in d.sentences)
        doms = Counter(d.source.split("/", 1)[1] for d in docs)
        n = sum(roles.values())
        out += ["### Bhattacharya et al. 2019 (MIT)",
                f"- judgments: {len(docs)}; sentences: {n}; mean sentences/judgment: {n / len(docs):.0f}",
                f"- domains: {dict(doms)}",
                "- schema: one file per judgment, `sentence<TAB>role` per line (sentence-final periods already stripped)",
                "- role distribution: " + ", ".join(f"{r} {c} ({c / n:.1%})" for r, c in roles.most_common()),
                f"- majority-class baseline accuracy: {roles.most_common(1)[0][1] / n:.1%} ({roles.most_common(1)[0][0]})",
                "- no Issue role, and arguments are not split by party: Issue comes from our own annotation",
                "```json", json.dumps(_sample(docs), indent=1, ensure_ascii=False), "```", ""]

    a = raw / "aila2019" / "AILA_2019_dataset.zip"
    if a.exists():
        docs, q, qr = load_aila(a)
        rel = [len([1 for v in r.values() if v > 0]) for r in qr.values()]
        lens = sorted(len(d.text.split()) for d in docs)
        out += ["### AILA 2019 (CC BY 4.0)",
                f"- candidate judgments: {len(docs)}; queries: {len(q)}; queries with qrels: {len(qr)}",
                f"- relevant per query: mean {sum(rel) / max(len(rel), 1):.1f}, min {min(rel, default=0)}, max {max(rel, default=0)}",
                f"- candidate length (words): median {lens[len(lens) // 2] if lens else 0}",
                "- schema: Object_casedocs/C*.txt, Query_doc.txt (`AILA_Qn||text`), relevance_judgments_priorcases.txt (TREC qrels)",
                "```json", json.dumps(_sample(docs), indent=1, ensure_ascii=False), "```",
                "Sample query: " + next(iter(q.values()), "")[:300] + "…", ""]

    sc = raw / "sc-judgments" / "pdf"
    if sc.exists():
        docs = load_sc_judgments(sc, raw / "sc-judgments" / "metadata", limit=300)
        have = lambda f: sum(1 for d in docs if getattr(d, f)) / max(len(docs), 1)
        out += ["### Indian Supreme Court Judgments, AWS Open Data (CC BY 4.0) — first 300 PDFs",
                f"- PDFs with a text layer: {len(docs)}",
                f"- metadata coverage: date {have('date'):.0%}, bench (from `judge`) {have('bench_size'):.0%}, "
                f"citation aliases {have('aliases'):.0%}, title {have('title'):.0%}",
                "- schema: data/tar/year=YYYY/english/*.tar (PDFs) + metadata/parquet/year=YYYY/metadata.parquet",
                "```json", json.dumps(_sample(docs), indent=1, ensure_ascii=False), "```", ""]

    if len(out) == 2:
        raise SystemExit("No datasets found under data/raw. Run: python -m stage0_corpus.download")
    readme = path("raw").parent / "README.md"
    head = readme.read_text(encoding="utf-8").split(MARK)[0] if readme.exists() else "# Data\n\n"
    readme.write_text(head + "\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\n-> {readme}")


if __name__ == "__main__":
    main()
