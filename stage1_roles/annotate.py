"""Hand annotation for the roles the open data lacks (Issue). Round trip:

1) export ~20 SC judgments, pre-labelled by the current model (Issue pre-marked by cue rules):
   python -m stage1_roles.annotate export --n 20 [--max-sentences 120]   # cap length to keep it hand-checkable
2) correct the `gold` column in data/validation/roles_to_annotate.csv. Allowed labels:
   Facts | Ruling by Lower Court | Issue | Argument | Statute | Precedent | Ratio of the decision | Ruling by Present Court
3) import -> data/processed/roles_gold.jsonl (picked up automatically by stage1_roles.train)
   python -m stage1_roles.annotate import [--second-annotator other.csv]   # Cohen's kappa on overlap
"""
import argparse
import csv
import random
import re
from collections import defaultdict

from common.config import cfg, path
from common.io import load_judgments, save_judgments
from common.schema import Judgment, Sentence

ISSUE_CUE = re.compile(r"\b(question|issue|point)s?\b.{0,60}\b(whether|is|are|arise)|\bwhether\b|"
                       r"falls? for (consideration|determination)|for (our|the) consideration", re.I)


def main():
    labels = cfg()["stage1"]["labels"]
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--n", type=int, default=20)
    e.add_argument("--out", default=str(path("validation") / "roles_to_annotate.csv"))
    e.add_argument("--seed", type=int, default=13)
    e.add_argument("--max-sentences", type=int, help="only judgments with at most this many sentences")
    i = sub.add_parser("import")
    i.add_argument("--csv", default=str(path("validation") / "roles_to_annotate.csv"))
    i.add_argument("--second-annotator")
    i.add_argument("--out", default=str(path("processed") / "roles_gold.jsonl"))
    a = p.parse_args()

    if a.cmd == "export":
        docs = [d for d in load_judgments(path("judgments")) if not d.has_roles and d.source == "AWS-SC"
                and (not a.max_sentences or len(d.sentences) <= a.max_sentences)]
        if not docs:
            raise SystemExit("No unlabelled SC judgments in judgments.jsonl yet (download + build_corpus first).")
        random.Random(a.seed).shuffle(docs)
        docs = docs[: a.n]
        try:
            from stage1_roles.label_corpus import Labeller
            Labeller().label(docs)
        except FileNotFoundError:
            print("no trained model yet: rows are pre-filled with Facts")
        path("validation").mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["case_id", "sent_idx", "sentence", "predicted", "gold"])
            for d in docs:
                for s in d.sentences:
                    pred = "Issue" if ISSUE_CUE.search(s.text) and s.idx < 0.5 * len(d.sentences) else (s.role or "Facts")
                    w.writerow([d.case_id, s.idx, s.text, pred, pred])
        print(f"{sum(len(d.sentences) for d in docs)} sentences from {len(docs)} judgments -> {a.out}")
        return

    def read(fn):
        rows = list(csv.DictReader(open(fn, encoding="utf-8")))
        bad = [r for r in rows if r["gold"] not in labels]
        if bad:
            raise SystemExit(f"{len(bad)} invalid labels in {fn}, e.g. {bad[0]['gold']!r}. Allowed: {labels}")
        return rows

    rows = read(a.csv)
    by = defaultdict(list)
    for r in rows:
        by[r["case_id"]].append(r)
    docs = []
    for cid, rs in by.items():
        rs.sort(key=lambda r: int(r["sent_idx"]))
        j = Judgment(case_id=cid, source="hand-annotated", court="SC",
                     sentences=[Sentence(k, r["sentence"], role=r["gold"]) for k, r in enumerate(rs)])
        j.text = " ".join(s.text for s in j.sentences)
        docs.append(j)
    save_judgments(docs, a.out, with_roles=True)
    changed = sum(r["gold"] != r["predicted"] for r in rows) / max(len(rows), 1)
    print(f"{len(docs)} judgments, {len(rows)} sentences -> {a.out}; {changed:.1%} of pre-labels corrected")
    if a.second_annotator:
        from sklearn.metrics import cohen_kappa_score
        other = {(r["case_id"], r["sent_idx"]): r["gold"] for r in read(a.second_annotator)}
        pairs = [(r["gold"], other[(r["case_id"], r["sent_idx"])]) for r in rows if (r["case_id"], r["sent_idx"]) in other]
        if pairs:
            print(f"Cohen's kappa ({len(pairs)} sentences): {cohen_kappa_score(*zip(*pairs)):.3f}")


if __name__ == "__main__":
    main()
