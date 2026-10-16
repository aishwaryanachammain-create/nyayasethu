"""Step 6 (part 2) — score the splitter against the corrected sample.

Reads data/validation/splitter_annotate.txt (after manual correction) and the untouched
splitter_sample.jsonl, then reports boundary precision / recall / F1 and the ERROR RATE

    error rate = (wrong boundaries + missed boundaries) / gold sentences

It logs every error with context to logs/splitter_errors.tsv (use these to grow the exception
lists in corpus/splitter.py, then re-run) and appends a dated summary to logs/week1_log.md.

    python -m stage0_corpus.splitter_validate [--rerun-splitter]
"""
import argparse
import csv
import datetime as dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

from common.config import path
from stage0_corpus.splitter import split_sentences

p = argparse.ArgumentParser()
p.add_argument("--dir", default=str(path("validation")))
p.add_argument("--logs", default=str(path("logs")))
p.add_argument("--rerun-splitter", action="store_true",
               help="re-split the raw passages with the CURRENT splitter (use after improving it)")
a = p.parse_args()

d = Path(a.dir)
raw = {json.loads(l)["pid"]: json.loads(l) for l in open(d / "splitter_sample.jsonl", encoding="utf-8")}
gold, cur = {}, None
for line in open(d / "splitter_annotate.txt", encoding="utf-8"):
    line = line.rstrip("\n")
    if line.startswith("#") and not line.startswith("###"):
        continue
    if line.startswith("###"):
        cur = line.split()[1]
        gold[cur] = []
    elif cur and line.strip():
        gold[cur].append(line.strip())

squash = lambda s: re.sub(r"\s+", "", s)


def boundaries(sents):
    pos, out = 0, []
    for s in sents:
        pos += len(squash(s))
        out.append(pos)
    return out[:-1]          # the passage end is not a decision


tp = fp = fn = n_gold = 0
errors, edited = [], []
for pid, g in gold.items():
    r = raw.get(pid)
    if r is None:
        continue
    if squash(" ".join(g)) != squash(r["text"]):
        edited.append(pid)    # annotator changed characters, not just line breaks
        continue
    pred = split_sentences(r["text"]) if a.rerun_splitter else r["predicted"]
    gb, pb = set(boundaries(g)), set(boundaries(pred))
    tp += len(gb & pb)
    fp += len(pb - gb)
    fn += len(gb - pb)
    n_gold += len(g)
    flat = squash(r["text"])
    for kind, pos in [("false_split", x) for x in pb - gb] + [("missed_split", x) for x in gb - pb]:
        errors.append({"pid": pid, "case_id": r["case_id"], "type": kind,
                       "context": flat[max(0, pos - 40):pos] + " || " + flat[pos:pos + 40]})

P = tp / (tp + fp) if tp + fp else 0.0
R = tp / (tp + fn) if tp + fn else 0.0
F = 2 * P * R / (P + R) if P + R else 0.0
err = (fp + fn) / n_gold if n_gold else 0.0
kinds = Counter(e["type"] for e in errors)

logs = Path(a.logs)
logs.mkdir(parents=True, exist_ok=True)
with open(logs / "splitter_errors.tsv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["pid", "case_id", "type", "context"], delimiter="\t")
    w.writeheader()
    w.writerows(errors)
summary = (f"\n## {dt.date.today()} — sentence splitter validation\n"
           f"- passages scored: {len(gold) - len(edited)} (skipped, text edited: {len(edited)})\n"
           f"- gold sentences: {n_gold}\n"
           f"- boundary precision {P:.3f}, recall {R:.3f}, F1 {F:.3f}\n"
           f"- false splits {kinds['false_split']}, missed splits {kinds['missed_split']}\n"
           f"- **error rate {err:.2%}** ((false + missed) / gold sentences)\n"
           f"- splitter: {'current code (re-run)' if a.rerun_splitter else 'as sampled'}\n")
with open(logs / "week1_log.md", "a", encoding="utf-8") as f:
    f.write(summary)
print(summary)
if edited:
    print("WARNING: these passages had characters changed, not just line breaks, and were skipped:", edited)
print(f"errors with context -> {logs / 'splitter_errors.tsv'}")
sys.exit(0)
