"""Step 6 (part 1) — draw ~200 sentences for manual splitter validation from RAW judgment text
(SC PDFs / AILA), over-sampling passages with known hard cases.

    python -m stage0_corpus.splitter_sample
-> data/validation/splitter_sample.jsonl  (untouched raw passages)
-> data/validation/splitter_annotate.txt  (splitter output, one sentence per line)
Fix ONLY the line breaks in splitter_annotate.txt (join wrong splits, break missed ones), then run
python -m stage0_corpus.splitter_validate
"""
import argparse
import json
import random
import re

from common.config import cfg, path
from common.io import load_judgments
from stage0_corpus.splitter import split_sentences

HARD = re.compile(r"\bv(s)?\.|\bS\.\s*\d|\bAIR\b|SCC|SCR|\bNo\.|\bi\.e\.|\bviz\.|\b[A-Z]\.\s?[A-Z]\.|\d+\.\d+|\bpara\b|"
                  r"Hon'?ble|\bArt\.|\bLtd\.", re.I)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judgments", default=str(path("judgments")))
    p.add_argument("--target", type=int, default=cfg()["stage0"]["splitter_sample_sentences"])
    p.add_argument("--per-passage", type=int, default=5)
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    rng = random.Random(a.seed)
    # Bhattacharya text is pre-split with periods stripped, so it is not a fair splitter test
    docs = [d for d in load_judgments(a.judgments) if not d.source.startswith("Bhattacharya") and len(d.text) > 1500]
    if not docs:
        raise SystemExit("Need raw judgments (SC PDFs or AILA) in judgments.jsonl.")
    rng.shuffle(docs)
    passages, total, seen = [], 0, set()
    for d in docs * 3:
        if total >= a.target:
            break
        paras = [x for x in re.split(r"\n\s*\n|\n(?=\s*\d{1,3}\.\s)", d.text) if len(x) > 300] or [d.text]
        hard = [x for x in paras if HARD.search(x)]
        para = rng.choice(hard if hard and rng.random() < 0.7 else paras)
        sents = split_sentences(para)
        if len(sents) < 2:
            continue
        st = rng.randrange(0, max(1, len(sents) - a.per_passage + 1))
        raw = re.sub(r"\s+", " ", " ".join(sents[st:st + a.per_passage])).strip()
        if (d.case_id, raw[:80]) in seen:
            continue
        seen.add((d.case_id, raw[:80]))
        pred = split_sentences(raw)
        passages.append({"pid": f"P{len(passages) + 1:03d}", "case_id": d.case_id, "text": raw, "predicted": pred})
        total += len(pred)
    out = path("validation")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "splitter_sample.jsonl", "w", encoding="utf-8") as f:
        for x in passages:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(out / "splitter_annotate.txt", "w", encoding="utf-8") as f:
        f.write("# Fix line breaks only: one true sentence per line. Keep the ### headers.\n")
        for x in passages:
            f.write(f"\n### {x['pid']} {x['case_id']}\n" + "\n".join(x["predicted"]) + "\n")
    print(f"{len(passages)} passages, {total} sentences -> {out / 'splitter_annotate.txt'}")


if __name__ == "__main__":
    main()
