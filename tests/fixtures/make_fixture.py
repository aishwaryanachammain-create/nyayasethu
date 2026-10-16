"""Build a miniature, fully offline workspace from the fictional judgments in this folder:

  <root>/data/raw/bhattacharya2019/semantic-segmentation.zip   (roles mapped to Bhattacharya labels)
  <root>/data/raw/aila2019/AILA_2019_dataset.zip               (11 candidates, 5 queries, qrels)
  <root>/data/judgments.jsonl                                   (as AWS-SC judgments with titles/aliases)
  <root>/models/tiny-bert/                                      (random tiny BERT: encoder + cross-encoder)
  <root>/config.yaml                                            (paths -> <root>, fast training settings)
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).parent
REPO = HERE.parents[1]
TO_BHATT = {"Facts": "Facts", "Issue": "Ratio of the decision", "Arguments of Petitioner": "Argument",
            "Arguments of Respondent": "Argument", "Reasoning": "Ratio of the decision",
            "Decision": "Ruling by Present Court", "None": "Facts"}

QUERIES = {
    "AILA_Q1": "I resigned after many years of service without giving notice. My employer is refusing to pay any gratuity. Can they keep it without proving loss?",
    "AILA_Q2": "I cancelled a flat booking and the builder kept a large part of the price under the allotment letter. Is that a penalty?",
    "AILA_Q3": "We paid earnest money at a public auction for a mining lease and failed to pay the balance. The State forfeited it.",
    "AILA_Q4": "My landlady wants her son to occupy the house after returning from abroad. Is that bona fide need for eviction?",
    "AILA_Q5": "The insurer rejected my fire claim because I informed them late although their surveyor confirmed the loss.",
}
QRELS = {"AILA_Q1": ["D04", "D02"], "AILA_Q2": ["D06", "D02"], "AILA_Q3": ["D05", "D02"],
         "AILA_Q4": ["D10"], "AILA_Q5": ["D09"]}


def parse(p: Path):
    raw = p.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n", raw, re.S)
    meta = dict(l.split(":", 1) for l in m.group(1).splitlines() if ":" in l)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    body = raw[m.end():]
    tagged = re.findall(r"\[([A-Za-z ]+)\]\s*([^\[]+)", body)
    return meta, tagged


def build(root: Path) -> Path:
    root = Path(root)
    (root / "data/raw/bhattacharya2019").mkdir(parents=True, exist_ok=True)
    (root / "data/raw/aila2019").mkdir(parents=True, exist_ok=True)
    files = sorted(HERE.glob("judgments/D*.txt"))
    # Bhattacharya-format zip (sentence<TAB>label, periods kept)
    with zipfile.ZipFile(root / "data/raw/bhattacharya2019/semantic-segmentation.zip", "w") as z:
        cats = []
        for p in files:
            meta, tagged = parse(p)
            lines = []
            for role, chunk in tagged:
                for s in re.split(r"(?<=[.])\s+(?=[A-Z])", chunk.strip()):
                    if s.strip():
                        lines.append(f"{s.strip()}\t{TO_BHATT[role.strip()]}")
            z.writestr(f"semantic-segmentation-master/data/text/{p.stem}.txt", "\n".join(lines))
            cats.append(p.stem)
        z.writestr("semantic-segmentation-master/categories.txt", "Demo\t" + " ".join(cats))
    # AILA-format zip
    with zipfile.ZipFile(root / "data/raw/aila2019/AILA_2019_dataset.zip", "w") as z:
        for p in files:
            meta, tagged = parse(p)
            z.writestr(f"AILA_2019_dataset/Object_casedocs/{p.stem}.txt", " ".join(c.strip() for _, c in tagged))
        z.writestr("AILA_2019_dataset/Query_doc.txt", "\n".join(f"{k}||{v}" for k, v in QUERIES.items()))
        z.writestr("AILA_2019_dataset/relevance_judgments_priorcases.txt",
                   "\n".join(f"{q} Q0 {d} 1" for q, ds in QRELS.items() for d in ds))
    # SC-style judgments.jsonl (what build_corpus writes for AWS-SC)
    rows = []
    for p in files:
        meta, tagged = parse(p)
        text = " ".join(c.strip() for _, c in tagged)
        sents = [s for s in re.split(r"(?<=[.])\s+(?=[A-Z])", text) if s.strip()]
        rows.append({"case_id": f"sc:{p.stem}", "source": "AWS-SC", "court": meta.get("court"),
                     "date": f"{meta['year']}-01-01", "bench_size": int(meta["bench_size"]),
                     "title": meta["title"], "aliases": [a.strip() for a in meta["aliases"].split(";")],
                     "text": text, "sentences": [{"idx": i, "text": s} for i, s in enumerate(sents)]})
    with open(root / "data/judgments.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # tiny random BERT for encoder / cross-encoder plumbing tests
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted({w.lower() for p in files
             for w in re.findall(r"[A-Za-z]+", p.read_text())})[:3000] + [f"##{c}" for c in "abcdefghijklmnopqrstuvwxyz"] + list("abcdefghijklmnopqrstuvwxyz0123456789.,()")
    tb = root / "models/tiny-bert"
    tb.mkdir(parents=True, exist_ok=True)
    (tb / "vocab.txt").write_text("\n".join(dict.fromkeys(vocab)))
    tok = BertTokenizerFast(vocab_file=str(tb / "vocab.txt"))
    tok.save_pretrained(tb)
    cfgb = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                      intermediate_size=64, max_position_embeddings=520, num_labels=1)
    BertForSequenceClassification(cfgb).save_pretrained(tb)
    # config pointing at root, with fast settings
    c = yaml.safe_load((REPO / "config.yaml").read_text())
    for k, v in c["paths"].items():
        c["paths"][k] = str(root / v)
    c["models"]["encoder"] = str(tb)
    c["stage1"].update({"features": "lsa", "epochs": 6, "patience": 3, "lstm_hidden": 32})
    c["stage2"].update({"chunk_tokens": 64, "chunk_stride": 16, "biencoder_batch": 4})
    c["stage3"].update({"max_len": 128, "ce_batch": 4})
    (root / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
    return root / "config.yaml"


if __name__ == "__main__":
    import sys
    print(build(Path(sys.argv[1])))
