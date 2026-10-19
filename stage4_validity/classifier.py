"""InLegalBERT treatment classifier: (citing sentence with [[citation]], +-2 context) -> followed /
distinguished / overruled / neutral. Class-weighted cross-entropy (weights ~ 1/frequency), because
overruled is rare. Trained on editorial weak labels; gold pairs (eval/gold_to_label.csv) are excluded.

    python -m stage4_validity.classifier train      # -> models/validity_clf/, logs/stage4_train.json
    python -m stage4_validity.classifier predict    # -> data/from_b/validity_preds.parquet (every mention)

On a Mac this runs on MPS; on Kaggle it uses CUDA with fp16.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from typing import List, Sequence

import numpy as np
import pandas as pd
import torch

from common.config import cfg, path

from .dataset import out_paths
from .export_gold import gold_pairs
from .labels import LABELS, TREATMENTS


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels=TREATMENTS) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def around_citation(s: str, chars: int = 600) -> str:
    """Long sentences (block quotes, lists): keep a window centred on the [[citation]]."""
    if len(s) <= chars:
        return s
    i = max(s.find("[["), 0)
    a = max(0, min(i - chars // 2, len(s) - chars))
    return s[a:a + chars]


class TreatmentClassifier:
    def __init__(self, name_or_dir: str, max_len: int = 256, dev: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = dev or device()
        self.tok = AutoTokenizer.from_pretrained(name_or_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name_or_dir, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
            label2id={l: i for i, l in enumerate(LABELS)}).to(self.device)
        self.max_len = max_len

    def _enc(self, sents: Sequence[str], ctxs: Sequence[str]):
        sents = [around_citation(s) for s in sents]
        return self.tok(list(sents), list(ctxs), truncation="longest_first", max_length=self.max_len,
                        padding=True, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def predict_proba(self, sents: Sequence[str], ctxs: Sequence[str], batch: int = 64) -> np.ndarray:
        self.model.eval()
        out = []
        for i in range(0, len(sents), batch):
            with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                logits = self.model(**self._enc(sents[i:i + batch], ctxs[i:i + batch])).logits
            out.append(torch.softmax(logits.float(), -1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(LABELS)))

    def fit(self, rows: List[tuple], dev_rows: List[tuple], epochs: int, batch: int, lr: float, out_dir: str):
        counts = np.bincount([LABELS.index(r[2]) for r in rows], minlength=len(LABELS)).astype(float)
        w = counts.sum() / np.maximum(counts, 1) / len(LABELS)
        lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=self.device))
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = max(1, (len(rows) + batch - 1) // batch * epochs)
        warm = max(1, steps // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(1, steps - warm))))
        scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")
        hist, best, step, t0 = [], -1.0, 0, time.time()
        for ep in range(epochs):
            random.shuffle(rows)
            self.model.train()
            for i in range(0, len(rows), batch):
                b = rows[i:i + batch]
                enc = self._enc([x[0] for x in b], [x[1] for x in b])
                y = torch.tensor([LABELS.index(x[2]) for x in b], device=self.device)
                with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                    loss = lossf(self.model(**enc).logits.float(), y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if step % 100 == 0:
                    print(f"  step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
            p = self.predict_proba([r[0] for r in dev_rows], [r[1] for r in dev_rows])
            pred = [LABELS[i] for i in p.argmax(1)]
            gold = [r[2] for r in dev_rows]
            h = {"epoch": ep + 1, "dev_macro_f1_3class": macro_f1(gold, pred),
                 "dev_macro_f1_4class": macro_f1(gold, pred, LABELS), "seconds": round(time.time() - t0)}
            print(" ", h, flush=True)
            hist.append(h)
            if h["dev_macro_f1_3class"] > best:
                best = h["dev_macro_f1_3class"]
                self.model.save_pretrained(out_dir)
                self.tok.save_pretrained(out_dir)
        return hist, w.tolist()


def load_rows(sources=("editorial",), neutral_cap: int | None = None, seed: int = 13):
    ed = pd.read_parquet(out_paths()[0])
    ed = ed[ed.weak_label.isin(LABELS) & ed.weak_source.isin(sources) & ed.sentence.notna()]
    gp = gold_pairs()
    n0 = len(ed)
    ed = ed[[(a, b) not in gp for a, b in zip(ed.citing_id, ed.cited_id)]]
    excluded = n0 - len(ed)
    split = {}
    for s in ("train", "dev", "test"):
        d = ed[ed.split == s]
        if s == "train" and neutral_cap:
            neu = d[d.weak_label == "neutral"]
            d = pd.concat([d[d.weak_label != "neutral"], neu.sample(min(len(neu), neutral_cap), random_state=seed)])
        split[s] = list(zip(d.sentence, d.context, d.weak_label, d.mention_id))
    return split, excluded


def train(a):
    s4 = cfg()["stage4"]
    random.seed(13)
    torch.manual_seed(13)
    split, excluded = load_rows(tuple(a.sources.split(",")), s4["neutral_cap"])
    tr, dv = split["train"], split["dev"]
    if a.limit:
        tr, dv = tr[:a.limit], dv[:max(a.limit // 5, 8)]
    print(f"train {len(tr)}, dev {len(dv)}, gold-pair rows excluded {excluded}; device {device()}", flush=True)
    clf = TreatmentClassifier(a.base, s4["max_len"])
    out = str(path("models") / "validity_clf")
    hist, w = clf.fit(tr, dv, a.epochs or s4["epochs"], s4["batch"], s4["lr"], out)
    from collections import Counter
    log = {"base": a.base, "device": clf.device, "sources": a.sources, "n_train": len(tr), "n_dev": len(dv),
           "train_by_class": dict(Counter(r[2] for r in tr)), "class_weights": dict(zip(LABELS, w)),
           "gold_pair_rows_excluded": excluded, "history": hist,
           "note": "dev scores are agreement with weak (editorial) labels, not gold"}
    if not a.limit:
        (path("logs") / "stage4_train.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


def predict(a):
    s4 = cfg()["stage4"]
    mp = out_paths()[0]
    ed = pd.read_parquet(mp)
    ed = ed[ed.sentence.notna()]
    clf = TreatmentClassifier(a.model, s4["max_len"])
    t0 = time.time()
    p = clf.predict_proba(ed.sentence.tolist(), ed.context.fillna("").tolist(), batch=a.batch)
    out = ed[["mention_id", "citing_id", "cited_id", "citing_sent_idx", "sentence", "editorial", "cue", "split"]].copy()
    for i, l in enumerate(LABELS):
        out[f"p_{l}"] = p[:, i]
    out["pred"] = [LABELS[i] for i in p.argmax(1)]
    out["conf"] = p.max(1)
    dst = mp.parent / "validity_preds.parquet"
    out.to_parquet(dst, index=False)
    print(f"{len(out)} mentions in {time.time() - t0:.0f}s -> {dst}")
    print(out.pred.value_counts().to_string())


def main():
    c = cfg()
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train", "predict"])
    p.add_argument("--base", default=c["models"]["encoder"])
    p.add_argument("--model", default=str(path("models") / "validity_clf"))
    p.add_argument("--sources", default="editorial")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, help="smoke test on a few rows (no log written)")
    a = p.parse_args()
    train(a) if a.cmd == "train" else predict(a)


if __name__ == "__main__":
    main()
