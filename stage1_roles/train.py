"""Steps 14-17 — train and evaluate the role labeller.

    python -m stage1_roles.train                         # 5-fold cross-validation + final model (features from config)
    python -m stage1_roles.train --features lsa          # offline CPU features
    python -m stage1_roles.train --annotated data/processed/roles_gold.jsonl   # add hand-annotated judgments

Data: Bhattacharya 2019 (50 judgments) + any hand-annotated judgments. With only ~50-70 documents a
single split is noisy, so we report k-fold CV (document-level, stratified by legal domain):
pooled per-class P/R/F1 over all test folds, macro-F1 mean +/- std across folds, and the
majority-class baseline measured on the same data. Early stopping on validation macro-F1.
The final model (trained on all documents, early-stopped on a held-out slice) goes to models/stage1/.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from common.config import cfg, path
from common.io import load_judgments
from common.schema import Judgment
from eval.classification import acceptance_gate_classes, classification_report, format_report
from stage0_corpus.loaders import load_bhattacharya
from stage1_roles.features import doc_slices, make_featurizer
from stage1_roles.model import RoleTagger, inverse_freq_weights
from stage1_roles.windows import make_windows, stitch


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def load_training_docs(annotated: str | None) -> List[Judgment]:
    docs = load_bhattacharya(path("raw") / "bhattacharya2019" / "semantic-segmentation.zip")
    if annotated and Path(annotated).exists():
        docs += load_judgments(annotated)
    return [d for d in docs if d.sentences and d.has_roles]


def _windows(X, y, docs, sl, size, overlap):
    items = []
    for d, s in zip(docs, sl):
        for a, b in make_windows(s.stop - s.start, size, size - overlap):
            items.append((X[s.start + a: s.start + b], None if y is None else y[s.start + a: s.start + b]))
    return items


def _pad(items, device):
    L = max(len(x) for x, _ in items)
    D = items[0][0].shape[1]
    xb = torch.zeros(len(items), L, D)
    yb = torch.zeros(len(items), L, dtype=torch.long)
    mb = torch.zeros(len(items), L, dtype=torch.bool)
    for i, (x, y) in enumerate(items):
        xb[i, :len(x)] = torch.from_numpy(x)
        mb[i, :len(x)] = True
        if y is not None:
            yb[i, :len(y)] = torch.from_numpy(y)
    return xb.to(device), yb.to(device), mb.to(device)


def predict_docs(model, X, docs, sl, size, overlap, labels, device):
    model.eval()
    out_l, out_c = [], []
    for d, s in zip(docs, sl):
        n = s.stop - s.start
        wins = make_windows(n, size, size - overlap)
        xb, _, mb = _pad([(X[s.start + a: s.start + b], None) for a, b in wins], device)
        paths, probs = model.decode(xb, mb)
        preds = [[labels[k] for k in p] for p in paths]
        confs = [[float(probs[w, i, k]) for i, k in enumerate(p)] for w, p in enumerate(paths)]
        lab, conf = stitch(n, wins, preds, confs)
        out_l.append(lab)
        out_c.append(conf)
    return out_l, out_c


def fit_one(Xtr, ytr, trd, Xva, yva, vad, labels, c, device, log=print):
    s1 = c["stage1"]
    L2I = {l: i for i, l in enumerate(labels)}
    sl_tr, sl_va = doc_slices(trd), doc_slices(vad)
    items = _windows(Xtr, ytr, trd, sl_tr, s1["window"], s1["overlap"])
    model = RoleTagger(Xtr.shape[1], len(labels), s1["lstm_hidden"], s1["lstm_layers"], s1["dropout"],
                       inverse_freq_weights(ytr, len(labels)), s1["ce_weight"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=s1["lr"], weight_decay=1e-4)
    best, best_state, bad = -1.0, None, 0
    gold_va = [l for d in vad for l in (s.role for s in d.sentences)]
    for ep in range(s1["epochs"]):
        model.train()
        random.shuffle(items)
        for i in range(0, len(items), 8):
            xb, yb, mb = _pad(items[i:i + 8], device)
            loss = model.loss(xb, yb, mb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        pl, _ = predict_docs(model, Xva, vad, sl_va, s1["window"], s1["overlap"], labels, device)
        f1 = classification_report(gold_va, [x for p in pl for x in p], labels)["macro_f1"]
        if f1 > best + 1e-4:
            best, bad = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= s1["patience"]:
                break
    model.load_state_dict(best_state)
    log(f"    stopped at epoch {ep + 1}, best val macro-F1 {best:.4f}")
    return model, best


def featurize(kind, c, trd, others):
    fz = make_featurizer(kind, c).fit(trd)
    return fz, fz.transform(trd), [fz.transform(o) for o in others]


def labels_of(docs, L2I):
    return np.array([L2I[s.role] for d in docs for s in d.sentences], dtype=np.int64)


def stratified_folds(docs: Sequence[Judgment], k: int, seed: int) -> List[List[int]]:
    by = {}
    for i, d in enumerate(docs):
        by.setdefault(d.source, []).append(i)
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    j = 0
    for key in sorted(by):
        idx = by[key]
        rng.shuffle(idx)
        for i in idx:
            folds[j % k].append(i)
            j += 1
    return folds


def main():
    c = cfg()
    s1 = c["stage1"]
    p = argparse.ArgumentParser()
    p.add_argument("--features", default=s1["features"], choices=["inlegalbert", "lsa"])
    p.add_argument("--annotated", default=str(path("processed") / "roles_gold.jsonl"))
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--no-final", action="store_true")
    a = p.parse_args()
    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    docs = load_training_docs(a.annotated)
    present = {s.role for d in docs for s in d.sentences}
    labels = [l for l in s1["labels"] if l in present] + sorted(present - set(s1["labels"]))
    L2I = {l: i for i, l in enumerate(labels)}
    print(f"{len(docs)} judgments, {sum(len(d.sentences) for d in docs)} sentences; labels: {labels}")
    print(f"features: {a.features}; device: {device}")
    missing = [l for l in s1["labels"] if l not in present]
    if missing:
        print(f"NOTE: no training examples for {missing} — annotate judgments (stage1_roles.annotate) to add them.")

    t0 = time.time()
    folds = stratified_folds(docs, a.cv, a.seed)
    gold_all, pred_all, fold_f1 = [], [], []
    for f, test_idx in enumerate(folds):
        rest = [i for i in range(len(docs)) if i not in set(test_idx)]
        random.Random(a.seed + f).shuffle(rest)
        nva = max(2, len(rest) // 8)
        va_idx, tr_idx = rest[:nva], rest[nva:]
        trd, vad, ted = [docs[i] for i in tr_idx], [docs[i] for i in va_idx], [docs[i] for i in test_idx]
        _, Xtr, (Xva, Xte) = featurize(a.features, c, trd, [vad, ted])
        print(f"  fold {f + 1}/{a.cv}: train {len(trd)} / val {len(vad)} / test {len(ted)} judgments")
        model, _ = fit_one(Xtr, labels_of(trd, L2I), trd, Xva, None, vad, labels, c, device)
        pl, _ = predict_docs(model, Xte, ted, doc_slices(ted), s1["window"], s1["overlap"], labels, device)
        g = [s.role for d in ted for s in d.sentences]
        pr = [x for q in pl for x in q]
        fold_f1.append(classification_report(g, pr, labels)["macro_f1"])
        gold_all += g
        pred_all += pr

    rep = classification_report(gold_all, pred_all, labels)
    rep["macro_f1_folds"] = fold_f1
    rep["macro_f1_mean"] = float(np.mean(fold_f1))
    rep["macro_f1_std"] = float(np.std(fold_f1))
    rep["gate"] = acceptance_gate_classes(rep, s1["gate_classes"], s1["gate_recall"])
    rep.update({"features": a.features, "cv_folds": a.cv, "n_docs": len(docs), "labels": labels,
                "seconds": round(time.time() - t0, 1)})
    print("\n" + format_report(rep))
    print(f"macro-F1 across folds: {rep['macro_f1_mean']:.4f} ± {rep['macro_f1_std']:.4f}")
    print("gate:", json.dumps(rep["gate"]))
    path("logs").mkdir(parents=True, exist_ok=True)
    (path("logs") / f"stage1_cv_{a.features}.json").write_text(json.dumps(rep, indent=2))

    if not a.no_final:
        rng = random.Random(a.seed)
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        nva = max(2, len(docs) // 10)
        vad, trd = [docs[i] for i in idx[:nva]], [docs[i] for i in idx[nva:]]
        fz, Xtr, (Xva,) = featurize(a.features, c, trd, [vad])
        print("final model:")
        model, best = fit_one(Xtr, labels_of(trd, L2I), trd, Xva, None, vad, labels, c, device)
        out = path("models") / "stage1"
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state": model.state_dict(), "in_dim": Xtr.shape[1], "labels": labels,
                    "features": a.features, "val_macro_f1": best}, out / "role_tagger.pt")
        fz.save(out / "featurizer.pkl")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
