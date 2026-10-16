"""Step 18 — run the trained labeller over data/judgments.jsonl and write data/segments.parquet
({case_id, sent_idx, text, role, conf}), the Week-4 hand-off to Member B.

    python -m stage1_roles.label_corpus              # gold roles kept where a judgment has them (conf = 1.0)
    python -m stage1_roles.label_corpus --predict-all
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import List

import torch

from common.config import cfg, path
from common.io import load_judgments, save_segments
from common.schema import Judgment
from stage1_roles.features import doc_slices, load_featurizer
from stage1_roles.model import RoleTagger
from stage1_roles.train import predict_docs


class Labeller:
    """Loads models/stage1 once; reused by the API at startup."""

    def __init__(self, model_dir: str | Path | None = None, device: str | None = None):
        c = cfg()
        d = Path(model_dir or path("models") / "stage1")
        ck = torch.load(d / "role_tagger.pt", map_location="cpu")
        self.labels = ck["labels"]
        self.fz = load_featurizer(d / "featurizer.pkl", c)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        s1 = c["stage1"]
        self.model = RoleTagger(ck["in_dim"], len(self.labels), s1["lstm_hidden"], s1["lstm_layers"], s1["dropout"])
        self.model.load_state_dict(ck["state"])
        self.model.to(self.device).eval()
        self.window, self.overlap = s1["window"], s1["overlap"]

    def label(self, docs: List[Judgment], batch_docs: int = 200) -> List[Judgment]:
        docs = [d for d in docs if d.sentences]
        for i in range(0, len(docs), batch_docs):
            chunk = docs[i:i + batch_docs]
            X = self.fz.transform(chunk)
            labs, confs = predict_docs(self.model, X, chunk, doc_slices(chunk), self.window, self.overlap,
                                       self.labels, self.device)
            for d, l, c in zip(chunk, labs, confs):
                for s, r, p in zip(d.sentences, l, c):
                    s.role, s.conf = r, round(p, 4)
        return docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judgments", default=str(path("judgments")))
    p.add_argument("--out", default=str(path("segments")))
    p.add_argument("--predict-all", action="store_true", help="also re-label judgments that have gold roles")
    a = p.parse_args()
    docs = load_judgments(a.judgments)
    gold = {d.case_id for d in docs if d.has_roles} if not a.predict_all else set()
    lab = Labeller()
    todo = [d for d in docs if d.case_id not in gold]
    lab.label(todo)
    for d in docs:
        if d.case_id in gold:
            for s in d.sentences:
                s.conf = 1.0
    n = save_segments(docs, a.out)
    dist = Counter(s.role for d in docs for s in d.sentences)
    print(f"{n} sentences from {len(docs)} judgments -> {a.out}")
    print("role distribution:", dict(dist.most_common()))


if __name__ == "__main__":
    main()
