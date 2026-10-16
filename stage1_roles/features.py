"""Step 12 — sentence vectors for the role labeller, computed ONCE and cached as .npy.

* `inlegalbert` (the method): frozen law-ai/InLegalBERT, [CLS] vector, max_len 128,
  fp16 + batch 256 on a GPU (Kaggle T4). 768-d.
* `lsa` (offline fallback): TF-IDF (word 1-2 grams, fitted on TRAIN sentences only) -> truncated SVD,
  256-d. Used when the model weights cannot be downloaded; results are reported as such.

Both append one extra feature: the sentence's relative position in the judgment (0..1), because
role order is strongly positional (facts early, ruling last).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Sequence

import numpy as np

from common.schema import Judgment


def _with_position(X: np.ndarray, docs: Sequence[Judgment]) -> np.ndarray:
    pos = np.concatenate([np.linspace(0, 1, max(len(d.sentences), 1))[: len(d.sentences)] for d in docs])
    return np.hstack([X, pos[:, None]]).astype(np.float32)


class LSAFeaturizer:
    name = "lsa"

    def __init__(self, dims: int = 256):
        self.dims = dims

    def fit(self, docs: Sequence[Judgment]) -> "LSAFeaturizer":
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [s.text for d in docs for s in d.sentences]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000, sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        self.svd = TruncatedSVD(n_components=min(self.dims, X.shape[1] - 1), random_state=0).fit(X)
        return self

    def transform(self, docs: Sequence[Judgment]) -> np.ndarray:
        texts = [s.text for d in docs for s in d.sentences]
        X = self.svd.transform(self.vec.transform(texts))
        X /= np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
        return _with_position(X, docs)

    def save(self, path: str | Path):
        with open(path, "wb") as f:
            pickle.dump(self, f)


class InLegalBERTFeaturizer:
    name = "inlegalbert"

    def __init__(self, model_name: str = "law-ai/InLegalBERT", max_len: int = 128, batch: int = 256,
                 device: str | None = None, model=None, tokenizer=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = (model or AutoModel.from_pretrained(model_name)).to(self.device).eval()
        if self.device == "cuda":
            self.model.half()
        self.max_len, self.batch = max_len, batch

    def fit(self, docs):            # frozen encoder: nothing to fit
        return self

    def transform(self, docs: Sequence[Judgment]) -> np.ndarray:
        torch = self.torch
        texts = [s.text for d in docs for s in d.sentences]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch):
                enc = self.tok(texts[i:i + self.batch], truncation=True, max_length=self.max_len, padding=True,
                               return_tensors="pt").to(self.device)
                out.append(self.model(**enc).last_hidden_state[:, 0].float().cpu().numpy())
        X = np.vstack(out) if out else np.zeros((0, self.model.config.hidden_size), np.float32)
        return _with_position(X, docs)

    def save(self, path: str | Path):
        Path(path).write_text(self.name)       # weights are the public checkpoint; nothing to store


def make_featurizer(kind: str, cfg: dict):
    if kind == "inlegalbert":
        s1 = cfg["stage1"]
        return InLegalBERTFeaturizer(cfg["models"]["encoder"], s1["embed_max_len"], s1["embed_batch"])
    if kind == "lsa":
        return LSAFeaturizer()
    raise ValueError(kind)


def load_featurizer(path: str | Path, cfg: dict):
    p = Path(path)
    if p.read_bytes()[:11] == b"inlegalbert":
        return make_featurizer("inlegalbert", cfg)
    with open(p, "rb") as f:
        return pickle.load(f)


def doc_slices(docs: Sequence[Judgment]) -> List[slice]:
    out, o = [], 0
    for d in docs:
        out.append(slice(o, o + len(d.sentences)))
        o += len(d.sentences)
    return out
