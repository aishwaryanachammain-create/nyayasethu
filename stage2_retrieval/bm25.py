"""Okapi BM25 implemented directly (term weighting from the syllabus: IDF x saturated TF with
document-length normalisation). Kept as a strong sparse baseline — it catches exact statute and
citation matches ('section_74') that dense models blur."""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Callable, Dict, List, Sequence, Tuple

from common.text import tokenize


class BM25:
    def __init__(self, k1: float = 1.2, b: float = 0.75, tokenizer: Callable[[str], List[str]] = tokenize):
        self.k1, self.b, self.tok = k1, b, tokenizer
        self.ids: List[str] = []
        self.doc_len: List[int] = []
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.idf: Dict[str, float] = {}
        self.avgdl = 0.0

    def fit(self, ids: Sequence[str], texts: Sequence[str]) -> "BM25":
        self.ids = list(ids)
        df = Counter()
        for i, t in enumerate(texts):
            tf = Counter(self.tok(t))
            self.doc_len.append(sum(tf.values()))
            for term, c in tf.items():
                self.postings[term].append((i, c))
                df[term] += 1
        N = len(self.ids)
        self.avgdl = sum(self.doc_len) / N if N else 0.0
        # Robertson-Sparck Jones IDF with +1 inside the log to keep weights non-negative
        self.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        return self

    def score(self, query: str) -> Dict[str, float]:
        scores: Dict[int, float] = defaultdict(float)
        for term, qtf in Counter(self.tok(query)).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1))
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return {self.ids[i]: s for i, s in scores.items()}

    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        s = self.score(query)
        return sorted(s.items(), key=lambda x: -x[1])[:k]


class TfidfCosine:
    """TF-IDF cosine — the 'alternative considered' sparse model, kept for the ablation table."""

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None, sublinear_tf=True)

    def fit(self, ids, texts):
        self.ids = list(ids)
        self.X = self.vec.fit_transform(texts)
        return self

    def search(self, query: str, k: int = 100):
        q = self.vec.transform([query])
        s = (self.X @ q.T).toarray().ravel()
        order = s.argsort()[::-1][:k]
        return [(self.ids[i], float(s[i])) for i in order if s[i] > 0]
