"""Steps 19, 22, 23 — role-partitioned indexes and the hybrid retriever.

Per document: Facts text (facts_roles) and Reasoning text (reasoning_roles) from Stage-1 roles,
giving two BM25 indexes and two dense indexes. A document with no labelled text in a partition
falls back to its full text so it is never unreachable.

Routing:   query facts -> Facts index (BM25 + dense);   full query -> Reasoning index (BM25 + dense)
Fusion:    RRF over the 4 ranked lists, score = sum 1/(60 + rank), keep top 100.
Ablation switches (step 33): use_dense, use_roles.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from common.config import cfg
from common.schema import Judgment
from stage2_retrieval.bm25 import BM25
from stage2_retrieval.fusion import rrf
from stage2_retrieval.query import split_query


def partition_texts(docs: Sequence[Judgment]) -> Dict[str, Tuple[List[str], List[str]]]:
    c = cfg()["stage2"]
    parts = {"facts": set(c["facts_roles"]), "reasoning": set(c["reasoning_roles"])}
    out = {}
    for name, roles in parts.items():
        ids, texts = [], []
        for d in docs:
            t = " ".join(s.text for s in d.sentences if s.role in roles)
            ids.append(d.case_id)
            texts.append(t if t.strip() else d.text)
        out[name] = (ids, texts)
    out["full"] = ([d.case_id for d in docs], [d.text for d in docs])
    return out


class HybridRetriever:
    def __init__(self, encoder=None, use_dense: bool = True, use_roles: bool = True, labeller=None):
        self.c = cfg()["stage2"]
        self.encoder, self.use_dense, self.use_roles, self.labeller = encoder, use_dense and encoder is not None, use_roles, labeller
        self.bm25: Dict[str, BM25] = {}
        self.dense: Dict[str, object] = {}
        self.texts: Dict[str, Dict[str, str]] = {}

    def build(self, docs: Sequence[Judgment]) -> "HybridRetriever":
        from stage2_retrieval.dense import DenseIndex
        parts = partition_texts(docs)
        names = ["facts", "reasoning"] if self.use_roles else ["full"]
        for n in names:
            ids, texts = parts[n]
            self.bm25[n] = BM25(self.c["bm25_k1"], self.c["bm25_b"]).fit(ids, texts)
            self.texts[n] = dict(zip(ids, texts))
            if self.use_dense:
                self.dense[n] = DenseIndex(self.encoder, self.c["chunk_tokens"], self.c["chunk_stride"]).build(ids, texts)
        return self

    def runs(self, query: str, facts: Optional[str] = None) -> Dict[str, List[Tuple[str, float]]]:
        k = self.c["depth"]
        if facts is None:
            facts, _ = split_query(query, self.labeller)
        plan = [("facts", facts), ("reasoning", query)] if self.use_roles else [("full", query)]
        out = {}
        for part, text in plan:
            out[f"bm25:{part}"] = self.bm25[part].search(text, k)
            if part in self.dense:
                out[f"dense:{part}"] = self.dense[part].search(text, k)
        return out

    def search(self, query: str, facts: Optional[str] = None, k: Optional[int] = None):
        runs = self.runs(query, facts)
        fused = rrf(list(runs.values()), k=self.c["rrf_k"], depth=k or self.c["depth"])
        return fused, runs

    # ---- persistence: BM25 + texts pickled, FAISS written natively
    def save(self, d: str | Path):
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "sparse.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "texts": self.texts, "use_roles": self.use_roles}, f)
        for n, idx in self.dense.items():
            idx.save(d / f"dense_{n}")

    @classmethod
    def load(cls, d: str | Path, encoder=None, labeller=None) -> "HybridRetriever":
        from stage2_retrieval.dense import DenseIndex
        d = Path(d)
        with open(d / "sparse.pkl", "rb") as f:
            st = pickle.load(f)
        r = cls(encoder, use_dense=encoder is not None, use_roles=st["use_roles"], labeller=labeller)
        r.bm25, r.texts = st["bm25"], st["texts"]
        if encoder is not None:
            for n in r.bm25:
                if Path(str(d / f"dense_{n}") + ".faiss").exists():
                    r.dense[n] = DenseIndex(encoder, r.c["chunk_tokens"], r.c["chunk_stride"]).load(d / f"dense_{n}")
        return r
