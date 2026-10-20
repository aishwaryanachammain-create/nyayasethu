"""Stages 1-3 behind one object, loaded ONCE (step 29) and shared by the CLI demo and the API.

Loads whatever exists and degrades gracefully, saying so in `components`:
  role labeller   models/stage1/            (else: no query-side role split)
  indexes         models/index_corpus/      (build: python -m stage2_retrieval.build_index --corpus)
  bi-encoder      models/biencoder/ or InLegalBERT (dense search only if the index has FAISS files)
  cross-encoder   models/cross_encoder/     (else: rank by authority prior over the hybrid list)
  prior           Member B pagerank.csv, else citation in-degree
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

from common.config import cfg, path
from common.io import attach_segments, load_judgments
from common.text import tokenize


class Engine:
    def __init__(self, load_dense: bool = True):
        t0 = time.time()
        c = cfg()
        self.components: Dict[str, str] = {}
        try:
            from stage1_roles.label_corpus import Labeller
            self.labeller = Labeller()
            self.components["labeller"] = "models/stage1"
        except FileNotFoundError:
            self.labeller = None
            self.components["labeller"] = "missing"
        idx = path("models") / "index_corpus"
        if not (idx / "sparse.pkl").exists():
            raise FileNotFoundError("No corpus index. Run: python -m stage2_retrieval.build_index --corpus")
        enc = None
        if load_dense and any(idx.glob("dense_*.faiss")):
            from stage2_retrieval.build_index import encoder_path
            from stage2_retrieval.dense import load_encoder
            enc = load_encoder(encoder_path())
            self.components["bi-encoder"] = encoder_path()
        from stage2_retrieval.hybrid import HybridRetriever
        self.retriever = HybridRetriever.load(idx, enc, self.labeller)
        self.components["retrieval"] = "BM25 + dense (RRF)" if self.retriever.dense else "BM25 (RRF over partitions)"
        ce_dir = path("models") / "cross_encoder"
        self.ce = None
        if ce_dir.exists():
            from stage3_rerank.cross_encoder import CrossEncoder
            self.ce = CrossEncoder(str(ce_dir), c["stage3"]["max_len"])
        self.components["cross-encoder"] = "models/cross_encoder" if self.ce else "missing (prior-only rerank)"
        from stage3_rerank.authority import load_prior
        from stage3_rerank.rerank import Reranker
        prior, src = load_prior()
        self.components["prior"] = src
        ap = path("models") / "stage3_alpha.json"
        alpha = json.loads(ap.read_text())["alpha"] if ap.exists() else 0.8
        self.reranker = Reranker(self.ce, prior, alpha, c["stage3"]["top_k"])
        self.docs = {d.case_id: d for d in load_judgments(path("judgments"))}
        if path("segments").exists():
            attach_segments(list(self.docs.values()), path("segments"))
        self.load_seconds = round(time.time() - t0, 2)

    def _best_sentence(self, query: str, case_id: str, roles) -> Optional[Dict]:
        d = self.docs.get(case_id)
        if d is None:
            return None
        q = set(tokenize(query))
        best, bs = None, 0.0
        for s in d.sentences:
            if roles and s.role not in roles:
                continue
            t = set(tokenize(s.text))
            sc = len(q & t) / (len(t) ** 0.5 + 1)
            if sc > bs:
                best, bs = s, sc
        return {"sent_idx": best.idx, "role": best.role, "text": best.text} if best else None

    def search(self, query: str, k: int = 10) -> Dict:
        from stage2_retrieval.query import split_query
        c = cfg()["stage2"]
        t = {}
        t0 = time.time()
        facts, _ = split_query(query, self.labeller)
        t["query_split"] = time.time() - t0
        t0 = time.time()
        fused, runs = self.retriever.search(query, facts=facts)
        t["retrieve"] = time.time() - t0
        t0 = time.time()
        ids = [d for d, _ in fused]
        texts = self.retriever.texts.get("reasoning") or self.retriever.texts["full"]
        top = self.reranker.combine(ids, self.reranker.ce_scores(facts, ids, texts), k=k)
        t["rerank"] = time.time() - t0
        ranks = {n: {d: i + 1 for i, (d, _) in enumerate(r)} for n, r in runs.items()}
        results = []
        for r in top:
            d = self.docs.get(r["case_id"])
            results.append({**r, "score": round(r["score"], 4),
                            "title": d.title if d else "", "court": d.court if d else None,
                            "date": d.date if d else None, "bench_size": d.bench_size if d else None,
                            "surfaced_by": {n: m[r["case_id"]] for n, m in ranks.items() if r["case_id"] in m},
                            "matched_facts": self._best_sentence(facts, r["case_id"], set(c["facts_roles"])),
                            "matched_reasoning": self._best_sentence(query, r["case_id"], set(c["reasoning_roles"]))})
        return {"query": query, "facts": facts, "results": results, "alpha": self.reranker.alpha,
                "timings": {k2: round(v, 3) for k2, v in t.items()}, "components": self.components}
