"""Directed cited-by graph with sentence-level edge provenance (Stages 3 and 4)."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

import networkx as nx

from common.schema import Judgment


@dataclass
class CitingMention:
    citing_doc: str
    sentence_idx: int
    cited_doc: str


class CitationGraph:
    """Edge citing -> cited. `mentions[cited]` holds every sentence that cites it."""

    def __init__(self):
        self.g = nx.DiGraph()
        self.mentions: Dict[str, List[CitingMention]] = defaultdict(list)

    @classmethod
    def build(cls, docs: Iterable[Judgment]) -> "CitationGraph":
        cg = cls()
        for d in docs:
            cg.g.add_node(d.doc_id, year=d.year, court=d.court, bench=d.bench_size)
        for d in docs:
            for s in d.sentences:
                for cited in s.citations:
                    if cited == d.doc_id or cited not in cg.g:
                        continue
                    if cg.g.has_edge(d.doc_id, cited):
                        cg.g[d.doc_id][cited]["weight"] += 1
                    else:
                        cg.g.add_edge(d.doc_id, cited, weight=1)
                    cg.mentions[cited].append(CitingMention(d.doc_id, s.idx, cited))
        return cg

    def cited_by(self, doc_id: str) -> List[str]:
        return list(self.g.predecessors(doc_id)) if doc_id in self.g else []

    def pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        if self.g.number_of_nodes() == 0:
            return {}
        try:
            return nx.pagerank(self.g, alpha=alpha, weight="weight")
        except Exception:  # pragma: no cover - scipy-free fallback
            return nx.pagerank(self.g, alpha=alpha, weight="weight", max_iter=200, tol=1e-6)

    def authority_prior(self, alpha: float = 0.85) -> Dict[str, float]:
        """PageRank rescaled to [0,1] on a log scale so a few landmark cases do not dominate."""
        pr = self.pagerank(alpha)
        if not pr:
            return {}
        logs = {k: math.log(v) for k, v in pr.items()}
        lo, hi = min(logs.values()), max(logs.values())
        return {k: (v - lo) / (hi - lo) if hi > lo else 0.0 for k, v in logs.items()}

    def stats(self, min_citing: int = 3) -> Dict[str, float]:
        """Week-2 analysis: coverage of Stage 4 = share of precedents with enough citing judgments."""
        n = self.g.number_of_nodes()
        indeg = [d for _, d in self.g.in_degree()]
        cited = [x for x in indeg if x > 0]
        return {
            "nodes": n,
            "edges": self.g.number_of_edges(),
            "mentions": sum(len(v) for v in self.mentions.values()),
            "mean_in_degree": sum(indeg) / n if n else 0.0,
            "max_in_degree": max(indeg) if indeg else 0,
            "share_cited_at_least_once": len(cited) / n if n else 0.0,
            f"share_with_>={min_citing}_citing": sum(1 for x in indeg if x >= min_citing) / n if n else 0.0,
        }
