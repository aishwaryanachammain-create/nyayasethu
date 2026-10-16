"""Reciprocal Rank Fusion (Cormack et al., 2009). Parameter-free apart from k=60, and needed
because BM25 scores and cosine similarities have no common scale."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

Ranked = Sequence[Tuple[str, float]]


def rrf(runs: Sequence[Ranked], k: int = 60, weights: Sequence[float] | None = None,
        depth: int = 100) -> List[Tuple[str, float]]:
    weights = weights or [1.0] * len(runs)
    fused: Dict[str, float] = defaultdict(float)
    for run, w in zip(runs, weights):
        for rank, (doc, _) in enumerate(run, start=1):
            fused[doc] += w / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])[:depth]


def linear_interpolation(runs: Sequence[Ranked], weights: Sequence[float], depth: int = 100):
    """Min-max-normalised linear fusion — the 'alternative considered', kept for ablation."""
    fused: Dict[str, float] = defaultdict(float)
    for run, w in zip(runs, weights):
        if not run:
            continue
        vals = [s for _, s in run]
        lo, hi = min(vals), max(vals)
        for doc, s in run:
            fused[doc] += w * ((s - lo) / (hi - lo) if hi > lo else 1.0)
    return sorted(fused.items(), key=lambda x: -x[1])[:depth]
