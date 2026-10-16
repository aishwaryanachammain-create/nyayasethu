"""Overlapping sentence windows. The mean judgment is ~7,650 tokens (14.9x BERT's 512),
so Stage 1 tags windows of sentences and stitches them: each sentence takes the
prediction from the window in which it sits most centrally (most context both sides)."""
from __future__ import annotations

from typing import List, Sequence, Tuple


def make_windows(n: int, size: int, stride: int) -> List[Tuple[int, int]]:
    if n <= size:
        return [(0, n)]
    wins, start = [], 0
    while True:
        end = min(start + size, n)
        wins.append((start, end))
        if end == n:
            break
        start += stride
    return wins


def stitch(n: int, windows: Sequence[Tuple[int, int]], preds: Sequence[Sequence], confs=None):
    """preds[w][k] = label for sentence windows[w][0]+k. Returns (labels, confs) of length n."""
    best_label = [None] * n
    best_conf = [0.0] * n
    best_centrality = [-1.0] * n
    for w, (s, e) in enumerate(windows):
        mid = (s + e - 1) / 2
        half = max((e - s) / 2, 1)
        for k, lab in enumerate(preds[w]):
            i = s + k
            c = 1 - abs(i - mid) / half
            if c > best_centrality[i]:
                best_centrality[i] = c
                best_label[i] = lab
                best_conf[i] = confs[w][k] if confs is not None else 1.0
    return best_label, best_conf
