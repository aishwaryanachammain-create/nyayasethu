"""Steps 13-15 — sentence vectors -> BiLSTM (2 layers, hidden 256) -> linear emissions -> CRF.

The CRF (pytorch-crf) scores whole label sequences, so the model learns role order.
CRF NLL cannot take class weights, so an auxiliary class-weighted cross-entropy on the emissions
handles imbalance:   loss = crf_nll + ce_weight * weighted_ce,   weights = 1/freq, normalised.
"""
from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF


def inverse_freq_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, 1.0 / np.maximum(counts, 1), 0.0)
    w = w / w[w > 0].mean()                      # normalised so the mean present-class weight is 1
    return torch.tensor(w, dtype=torch.float32)


class RoleTagger(nn.Module):
    def __init__(self, in_dim: int, n_tags: int, hidden: int = 256, layers: int = 2, dropout: float = 0.3,
                 class_weights: torch.Tensor | None = None, ce_weight: float = 0.5):
        super().__init__()
        self.inp_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out_drop = nn.Dropout(dropout)
        self.emit = nn.Linear(2 * hidden, n_tags)
        self.crf = CRF(n_tags, batch_first=True)
        self.register_buffer("cw", class_weights if class_weights is not None else torch.ones(n_tags))
        self.ce_weight = ce_weight

    def emissions(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(self.inp_drop(x))
        return self.emit(self.out_drop(h))

    def loss(self, x, y, mask) -> torch.Tensor:
        em = self.emissions(x)
        nll = -self.crf(em, y, mask=mask, reduction="mean")
        ce = F.cross_entropy(em[mask], y[mask], weight=self.cw)
        return nll + self.ce_weight * ce

    @torch.no_grad()
    def decode(self, x, mask) -> tuple[List[List[int]], np.ndarray]:
        em = self.emissions(x)
        paths = self.crf.decode(em, mask=mask)
        probs = em.softmax(-1).cpu().numpy()
        return paths, probs
