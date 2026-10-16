"""Benchmark access shared by Stages 2-3 and the ablation.

    aila      AILA 2019: 50 queries over 2,914 candidates (CC BY 4.0) — comparable to published work.
              splits: dev = Q1-Q20 (tuning only), test = Q21-Q50, all = Q1-Q50
    sc-pcr    SC-PCR built by stage0_corpus/build_sc_pcr.py (train / dev / test by year)

Candidates are taken from data/judgments.jsonl so they carry Stage-1 roles (segments.parquet).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from common.config import path
from common.io import attach_segments, load_judgments
from common.schema import Judgment, Sentence
from stage0_corpus.loaders import load_aila, load_pcr
from stage0_corpus.splitter import split_sentences

Benchmark = Tuple[List[Judgment], Dict[str, str], Dict[str, Dict[str, int]]]


def _with_roles(cands: List[Judgment]) -> List[Judgment]:
    """Replace benchmark candidates by their judgments.jsonl version (sentences + roles) when available."""
    jp, sp = path("judgments"), path("segments")
    if not jp.exists():
        for d in cands:
            d.sentences = d.sentences or [Sentence(i, s) for i, s in enumerate(split_sentences(d.text))]
        return cands
    corpus = {d.case_id: d for d in load_judgments(jp)}
    if sp.exists():
        attach_segments(list(corpus.values()), sp)
    out = []
    for d in cands:
        j = corpus.get(d.case_id)
        if j is None:
            d.sentences = d.sentences or [Sentence(i, s) for i, s in enumerate(split_sentences(d.text))]
            j = d
        out.append(j)
    return out


def load_benchmark(name: str, split: str = "test") -> Benchmark:
    if name == "aila":
        docs, q, qr = load_aila(path("raw") / "aila2019" / "AILA_2019_dataset.zip")
        ids = sorted(q, key=lambda x: int(x.split("Q")[-1]) if x.split("Q")[-1].isdigit() else x)
        # AILA has no official train split: dev = first 20 queries (for tuning), test = last 30, all = 50
        keep = {"dev": ids[:20], "train": ids[:20], "test": ids[20:], "all": ids}[split]
        q = {k: q[k] for k in keep}
        return _with_roles(docs), q, {k: v for k, v in qr.items() if k in q}
    if name == "sc-pcr":
        docs, q, qr = load_pcr(path("processed") / "sc-pcr", split)
        return _with_roles(docs), q, qr
    raise ValueError(f"unknown benchmark {name!r} (aila | sc-pcr)")
