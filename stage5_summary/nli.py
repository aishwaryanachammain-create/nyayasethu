"""Faithfulness filter: drop generated sentences the source does not entail.

For each generated sentence (A's legal sentence splitter), the premise is the `nli_premises` (3) source
sentences most similar to it (TF-IDF cosine), joined. A DeBERTa-v3 NLI model
(MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) gives P(entailment); the sentence is kept if
P > stage5.nli_threshold (0.5), otherwise it goes to `rejected` with its score.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from common.config import cfg
from stage0_corpus.splitter import split_sentences


def top_premise(hyp: str, source: Sequence[str], k: int) -> str:
    if not source:
        return ""
    from sklearn.feature_extraction.text import TfidfVectorizer
    try:
        m = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(list(source) + [hyp])
    except ValueError:
        return " ".join(source[:k])
    sim = (m[:-1] @ m[-1].T).toarray().ravel()
    idx = sorted(sorted(range(len(source)), key=lambda i: -sim[i])[:k])
    return " ".join(source[i] for i in idx)


class NLIScorer:
    """P(entailment | premise, hypothesis) with a Hugging Face NLI model."""

    def __init__(self, model: Optional[str] = None, dev: Optional[str] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from .generate import device
        self.name = model or cfg()["models"]["nli"]
        self.device = dev or device()
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name).to(self.device).eval()
        lab = {v.lower(): int(k) for k, v in self.model.config.id2label.items()}
        self.entail = lab["entailment"]
        self.torch = torch

    def __call__(self, pairs: List[Tuple[str, str]], batch: int = 16) -> List[float]:
        out = []
        for i in range(0, len(pairs), batch):
            b = pairs[i:i + batch]
            enc = self.tok([p for p, _ in b], [h for _, h in b], truncation="only_first", max_length=512,
                           padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                p = self.torch.softmax(self.model(**enc).logits.float(), -1)[:, self.entail]
            out += p.cpu().tolist()
        return out


def filter_text(text: str, source: Sequence[str], scorer: Callable[[List[Tuple[str, str]]], List[float]],
                threshold: Optional[float] = None, k: Optional[int] = None) -> Tuple[str, List[Dict]]:
    """-> (kept text, [{sentence, p_entail, premise}] rejected)."""
    s5 = cfg()["stage5"]
    threshold = s5["nli_threshold"] if threshold is None else threshold
    k = k or s5["nli_premises"]
    hyps = split_sentences(text) if text.strip() else []
    if not hyps:
        return "", []
    premises = [top_premise(h, source, k) for h in hyps]
    probs = scorer(list(zip(premises, hyps)))
    kept, rej = [], []
    for h, pr, p in zip(hyps, premises, probs):
        if p > threshold:
            kept.append(h)
        else:
            rej.append({"sentence": h, "p_entail": round(float(p), 4), "premise": pr})
    return " ".join(kept), rej
