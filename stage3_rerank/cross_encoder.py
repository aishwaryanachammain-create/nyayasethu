"""Step 25 — InLegalBERT cross-encoder over (query facts, candidate Reasoning) pairs, truncated to 512.
Positives = relevant candidates; negatives = top BM25 non-relevant candidates (hard negatives).
Trained on SC-PCR TRAIN queries only.

    python -m stage3_rerank.cross_encoder train               # -> models/cross_encoder/
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Sequence

import torch

from common.config import cfg, path


class CrossEncoder:
    def __init__(self, name_or_dir: str, max_len: int = 512, device: str | None = None, model=None, tokenizer=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = tokenizer or AutoTokenizer.from_pretrained(name_or_dir)
        self.model = (model or AutoModelForSequenceClassification.from_pretrained(name_or_dir, num_labels=1)).to(self.device)
        self.max_len = max_len

    def _enc(self, a: Sequence[str], b: Sequence[str]):
        return self.tok(list(a), list(b), truncation="longest_first", max_length=self.max_len, padding=True,
                        return_tensors="pt").to(self.device)

    @torch.no_grad()
    def score(self, query_facts: str, cand_texts: Sequence[str], batch: int = 16) -> List[float]:
        self.model.eval()
        out = []
        for i in range(0, len(cand_texts), batch):
            c = cand_texts[i:i + batch]
            out += self.model(**self._enc([query_facts] * len(c), c)).logits.squeeze(-1).float().tolist()
        return out

    def fit(self, pairs, epochs: int = 1, batch: int = 8, lr: float = 2e-5, log_every: int = 100):
        lossf = torch.nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = max(1, (len(pairs) // batch) * epochs)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)))
        scaler = torch.amp.GradScaler("cuda", enabled=self.device == "cuda")
        self.model.train()
        step = 0
        for _ in range(epochs):
            random.shuffle(pairs)
            for i in range(0, len(pairs), batch):
                b = pairs[i:i + batch]
                enc = self._enc([x[0] for x in b], [x[1] for x in b])
                y = torch.tensor([x[2] for x in b], dtype=torch.float32, device=self.device)
                with torch.autocast(device_type="cuda", enabled=self.device == "cuda"):
                    loss = lossf(self.model(**enc).logits.squeeze(-1).float(), y)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                step += 1
                if step % log_every == 0:
                    print(f"  step {step}/{steps} loss {loss.item():.4f}")
        return self

    def save(self, d: str | Path):
        self.model.save_pretrained(d)
        self.tok.save_pretrained(d)


def training_pairs(docs, queries, qrels, retriever, neg_per_pos: int, labeller=None):
    from stage2_retrieval.query import split_query
    reasoning = retriever.texts.get("reasoning") or retriever.texts["full"]
    pairs = []
    for qid, text in queries.items():
        rel = {d for d, v in qrels.get(qid, {}).items() if v > 0 and d in reasoning}
        if not rel:
            continue
        facts, _ = split_query(text, labeller)
        bm = retriever.bm25.get("reasoning") or retriever.bm25["full"]
        hard = [d for d, _ in bm.search(text, 50) if d not in rel][: neg_per_pos * len(rel)]
        pairs += [(facts, reasoning[d], 1.0) for d in rel] + [(facts, reasoning[d], 0.0) for d in hard]
    return pairs


def main():
    c = cfg()
    s3 = c["stage3"]
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["train"])
    p.add_argument("--base", default=c["models"]["encoder"])
    p.add_argument("--bench", default="sc-pcr")
    p.add_argument("--out", default=str(path("models") / "cross_encoder"))
    a = p.parse_args()
    from eval.benchmarks import load_benchmark
    from stage2_retrieval.hybrid import HybridRetriever
    labeller = None
    try:
        from stage1_roles.label_corpus import Labeller
        labeller = Labeller()
    except FileNotFoundError:
        pass
    docs, q, qr = load_benchmark(a.bench, "train")
    r = HybridRetriever(None, use_dense=False).build(docs)
    pairs = training_pairs(docs, q, qr, r, s3["negatives_per_positive"], labeller)
    if not pairs:
        raise SystemExit("No training pairs (build SC-PCR first).")
    print(f"{len(pairs)} pairs ({sum(x[2] for x in pairs):.0f} positive) from {len(q)} train queries")
    ce = CrossEncoder(a.base, s3["max_len"]).fit(pairs, s3["ce_epochs"], s3["ce_batch"])
    ce.save(a.out)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
