"""Step 20 — fine-tune the bi-encoder (InLegalBERT + mean pooling) with sentence-transformers
MultipleNegativesRankingLoss on SC-PCR TRAIN queries. Test queries are never used.

Pairs: (query facts, relevant candidate's Reasoning text) with `hard_negatives` BM25-retrieved
non-relevant candidates per pair, so in-batch negatives are joined by hard ones.

    python -m stage2_retrieval.train_biencoder              # -> models/biencoder/
"""
import argparse
import random

from common.config import cfg, path
from eval.benchmarks import load_benchmark
from stage2_retrieval.bm25 import BM25
from stage2_retrieval.dense import load_encoder
from stage2_retrieval.hybrid import partition_texts
from stage2_retrieval.query import split_query


def build_examples(docs, queries, qrels, n_hard: int, labeller=None, seed: int = 13):
    rng = random.Random(seed)
    parts = partition_texts(docs)
    reasoning = dict(zip(*parts["reasoning"]))
    bm = BM25().fit(*parts["full"])
    ex = []
    for qid, text in queries.items():
        rel = [d for d, v in qrels.get(qid, {}).items() if v > 0 and d in reasoning]
        if not rel:
            continue
        facts, _ = split_query(text, labeller)
        hard = [d for d, _ in bm.search(text, 50) if d not in rel][: max(n_hard * 3, 10)]
        for d in rel:
            negs = rng.sample(hard, min(n_hard, len(hard)))
            ex.append([facts[:4000], reasoning[d][:6000]] + [reasoning[n][:6000] for n in negs])
    return ex


def main():
    c = cfg()
    s2 = c["stage2"]
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=c["models"]["encoder"])
    p.add_argument("--bench", default="sc-pcr")
    p.add_argument("--out", default=str(path("models") / "biencoder"))
    p.add_argument("--epochs", type=int, default=s2["biencoder_epochs"])
    p.add_argument("--batch", type=int, default=s2["biencoder_batch"])
    a = p.parse_args()

    from datasets import Dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import MultipleNegativesRankingLoss

    labeller = None
    try:
        from stage1_roles.label_corpus import Labeller
        labeller = Labeller()
    except FileNotFoundError:
        pass
    docs, q, qr = load_benchmark(a.bench, "train")
    ex = build_examples(docs, q, qr, s2["hard_negatives"], labeller)
    if not ex:
        raise SystemExit("No training pairs: build SC-PCR first (python -m stage0_corpus.build_sc_pcr).")
    k = len(ex[0])
    cols = ["anchor", "positive"] + [f"negative_{i}" for i in range(1, k - 1)]
    ex = [e for e in ex if len(e) == k]
    ds = Dataset.from_dict({col: [e[i] for e in ex] for i, col in enumerate(cols)})
    print(f"{len(ex)} training tuples ({k - 2} hard negatives each) from {len(q)} train queries")
    model = load_encoder(a.base)
    args = SentenceTransformerTrainingArguments(output_dir=a.out + "_ckpt", num_train_epochs=a.epochs,
                                                per_device_train_batch_size=a.batch, learning_rate=2e-5,
                                                warmup_ratio=0.1, fp16=_cuda(), save_strategy="no",
                                                logging_steps=50, report_to=[])
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds,
                               loss=MultipleNegativesRankingLoss(model)).train()
    model.save(a.out)
    print(f"-> {a.out}")


def _cuda():
    import torch
    return torch.cuda.is_available()


if __name__ == "__main__":
    main()
