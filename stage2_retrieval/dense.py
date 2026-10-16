"""Dense encoder + FAISS (steps 20-21).

Documents are cut into 512-token chunks with 128-token stride, every chunk is embedded, and the
chunk vectors are MAX-POOLED into one vector per document (per partition). Vectors are
L2-normalised and stored in a FAISS IndexFlatIP (inner product = cosine).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


def load_encoder(name_or_dir: str, device: Optional[str] = None):
    """A sentence-transformers model. A plain HF checkpoint (e.g. law-ai/InLegalBERT) is wrapped with
    mean pooling; a fine-tuned directory from train_biencoder.py loads directly."""
    from sentence_transformers import SentenceTransformer, models
    p = Path(name_or_dir)
    if p.exists() and (p / "modules.json").exists():
        return SentenceTransformer(str(p), device=device)
    word = models.Transformer(name_or_dir, max_seq_length=512)
    pool = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    return SentenceTransformer(modules=[word, pool], device=device)


def chunk_text(text: str, tokenizer, size: int = 512, stride: int = 128) -> List[str]:
    ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    body = size - 2
    step = body - stride
    if len(ids) <= body:
        return [text]
    return [tokenizer.decode(ids[i:i + body]) for i in range(0, len(ids) - stride, step)]


class DenseIndex:
    def __init__(self, encoder, size: int = 512, stride: int = 128, batch: int = 32):
        self.enc, self.size, self.stride, self.batch = encoder, size, stride, batch
        self.ids: List[str] = []
        self.index = None

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        tok = self.enc.tokenizer
        chunks, owner = [], []
        for i, t in enumerate(texts):
            for c in chunk_text(t or " ", tok, self.size, self.stride):
                chunks.append(c)
                owner.append(i)
        E = self.enc.encode(chunks, batch_size=self.batch, convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=len(chunks) > 2000)
        out = np.full((len(texts), E.shape[1]), -np.inf, dtype=np.float32)
        for v, o in zip(E, owner):
            out[o] = np.maximum(out[o], v)                     # max-pool chunk vectors per document
        out /= np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)
        return out

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> "DenseIndex":
        import faiss
        V = self.encode_docs(texts)
        self.ids = list(ids)
        self.index = faiss.IndexFlatIP(V.shape[1])
        self.index.add(V)
        return self

    def search(self, query: str, k: int = 100) -> List[Tuple[str, float]]:
        q = self.enc.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        s, i = self.index.search(q, min(k, len(self.ids)))
        return [(self.ids[j], float(v)) for v, j in zip(s[0], i[0]) if j >= 0]

    def save(self, prefix: str | Path):
        import faiss
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(prefix) + ".faiss")
        Path(str(prefix) + ".ids").write_text("\n".join(self.ids), encoding="utf-8")

    def load(self, prefix: str | Path) -> "DenseIndex":
        import faiss
        self.index = faiss.read_index(str(prefix) + ".faiss")
        self.ids = Path(str(prefix) + ".ids").read_text(encoding="utf-8").split("\n")
        return self
