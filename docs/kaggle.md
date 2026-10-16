# GPU steps on Kaggle (T4)

1. New notebook → Settings: **Accelerator GPU T4**, **Internet on**.
2. Clone and install:

```python
!git clone https://github.com/aishwaryanachammain-create/nyayasethu
%cd nyayasethu
!pip install -q -r requirements.txt
```

3. Data: run `!python -m stage0_corpus.download --years 2018-2023`, or attach a Kaggle dataset that
   already holds `data/raw/` and symlink it. Then build the corpus:

```python
!python -m stage0_corpus.build_corpus
!python -m stage0_corpus.build_sc_pcr
```

4. Stage 1 with frozen InLegalBERT features (step 12). Embeddings are computed once per fold, using
   fp16 and batch 256:

```python
!python -m stage1_roles.train --features inlegalbert
!python -m stage1_roles.label_corpus
```

5. Bi-encoder (step 20) and cross-encoder (step 25):

```python
!python -m stage2_retrieval.train_biencoder
!python -m stage2_retrieval.build_index --corpus
!python -m stage3_rerank.cross_encoder train
!python -m stage3_rerank.rerank --bench sc-pcr
!python -m eval.ablation
```

6. Download `models/`, `logs/` and `data/segments.parquet` from the notebook's output panel. Commit
   only `logs/` and the hand-off files; share `models/` as a Kaggle dataset.
