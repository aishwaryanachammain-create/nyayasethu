# NyayaSethu — Retrieval track (Member A)

Validity-aware precedent retrieval and summarization for Indian case law. NLP Digital Assignment by
Aishwarya Nachammai (23MIS1080) and Deepika K (23MIS1011).

This repository holds **Member A's part**: Stage 0 (corpus), Stage 1 (rhetorical-role labeller),
Stage 2 (role-aware hybrid retrieval), Stage 3 (reranking), the FastAPI backend, and retrieval
evaluation. Member B's Stage 4 (validity) and Stage 5 (summaries) plug in through
`app/b_interface.py`. Until they do, the API marks those fields `pending`.

Every dataset is **open access** (no request forms, logins or gated downloads). See `data/README.md`.

```
stage0_corpus/     download, loaders, legal-aware splitter (+ validation), metadata regex, corpus build, SC-PCR benchmark
stage1_roles/      sentence features (frozen InLegalBERT / LSA), BiLSTM-CRF tagger, CV training, corpus labelling, annotation tool
stage2_retrieval/  BM25 baseline + lexical overlap, bi-encoder fine-tuning, 512/128 chunk + max-pool FAISS, routed RRF hybrid
stage3_rerank/     cross-encoder, authority prior (B's PageRank or in-degree), alpha tuning, Review-2 CLI demo
app/               FastAPI: POST /search, GET /case/{id}, GET /health; latency test
eval/              metrics (ranx), benchmarks, ablation table, error analysis
common/            schema (hand-off formats), config, I/O, text utilities
config.yaml        every hyper-parameter and path
```

## Setup

```bash
git clone https://github.com/aishwaryanachammain-create/nyayasethu && cd nyayasethu
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                                 # offline end-to-end test on fictional fixtures
```

## Run order (the 36-step checklist)

| Steps | Command | Output |
|---|---|---|
| 3 | `python -m stage0_corpus.download` | data/raw/ (AILA, Bhattacharya, SC metadata + PDFs) |
| 4 | `python -m stage0_corpus.inspect_datasets` | data/README.md (schemas, counts, samples) |
| 7–9 | `python -m stage0_corpus.build_corpus` | **data/judgments.jsonl** → Member B; logs/corpus_build.json (metadata coverage) |
| 5–6 | `python -m stage0_corpus.splitter_sample` → fix line breaks → `python -m stage0_corpus.splitter_validate` | logs/week1_log.md (error rate) |
| — | `python -m stage0_corpus.build_sc_pcr` | data/processed/sc-pcr/ (open replacement for IL-PCR) |
| 10–11 | `python -m stage2_retrieval.bm25_baseline` | logs/results/bm25_*.json (baseline row + overlap) |
| 11b | `python -m stage1_roles.annotate export` → annotate → `... import` | data/processed/roles_gold.jsonl (Issue labels) |
| 12–17 | `python -m stage1_roles.train` (GPU: `--features inlegalbert`) | logs/stage1_cv_*.json, models/stage1/ |
| 18 | `python -m stage1_roles.label_corpus` | **data/segments.parquet** → Member B |
| 20 | `python -m stage2_retrieval.train_biencoder` | models/biencoder/ |
| 19, 21 | `python -m stage2_retrieval.build_index --corpus` | models/index_corpus/ |
| 25 | `python -m stage3_rerank.cross_encoder train` | models/cross_encoder/ |
| 26–27 | `python -m stage3_rerank.rerank --bench sc-pcr` | alpha tuned on dev; P@10 / MRR / nDCG@10 on test |
| 28 | `python -m stage3_rerank.demo_cli "..."` | Review-2 demo |
| 29–31 | `uvicorn app.main:app --port 8000` | API |
| 32 | `python -m app.latency_test` | logs/latency.json |
| 24, 33 | `python -m eval.ablation` | logs/results/ablation.md |
| 34 | `python -m eval.error_analysis --bench sc-pcr` | logs/results/error_analysis_sc-pcr.md |

The GPU steps (12, 20, 25) run on a Kaggle T4. See `docs/kaggle.md`.

## Demo UI

```bash
python scripts/build_demo_cache.py --abstractive 3 --llm   # once, before a demo (minutes: Legal-LED + LLM calls)
streamlit run ui/app.py                                    # real data; add `-- --mock` or `-- --api http://127.0.0.1:8000`
```

* **Example queries** (buttons under the hero): the five LLM-baseline queries plus two whose results include an
  overruled precedent (*Danamma*, overruled by *Vineeta Sharma*; *Garware Wall Ropes*). `?q=...` in the URL runs a
  query on load.
* **Each result card**: validity verdict with confidence, a red *not good law* strip naming the overruling judgment,
  and tabs for the summary (precomputed Legal-LED + NLI summaries load instantly; removed sentences are listed),
  validity evidence, citation graph (offline SVG), ranking trace (BM25 → RRF → cross-encoder → authority prior) and
  the role-coloured judgment with the passages closest to the query.
* **Compare with a plain LLM**: saved replies for the baseline queries (`eval/llm_only/`), live for any other query.
* **Sidebar filters** (court, year, hide overruled, bench size), **brief export** (Markdown, or print-ready HTML →
  PDF), and an **Evaluation** page that reads the committed results files.
* **Demo safety**: if the engine or API fails, the UI shows the saved results from `ui/demo_cache.json` with a banner.

## Results so far

**Stage 1, on real data:** the 50 Bhattacharya judgments (9,380 sentences), 5-fold document-level
cross-validation, offline LSA features (`logs/stage1_cv_lsa.json`):

| | macro-F1 | accuracy | majority baseline |
|---|---|---|---|
| BiLSTM-CRF, LSA features | **0.530 ± 0.029** | 0.619 | 0.386 (Ratio of the decision) |

| Role | P | R | F1 | n |
|---|---|---|---|---|
| Facts | 0.720 | 0.747 | 0.733 | 2219 |
| Ruling by Lower Court | 0.051 | 0.022 | 0.031 | 316 |
| Argument | 0.470 | 0.374 | 0.417 | 845 |
| Statute | 0.752 | 0.577 | 0.653 | 646 |
| Precedent | 0.538 | 0.568 | 0.553 | 1468 |
| Ratio of the decision | 0.613 | 0.677 | 0.644 | 3624 |
| Ruling by Present Court | 0.734 | 0.611 | 0.667 | 262 |

Gate (step 17): Ratio recall 0.677 passes the 0.6 threshold. **Issue is not measurable yet**,
because the open data has no Issue label. It needs the hand annotation in step 11b, so the gate is
reported as *not passed* until then. The InLegalBERT-feature run on Kaggle replaces the LSA row.

**Stages 2-3:** code is complete and tested end-to-end on fixtures. Real numbers need the AILA and
SC downloads, plus the GPU training for the bi-encoder and cross-encoder.

## Design notes

* **Open-data substitutions:** LegalSeg → Bhattacharya 2019 + own annotation; IL-PCR → SC-PCR
  (citation-masked, year-split, query/candidate pools disjoint); ILDC → AWS SC judgments.
  AILA 2019 keeps the results comparable to published work.
* **Role windows:** Stage 1 tags overlapping windows of 64 sentences (16 overlap). Each sentence
  keeps the prediction from the window where it sits most centrally.
* **Loss:** CRF NLL + 0.5 × class-weighted CE (weights ∝ 1/frequency) on the emissions.
* **Routing:** query facts → Facts index; full query → Reasoning index; BM25 + dense on both → RRF
  (k = 60) → top 100 → cross-encoder → `α·CE + (1−α)·log-prior` → top 10.
* **SC-PCR queries are masked.** Member B must build the citation graph from `judgments.jsonl`, not
  from the benchmark files.
* **Fixtures:** `tests/fixtures/judgments/` is fictional and exists only for offline tests.
