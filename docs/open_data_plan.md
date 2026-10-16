# Member A — Retrieval Track, open-data version

Every dataset below downloads directly, with no form, login, registration key, or gated
"agree to share your contact information" step. Licences and access were checked on
16 Oct 2026; re-check them before the review and quote the DOIs.

## Dataset swap

| Original | Why it's out | Open replacement | Licence | Covers |
|---|---|---|---|---|
| LegalSeg (7,120 judgments) | Hugging Face gated (contact-info agreement) | **Bhattacharya et al. 2019** — 50 SC judgments, 7 roles (github.com/Law-AI/semantic-segmentation) **+ team hand-annotation** of ~20 SC judgments | MIT; own labels | Stage 1 training and test |
| IL-PCR (IL-TUR) | Hugging Face gated | **SC-PCR**, built by us from the SC corpus by masking citations (`stage0_corpus/build_sc_pcr.py`) | CC BY 4.0 (derived) | Stage 2-3 training + main test |
| AILA 2019 | — (already open) | **AILA 2019**, Zenodo 10.5281/zenodo.4063986 (50 queries, 2,914 candidates) | CC BY 4.0 | Stage 2-3 test, comparable to published BM25 |
| ILDC | Access form | **Indian Supreme Court Judgments**, AWS Open Data (`s3://indian-supreme-court-judgments`, ~35K English judgments 1950–, PDFs + metadata with judges, date, citation) | CC BY 4.0 | Corpus, metadata, citation graph |

What this costs:

* **Stage 1 has far less training data** (about 50 + 20 judgments instead of 7,120). Expect lower
  macro-F1 than LegalSeg's published numbers. Report the comparison to LegalSeg as context, not
  head-to-head.
* **Bhattacharya has no Issue role** and does not split arguments by party. Issue, which the
  acceptance gate depends on, comes only from the hand-annotated judgments. Annotate those
  carefully, with two people on 3–5 shared judgments to report Cohen's kappa.
* **The majority baseline will not be 50%.** That figure is a LegalSeg statistic. Compute it on
  your own test set; `evaluate` prints it.
* **SC-PCR numbers are not comparable to published IL-PCR numbers.** AILA 2019 stays comparable to
  published work, so report both.
* Bench size comes from the `judge` metadata field, with CORAM-header parsing as a fallback for
  judgments that lack it.

For Member B: OpenNyAI (Stage 4 distant supervision) is also gated, and there is no open dataset
with Relied / Not-Relied / Overruled labels. B's open route is the keyword-rule weak labels plus
300–500 hand-labelled citing sentences from `edges.csv`. IN-Abs / IN-Ext (Zenodo 10.5281/zenodo.7152317,
CC BY 4.0) are open and unchanged.

## Revised weekly plan

| Week | Tasks | Done when |
|---|---|---|
| 1 | Repo + folders . Download AILA 2019, Bhattacharya 2019 and 3–6 years of SC judgments + metadata (`python -m stage0_corpus.download`). Validate the legal-aware splitter on 200 sentences drawn **from the extracted SC PDF text**, since PDF line breaks and page headers are the realistic failure cases (`stage0_corpus.splitter_sample` / `splitter_validate`). | Splitter error rate logged |
| 2 | PDF → text (cached). Merge + dedup on case_id/CNR. Metadata: court, date, bench size from `judge`, falling back to the CORAM header. Build SC-PCR (`stage0_corpus.build_sc_pcr`), which also logs query–document lexical overlap. BM25 baseline on AILA 2019 and SC-PCR. | `judgments.jsonl` committed; BM25 MAP/nDCG recorded on both |
| 3 | Export 20 SC judgments for annotation (`stage1_roles.annotate export`), annotate (Issue, both argument roles), import. Build Stage 1: InLegalBERT → BiLSTM → CRF over **overlapping sentence windows** (64 sentences, stride 48), with class-weighted CE. | Training runs end-to-end on Bhattacharya + annotated set |
| 4 | Full training on Kaggle T4: train on Bhattacharya + ~14 annotated judgments; test on ~6 held-out annotated judgments. Macro-F1 with per-class table vs. the measured majority baseline. Label the whole corpus. | Issue and Reasoning recall meet the threshold; `segments.parquet` committed |
| 5 | Role-partitioned indexes (Facts, Reasoning). Bi-encoder over 512-token chunks (stride 128), max-pooled, in FAISS. RRF fusion with BM25. | Recall@100 vs. BM25 on SC-PCR and AILA |
| 6 | Cross-encoder over (query facts, candidate reasoning) on the top 100, trained on SC-PCR train queries, + PageRank prior from B's `pagerank.csv`. | P@10, MRR; **Review 2 demo** |
| 7 | FastAPI `/search` chaining Stages 1–3 + B's `get_validity` / `summarize`. | One query returns full JSON |
| 8 | Ablation: BM25 → +dense → +roles → +rerank → +prior, on both benchmarks. | Ablation table |
| 9 | Results tables + 5 failure cases. | Results section drafted |
| 10 | Report: Stages 0–3, architecture, datasets (licences + DOIs + the data limitations above). | **Final review** |

Commands: see the run-order table in the repository README.
