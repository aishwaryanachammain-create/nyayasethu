# Member B results (Stages 0-graph, 4, 5, integration)

Every number below comes from a run whose output is in the repository (file given per table). Anything not run
yet is marked **PENDING**. Scores against weak (silver) labels are reported as *agreement*, never as accuracy.

## 1. Citation graph (Stage 0, `logs/graph_build.json`)

Corpus: 7,949 judgments (4,985 Supreme Court 2018-2023 from the AWS open data, 2,914 AILA, 50 Bhattacharya).
Nodes also include every Supreme Court case in the AWS metadata (1950-2025), which can be cited but not cite.

| statistic | value |
|---|---|
| nodes (metadata + corpus) | 41,116 |
| citation mentions extracted | 108,706 |
| mentions resolved to a case | 37,372 (34.4%) |
| reporter citations resolved | 41.1% (SCR 91.0%, SCC 38.9%, AIR 30.5%) |
| bare case names resolved | 27.5% |
| edges, judgment level / mention level | 30,011 / 35,806 |
| cases cited at least once | 11,665 |
| cases with >= 3 citing judgments (enough for a validity verdict) | 3,389 (29.1% of cited cases) |
| max in-degree | 93 (*Sharad Birdhichand Sarda v. State of Maharashtra*, 1984) |
| mean in-degree of cited cases | 2.57 |

In-degree distribution: `eval/figures/indegree_distribution.png`.
Extraction and resolution precision: **PENDING** a human check of the 300-row sample in
`eval/citation_precision_sample.csv` (guide `eval/CITATION_CHECK_GUIDE.md`; score with
`python -m stage0_graph.precision`).

The sample carries AI pre-labels (`assistant_correct`, never scored), so these numbers are preliminary:

| check | AI pre-label "correct" |
|---|---|
| extraction, reporter citations | 76 / 76 |
| extraction, bare case names | 48 / 74 |
| resolution | 142 / 149 (1 unclear) |

Most bare-name errors are running page headers (the case title printed at the top of each page) being read as
citations. Headers of the citing judgment itself are already dropped as self-citations (8,518 mentions).
Two kinds of leak remain in the graph, both small:
* 197 `name_only` edges come from all-capitals names, most of them headers;
* 137 edges sit in an SCR editorial "Case Law Reference" list that the front-matter filter missed.

Together that is about 0.9% of 35,806 edges. The proposed fix is to drop all-capitals name mentions and lines
matching "<citation> referred to/relied on Para N". It has not been applied: it would need a full graph and
Stage 4 rebuild.

## 2. Doctrinal validity (Stage 4)

### 2.1 Weak labels (`logs/stage4_weak_labels.json`)

The SCR editors' "Case Law Reference" blocks give a treatment for 12,638 mentions: followed 4,511,
distinguished 613, overruled 93, neutral (referred to) 7,421. The keyword (cue) rules agree with the editors on
31.8% of the 2,284 mentions both label.

### 2.2 Treatment classifier: agreement with editorial labels on the held-out weak test split (`eval/stage4_results.json`)

1,384 mentions (followed 498, distinguished 63, overruled 4, neutral 819). Not gold.

| model | macro-F1, 3 classes (F/D/O) | macro-F1, 4 classes | F1 followed | F1 distinguished | F1 overruled |
|---|---|---|---|---|---|
| keyword rules | 0.171 | 0.299 | 0.205 | 0.072 | 0.235 |
| InLegalBERT classifier | 0.286 | 0.350 | 0.589 | 0.269 | 0.000 |

The classifier saw only 13 overruled training examples (`logs/stage4_train.json`), so it never predicts
overruled; overruling verdicts come from the editorial labels. Confusion matrices: `eval/figures/stage4_confusion_*.png`.

### 2.3 Against hand-labelled gold

| model | macro-F1 (3 classes) |
|---|---|
| keyword rules | **PENDING** (0 of 400 rows in `eval/gold_to_label.csv` labelled) |
| InLegalBERT classifier | **PENDING** |
| zero-shot LLM (`python -m stage4_validity.llm_baseline`) | **PENDING** (no API key in `.env`) |
| Indian Kanoon sentiment labels | not run (no API access) |

**Pre-annotation (not gold).** All 400 rows carry an `assistant_label` proposed by an AI assistant that read the
sentence and context but not `suggested_label`. Counts: neutral 231, followed 139, distinguished 23, overruled 5,
unclear 2 (137 marked low-confidence). These labels are never scored; gold stays PENDING until a person fills
`label` (workflow in `eval/GOLD_LABELLING_GUIDE.md`). Two observations, preliminary until a person checks them:

* Of the 60 rows whose SCR editorial label is "overruled", the assistant found the citing court itself
  overruling in only 5. Most of the others quote or report an overruling by another bench, typically the
  Constitution Bench in *Indore Development Authority* overruling *Pune Municipal Corporation*. The cited case
  is still overruled, so the validity verdict can be right. But the overruling judgment is often not the one
  whose bench size Stage 4 weighs.
* Two rows are extraction errors: g135 resolves a running page header, and g329 takes its citation from the
  SCR editors' Case Law Reference list, which the front-matter filter missed.

### 2.4 Coverage of the verdicts (`logs/stage4_aggregate.json`)

| | cases | share |
|---|---|---|
| cited cases | 11,665 | |
| followed | 2,991 | 25.6% |
| distinguished | 208 | 1.8% |
| overruled | 40 | 0.3% |
| insufficient evidence | 8,426 | 72.2% |
| **decided (coverage)** | **3,239** | **27.8%** |

The 40 overruled verdicts all rest on an SCR editors' "overruled" entry. Spot checks found real overrulings
(e.g. *Pune Municipal Corporation*, *Surya Dev Rai*, which was partly overruled, and *Siddharam Mhetre*).
Because the full-text corpus covers 2018-2023, recent judgments rarely have 3 later citers yet. That is why
most top-10 results in the demo show "insufficient evidence".

## 3. Summarisation (Stage 5)

### 3.1 Generic summarisation on IN-Abs test, 100 judgments (`eval/stage5_inabs_results.json`)

Legal-LED (`nsi319/legal-led-base-16384`) fine-tuned on Kaggle (T4, fp16, gradient checkpointing, batch 1,
accumulation 16, 8,192 input / 512 output tokens, lr 3e-5, 1 epoch over 6,830 training pairs; validation loss
1.584). Beam 2. I recomputed the ROUGE scores locally from `eval/stage5_inabs_predictions.csv` and got the same values.

| model | ROUGE-1 | ROUGE-2 | ROUGE-L | ROUGE-Lsum | BERTScore F1 |
|---|---|---|---|---|---|
| Legal-LED, untuned | 0.330 | 0.112 | 0.174 | 0.285 | 0.817 |
| Legal-LED, fine-tuned | **0.418** | **0.221** | **0.234** | **0.392** | **0.852** |

ROUGE-L is the sentence-agnostic LCS and ROUGE-Lsum the summary-level LCS. BERTScore uses roberta-large without baseline
rescaling. The reference headnotes average 932 words and the outputs about 300 (512-token cap), which limits recall.

### 3.2 Role coverage on IN-Ext, 50 judgments, two experts (`eval/stage5_inext_results.json`)

`coverage` is the share of the experts' extracted sentences that are in the source pool our field draws on.
`pool share` is how much of the judgment that pool is, so coverage is only meaningful next to it. ROUGE compares
our extractive field text (word budget 50-150) with the expert segment summary.

| field (expert segment) | roles from | coverage | pool share | ROUGE-1 | ROUGE-L |
|---|---|---|---|---|---|
| material_facts (facts) | gold roles | 0.996 | 0.261 | 0.399 | 0.329 |
| | position fallback | 0.972 | 0.247 | 0.372 | 0.287 |
| | A's labeller (in-sample) | 0.931 | 0.267 | 0.379 | 0.292 |
| ratio (analysis) | gold roles | 0.861 | 0.401 | 0.251 | 0.162 |
| | position fallback | 0.955 | 0.753 | 0.203 | 0.124 |
| | A's labeller (in-sample) | 0.725 | 0.367 | 0.251 | 0.168 |
| decision (judgement) | gold roles | 0.980 | 0.032 | 0.656 | 0.638 |
| | position fallback | 0.750 | 0.031 | 0.344 | 0.292 |
| | A's labeller (in-sample) | 0.860 | 0.033 | 0.470 | 0.436 |

The experts' argument and statute segments have no field of their own. A's labeller was trained on these same
50 judgments, so its row is optimistic. With gold roles, the decision field draws on 3% of a judgment and still
contains 98% of what the experts selected as the judgement.

### 3.3 Faithfulness filter: entailment rejection rate (`eval/stage5_nli_results.json`)

25 query-judgment pairs: the 5 queries in `app/latency_test.py` x Member A's top 5 for each (BM25 + PageRank
prior; no cross-encoder). Fine-tuned Legal-LED and NLI `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`, threshold
0.5. Premises are the 3 most similar source sentences of the field.

| output | generated sentences | rejected | rejection rate |
|---|---|---|---|
| query-focused, all fields | 441 | 163 | **37.0%** |
| issue | 44 | 15 | 34.1% |
| material_facts | 119 | 52 | 43.7% |
| ratio | 153 | 63 | 41.2% |
| decision | 51 | 19 | 37.3% |
| relevance_to_query | 74 | 14 | 18.9% |
| generic summary (premises from the whole judgment) | 306 | 147 | 48.0% |

14 of 25 query-focused summaries have at least one field left empty after filtering. A rejection is not
proven to be a hallucination: this is the filter's rejection rate, not an error rate. Reading a sample of
rejected sentences shows both kinds. The filter catches real errors: a wrong section number (s.15 for s.14 of
SARFAESI), a flipped charge ("failure **not** to discharge"), and a wrong party ("respondent no. 2-bank" for
no. 1). It also wrongly rejects near-verbatim sentences (P(entail) about 0.34). The precision of the filter
is **PENDING** a hand check. Mean time per pair on an M-series Mac (MPS): 58.7 s for both summaries.

A first run had a bug: the model sometimes copied the `Query: ... ||` input prefix into relevance_to_query.
The filter rejected those copies, which put that field's rejection rate at 44%. Fixed in `generate.strip_echo`
and re-run; the table is the fixed run. The other four fields were identical in both runs.

### 3.4 Human evaluation: generic vs query-focused

**PENDING.** The sheet `eval/human_eval_sheet.csv` has 25 blinded pairs, and the key is in
`eval/human_eval_key.csv`. Instructions: `eval/HUMAN_EVAL_GUIDE.md`. Score with
`python -m stage5_summary.eval_pairs --score <filled sheets>`.

## 4. End-to-end

| | value | file |
|---|---|---|
| API `/search`, top-10 with validity + extractive summary per hit | 0.59 s (one query, in-process) | Phase 4 check |
| LLM-only baseline (precedents exist? validity agrees with ours?) | **PENDING** (no API key) | `python -m baseline_llm.run --engine --nli` |
