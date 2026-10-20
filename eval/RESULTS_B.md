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
**Extraction and resolution precision: human check of the full 300-row sample.** People filled
`extraction_correct` (150 rows: 30 by K Deepika, 120 by Aishwarya Nachammai N) and `resolution_correct` (150
rows, Aishwarya Nachammai N) in `eval/citation_precision_sample.csv`, using `eval/check_tool.py` with the AI
pre-labels hidden by default. 18 reporter citations first marked "no" were flagged as inconsistent with the
guide (e.g. *Hanumant v. State of M.P.*, AIR 1952 SC 343) and re-checked by the same person, who changed all 18
to "yes". Source: `eval/graph_precision.json` (`python -m stage0_graph.precision`), which records this provenance.

| check | correct / checked | precision | 95% interval (Wilson) |
|---|---|---|---|
| extraction, all | 140 / 150 | **0.933** | 0.882 to 0.963 |
| extraction, reporter citations | 76 / 76 | 1.000 | |
| extraction, bare case names | 64 / 74 | 0.865 | |
| resolution | 145 / 150 | **0.967** | 0.924 to 0.986 |

* **Extraction errors (10, all bare case names):** 6 are all-capitals page headers or title lines read as
  citations, 2 cut the name at the wrong party (e.g. "Lrs. v. Babu (dead)"), and 2 run on into the following
  words ("... v. Nirmala and Another and").
* **Resolution errors (5):** 3 come from `name_only` matching, which picks a different case between the same
  parties (e.g. *Kishor R. Ajmera v. SEBI* resolved to *SEBI v. Kishore R. Ajmera*, 2016); 2 are reporter-key
  matches to the wrong judgment.
* An earlier AI check (Claude) of the same sample gave extraction 0.827 and resolution 0.953 (git history of
  `eval/graph_precision.json`, commit f616bad). The human check agrees on the pattern and finds fewer
  bare-name errors.

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

### 2.3 Against the 400-row human-reviewed gold sample (`eval/stage4_results.json`)

The `label` column of `eval/gold_to_label.csv` was reviewed by a person for all 400 rows (commit cee7305),
starting from the AI pre-labels in `assistant_label`. The human label differs from the AI pre-label on 39 rows.
Classes: neutral 213, followed 142, distinguished 23, overruled 22.

| system | n | macro-F1, 3 classes (F/D/O) | macro-F1, 4 classes | F1 followed | F1 distinguished | F1 overruled | F1 neutral |
|---|---|---|---|---|---|---|---|
| keyword rules | 400 | 0.385 | 0.452 | 0.155 | 0.323 | 0.679 | 0.651 |
| InLegalBERT classifier | 400 | 0.354 | 0.403 | 0.706 | 0.355 | 0.000 | 0.552 |
| SCR editorial label itself | 319 | 0.519 | 0.489 | 0.664 | 0.312 | 0.582 | 0.398 |
| zero-shot LLM (gpt-oss-120b via Groq, temperature 0) | 400 | **0.778** | **0.721** | 0.648 | 0.776 | 0.909 | 0.552 |
| Indian Kanoon sentiment | | not run (no API access) | | | | | |

Predictions: `eval/stage4_llm_preds.csv` (`python -m stage4_validity.llm_baseline`). Confusion matrices:
`eval/figures/stage4_confusion_gold_*.png`.

**How to read this.**
* **The zero-shot LLM is the strongest treatment classifier**, by a wide margin on distinguished and
  overruled. Its main error is calling neutral mentions "followed" (120 of 213 neutral rows); it rarely misses a
  real treatment (followed 127/142, distinguished 19/23, overruled 20/22 correct).
* **Caveat:** the human review started from AI (Claude) pre-labels, and 361 of 400 were kept unchanged. An LLM
  may agree more easily with labels another LLM proposed, so the LLM's lead may be somewhat inflated.
* **The fine-tuned classifier and the editorial labels agree poorly with the human gold** (0.354 and 0.519).
  The classifier still never predicts overruled (13 training examples). The editorial labels mark many
  mentions "overruled" where the citing court only reports an overruling by another bench (see below).
* Stage 4's validity verdicts are built from the classifier and editorial labels, not from the LLM. Using the
  LLM for treatment classification is the clearest improvement these results point to.

#### Earlier runs, before human labels (kept for comparison)

* **Run 1, copied suggestions** (`eval/stage4_results.json` as of commit f616bad; now replaced by the human run). The `label` column was filled by copying
  `suggested_label` (the SCR editors' label, else the keyword-cue guess). 340 rows; 60 rows with no
  suggestion are blank and skipped. Classes: followed 110, distinguished 100, overruled 60, neutral 70.
* **Run 2, AI review** (`eval/stage4_results_ai_review_gold.json`). `assistant_label` was scored as gold from
  a scratch copy; `eval/gold_to_label.csv` was not modified. Claude labelled each row from the sentence and
  context without reading `suggested_label`. 398 rows; 2 "unclear" rows skipped. Classes: neutral 230,
  followed 140, distinguished 21, overruled 7.
  * Three rows (g096, g364, g374) carry a "2nd review" note and differ from the first AI pass.

| system | Run 1: copied suggestions, macro-F1 3-class (4-class) | Run 2: AI review, macro-F1 3-class (4-class) |
|---|---|---|
| keyword rules | 0.350 (0.359) | 0.225 (0.327) |
| InLegalBERT classifier | 0.394 (0.375) | 0.345 (0.401) |
| SCR editorial label itself | 1.000 (1.000), n=319 | 0.402 (0.387), n=318 |
| zero-shot LLM | not run on these labels | not run on these labels |
| Indian Kanoon sentiment | not run (no API access) | not run |

Per-class F1 of the classifier: overruled 0.0 in both runs. Followed is 0.585 in Run 1 and 0.692 in Run 2;
distinguished is 0.598 in Run 1 and 0.343 in Run 2.

**How to read this.**
* **Run 1 is circular.** Its "gold" is the weak labels themselves: the editorial label scores 1.0 against
  itself, and the keyword rules get overruled F1 0.72 partly because 21 suggestions are the cue output. The
  classifier was also trained on editorial labels. Run 1 therefore measures agreement with the training
  signal, not accuracy.
* **Run 2 uses independent labels, but they are AI labels.** On them the editorial labels agree only 0.40
  (3-class macro-F1). Most of the 60 editorial "overruled" rows report an overruling by another bench (see
  below).
* **Both are superseded by the human-reviewed gold above.**

Observations from the AI review (made before the human review):

* Of the 60 rows whose SCR editorial label is "overruled", the AI review found the citing court itself
  overruling in only a handful. Most of the others quote or report an overruling by another bench, typically
  the Constitution Bench in *Indore Development Authority* overruling *Pune Municipal Corporation*. The cited
  case is still overruled, so the validity verdict can be right. But the overruling judgment is often not the
  one whose bench size Stage 4 weighs.
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
no. 1). It also wrongly rejects near-verbatim sentences (P(entail) about 0.34). Mean time per pair on an
M-series Mac (MPS): 58.7 s for both summaries.

**Hand check of the filter** (`eval/nli_check_sample.csv`, `eval/nli_check_results.json`). 40 kept and 40
rejected query-focused sentences (random, seed 13), shuffled together. Aishwarya Nachammai N marked each one
supported or not from the judgment, without seeing the filter's decision (`eval/check_tool.py`); "can't tell"
rows are skipped (4 kept, 2 rejected). `python -m stage5_summary.nli_check --score`.

| | correct / checked | precision | 95% interval (Wilson) |
|---|---|---|---|
| rejected sentences that are really unsupported | 25 / 38 | 0.658 | 0.499 to 0.788 |
| kept sentences that are really supported | 29 / 36 | 0.806 | 0.650 to 0.903 |

* **The filter helps but is far from exact.** About a third of its rejections throw away faithful sentences,
  and about 1 in 5 kept sentences is still unsupported. Since 14 of 25 summaries lose a whole field, the
  false rejections cost the most.
* Read with the rejection rates above: rejected sentences are unsupported about 66% of the time, kept ones
  about 19%. The filter therefore concentrates the errors in what it removes, but a 0.5 threshold on a
  3-sentence premise is too strict for faithful paraphrases.

A first run had a bug: the model sometimes copied the `Query: ... ||` input prefix into relevance_to_query.
The filter rejected those copies, which put that field's rejection rate at 44%. Fixed in `generate.strip_echo`
and re-run; the table is the fixed run. The other four fields were identical in both runs.

### 3.4 Human evaluation: generic vs query-focused

**Human evaluation: 2 raters, 25 blinded pairs** (`eval/human_eval_sheet_rater1.csv`, `_rater2.csv`;
scored in `eval/human_eval_results.json` by `stage5_summary/eval_pairs.py`; guide `eval/HUMAN_EVAL_GUIDE.md`).
Each pair is one retrieved case summarised two ways, shown as A and B in random order (key:
`eval/human_eval_key.csv`). Means over 50 ratings (25 pairs x 2 raters), 1-5 scale:

| | query-focused | generic |
|---|---|---|
| faithfulness | 3.64 | 3.80 |
| relevance to the query | 1.32 | 1.24 |
| usefulness | 2.24 | 2.22 |
| preferred (of 50 judgments) | 18 | 29 (3 ties) |

Rater agreement: same preference on 19 of 25 pairs (Cohen's kappa 0.56); same score on 100 of 150 score cells
and within one point on 144. Quadratic-weighted kappa per criterion: relevance 0.72, usefulness 0.36,
faithfulness 0.10.

* **Generic summaries are preferred, and the two styles score about the same.** Relevance is low for both
  because in most pairs the retrieved case itself is off-topic (no cross-encoder; corpus 2018-2023), and no
  summary can make an off-topic case relevant.
* **The faithfulness rating is unreliable** (kappa 0.10): the sheet holds only the summaries, so checking them
  needs the full judgment, which raters were told to open only if unsure. The automatic NLI check (§3.3) is the
  better faithfulness measure.
* The result matches the AI-judge stand-in below on direction (generic preferred overall).

**AI-judge stand-in, NOT a human evaluation (earlier, kept for comparison)** (`eval/ai_judge_sheet.csv`, `eval/ai_judge_results.json`).
* **Rater and bias:** Claude rated the 25 blinded pairs. It is the same assistant that built the pipeline, so
  self-preference bias is possible.
* **Blinding:** the 5-field format reveals which summary is query-focused.
* **Scope:** relevance and usefulness only. Faithfulness was not rated because the full judgments were not
  read; §3.3 covers it automatically.

| | query-focused | generic |
|---|---|---|
| all 25 pairs: relevance (1-5) | 2.16 | 1.72 |
| all 25 pairs: usefulness (1-5) | 2.40 | 2.84 |
| all 25 pairs: preferred | 8 | 14 (3 ties) |
| 8 pairs where the retrieved case is relevant: relevance | 4.00 | 2.88 |
| 8 pairs where the retrieved case is relevant: usefulness | 3.00 | 3.25 |
| 8 pairs where the retrieved case is relevant: preferred | 4 | 3 (1 tie) |

* **When the retrieved case is relevant**, the query-focused summary was judged more relevant and preferred
  slightly more often.
* **Overall, the generic summary is preferred.** In 17 of 25 pairs the retrieved case is off-topic: no
  cross-encoder, and the corpus is limited to 2018-2023. There, the 5-field summary is often fragmentary
  (fields cut off, or emptied by the NLI filter), while the generic summary reads better.

## 4. End-to-end

| | value | file |
|---|---|---|
| API `/search`, top-10 with validity + extractive summary per hit | 0.59 s (one query, in-process) | Phase 4 check |
| LLM-only baseline: named cases confirmed to exist (see 4.1) | 8 of 33 (24%) | `eval/llm_only/results.json`, `cases.csv` |

### 4.1 LLM-only baseline (`python -m baseline_llm.run --engine --nli`)

The LLM (gpt-oss-120b via Groq, temperature 0) was asked for up to 10 Supreme Court precedents, a 5-field
summary and a validity status for each of the 5 demo queries (`app/latency_test.py`). Raw replies:
`eval/llm_only/responses.jsonl`. Two replies were re-requested: q3 came back cut off mid-JSON during a network
outage, and q4 came back empty (the reasoning model used its whole token budget).

| | value |
|---|---|
| cases named (5 queries) | 33 (q1 5, q2 10, q3 5, q4 10, q5 3) |
| resolved by the Stage 0 resolver | 12 |
| of those, the matched case is clearly a different one (checked by reading titles) | 3 (*Rajesh Gautam* to *Rajiv Gupta*; *K. M. Nanavati* to *S. N. Dutt*; *Mohan Lal v. State of Punjab* to *State of M.P. v. Mohan Lal*) |
| uncertain match (party names only, no year) | 1 (*Hindustan Zinc Ltd. v. Union of India*) |
| **confirmed to exist in our graph** | **8 of 33 (24%)** |
| citations of the form "(year) N SCC 1" | 24 of 33 |
| distinct cases per query | q1 4 of 5, q3 1 of 5 (the same case five times) |
| status given by the LLM | good law 32, distinguished 1 |
| validity agreement where ours is decided | 6 of 7 (the LLM calls *Tulsiram Patel* distinguished; ours: followed) |
| overlap with our engine's top 10 | 0 |
| NLI check of the LLM's summaries | not possible: none of the 33 cases has full text in our corpus |

* **Unresolved does not mean invented.** Some unresolved cases are real but outside what the resolver can
  match (e.g. *Selvi v. State of Karnataka*, *K. M. Nanavati*). The resolver also makes false matches, so the
  12 raw matches were checked by hand.
* **Still, the signs of fabrication are strong:** most citations point to page 1 of a volume, one query
  repeats the same case five times, and the LLM labels almost every case good law. Without grounding it cannot
  flag a case that is no longer good law, which is what Stage 4 adds.
* **The LLM's cases are older landmark judgments:** all 33 are dated before 2018 (31 before 2010). Our engine
  searches the 2018-2023 full-text corpus, which explains zero overlap and the missing NLI check.
