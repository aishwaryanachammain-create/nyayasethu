# NyayaSethu: Member B report

Member B's part of NyayaSethu covers the citation graph (Stage 0), doctrinal validity (Stage 4), query-focused
summarisation with a faithfulness filter (Stage 5), integration with Member A's API, the demo UI, and the
evaluation. All numbers are in `eval/RESULTS_B.md`, which marks everything not yet run as PENDING. The work is
on branch `member-b`, one commit per phase.

## 1. Method

### 1.1 Citation graph (`stage0_graph/`)

* **Nodes:** every Supreme Court case in the AWS open-data metadata (38,152 cases, 1950-2025) plus the 7,949
  corpus judgments. Corpus judgments are the citing side, because only they have text.
* **Extraction** (`extract.py`): regular expressions for reporter citations (SCR, SCC, AIR, JT, SCALE, ILR, INSC,
  Suppl. forms), each paired with the nearest "X v. Y" party name, plus bare case names. Front matter
  (headnotes, the editors' reference list) is skipped so a judgment's own headnote does not count as a
  citation. All quantifiers that could backtrack badly are possessive or bounded. One ALL-CAPS heading hung
  the first build for 80 minutes.
* **Resolution** (`resolve.py`), in order of strength:
  * exact reporter key against the case's own SCR/INSC citations;
  * SCR page within a reported case's page range;
  * party names fuzzy-matched with rapidfuzz near the citation year, using rare-token blocking, threshold 85
    and a margin of 5 over the runner-up;
  * bare names with a stricter threshold of 92.
  A citation can never resolve to a case decided after the citing judgment. Each edge records its method
  and score.
* **Outputs:** PageRank (networkx) goes to `data/from_b/pagerank.csv`, which A's Stage 3 reads as its
  authority prior. Edges go to `data/from_b/edges.csv`.

### 1.2 Doctrinal validity (`stage4_validity/`)

* **Weak labels** (`labels.py`, `editorial.py`, `dataset.py`): the SCR editors' "Case Law Reference" block lists,
  for each case a judgment cites, how it was treated ("relied on", "followed", "distinguished", "overruled",
  "referred to"...). These map to followed / distinguished / overruled / neutral and are joined to graph edges
  by reporter key and paragraph. Keyword (cue) rules with negation handling give a second, weaker source and
  serve as the rule baseline.
* **Classifier** (`classifier.py`): InLegalBERT, given the citing sentence with the citation marked `[[ ]]`
  and the ±2-sentence context, 4 classes. It uses class-weighted cross-entropy and selects the epoch by dev
  3-class macro-F1. The 400 gold-sample mentions are excluded from training.
* **Aggregation** (`aggregate.py`):
  * Each citing judgment's vote is weighted by bench ratio (capped at 2) × exp(−age/15).
  * **overruled:** an overruling with confidence > 0.7 from a bench at least as large as the cited one.
  * **doubted:** an overruling from a smaller bench, or from an unknown bench backed only by the classifier.
  * **insufficient evidence:** fewer than 3 citing judgments, or no followed/distinguished votes.
  * **Otherwise:** the weighted majority between followed and distinguished.
  * The editors' label is preferred over the classifier wherever both exist.
* **API:** `get_validity(case_id)` gives the verdict, its confidence, the votes, and the evidence sentences.

### 1.3 Query-focused summarisation (`stage5_summary/`)

* **Roles** (`docs.py`): taken from A's segments (gold for Bhattacharya), else A's Stage 1 labeller run on
  the one judgment, else a position fallback. The source used is always reported.
* **Fields** (`fields.py`):
  * issue ← Issue sentences, else sentences stating a question;
  * material_facts ← Facts;
  * ratio ← Ratio + Ruling by Present Court;
  * decision ← Ruling by Present Court, else the last 5 sentences;
  * relevance_to_query ← Ratio + Precedent + Statute, ranked by TF-IDF similarity to the query.
* **Generation** (`generate.py`): Legal-LED fine-tuned on IN-Abs (`notebooks/stage5_led_finetune.ipynb`,
  Kaggle T4), with one call per field and the query prefixed for relevance_to_query. An extractive mode,
  with no model, is used by the API.
* **Faithfulness** (`nli.py`): each generated sentence is checked by DeBERTa-v3 NLI against the 3 most similar
  sentences of that field's source. It is kept if P(entail) > 0.5; rejected sentences are logged with their
  scores.
* **API:** `summarize(query, case_id)` returns the contract's five fields, `rejected`, and the provenance.

### 1.4 Integration and demo

* **A's API:** `validity/` and `summary/` are thin re-exports that A's `app/b_interface.py` imports, so A's
  `/search` now returns B's verdict and summary for every hit (0.59 s for a top-10) without any change to
  A's code. The API uses the extractive summary because Legal-LED takes about 20-35 s per case on a laptop.
* **Streamlit UI** (`ui/app.py`): a query box; top-k cards with a colour-coded validity badge; tabs for the
  5-field summary (Legal-LED + NLI on demand, with removed sentences listed), the validity evidence, and the
  role-coloured judgment. It has three modes: mock data, A's engine in-process, and A's API.
* **LLM baselines** (`baseline_llm/`): there are two.
  * Zero-shot treatment classification on the gold contexts.
  * An LLM-only pipeline: the LLM names precedents, summaries and validity, and the script checks whether
    each case exists in the graph, whether the LLM's validity matches ours, overlap with A's top-k, and NLI
    support for the LLM's summaries.
  
  The API key is read from `.env`, which is gitignored.

## 2. Deviations from Review 1, and why

| Review 1 | What was done | Why |
|---|---|---|
| Distant supervision from OpenNyAI precedent roles (Relied / Not relied / Overruled) | SCR editors' "Case Law Reference" treatments, with keyword rules as a secondary source and as the baseline | OpenNyAI's 354 judgments contain very few overruled examples and are not in our corpus. The editors' blocks sit inside our own SCR PDFs, give 12,638 labelled mentions, and name the treatment explicitly. |
| 3 classes: followed, distinguished, overruled | Adds a 4th class, **neutral** ("referred to") | Most citations are plain references. Forcing them into 3 classes would turn every mention into a vote. Macro-F1 is still reported over the 3 doctrinal classes. |
| Verdict from the classifier | The editors' label is preferred when present, the classifier otherwise | It is the more reliable signal. With 13 overruled training examples the classifier never predicts overruled. |
| Bench weight from both benches | Unknown bench size defaults to 2 (a division bench); an overruling from an unknown bench backed only by the classifier becomes "doubted" | The metadata judge field often lists only the author. The SCR header gives the bench for corpus judgments only. |
| ILDC for the graph | AWS Supreme Court metadata (1950-2025) as nodes, plus the corpus (A's choice) | ILDC was not used by A. The metadata provides the citable universe and its SCR keys. |
| Indian Kanoon sentiment baseline | Not run | No API access. |
| Legal-LED receives "query facts followed by role-ordered segments" and emits five fields in one pass | One generation per field from that field's role sentences; the query is prefixed only for relevance_to_query | IN-Abs fine-tuning teaches headnotes, not a 5-field format. Generating per field keeps each field grounded in its own roles and lets the NLI check use that field's source as premise. |
| Issue role | Issue sentences when labelled, else sentences that state a question | A's role data (Bhattacharya scheme) has no Issue class. |
| Query ranking and NLI premises | TF-IDF cosine | Fast, needs no model, and auditable. |
| Stage 1 labeller | A's LSA + BiLSTM-CRF labeller, trained locally into `models/stage1/` with A's code | A's trained model was not in the repo. My run wrote its logs elsewhere, so A's `logs/stage1_cv_lsa.json` is untouched. Local 2-fold CV macro-F1 was 0.475. |
| Human rating "for the same judgment under differing queries" | 25 pairs: 5 queries × A's top 5 (generic vs query-focused on the same judgment) | A's retrieval rarely returns one judgment for several of the queries. The sheet can be extended with `--per-query`. |

## 3. Results (details in `eval/RESULTS_B.md`)

* **Graph:**
  * 108,706 citation mentions, 34.4% resolved (SCR 91%, SCC 39%);
  * 30,011 judgment-level edges;
  * 3,389 cases with at least 3 citing judgments.
  * Extraction precision, human-checked on 30 of 150 sampled rows: 0.83 (25/30; reporter citations 17/17,
    bare case names 8/13). An earlier AI check of all 150 gave the same picture (0.83).
    
    The human check of resolution is pending.
* **Validity:**
  * 27.8% of cited cases get a verdict (followed 2,991, distinguished 208, overruled 40); the rest are
    insufficient evidence.
  * Against held-out editorial labels (agreement, not gold), the classifier reaches 3-class macro-F1 0.286
    vs 0.171 for the keyword rules.
  * On the 400-row human-reviewed gold sample (3-class macro-F1): zero-shot LLM (gpt-oss-120b) **0.778**,
    editorial label 0.519, keyword rules 0.385, classifier 0.354. The review started from AI pre-labels (39 of
    400 changed), which may favour the LLM. Earlier runs on copied suggestions and on AI labels are kept in
    `eval/RESULTS_B.md` §2.3 for comparison.
* **Summaries:**
  * IN-Abs test, untuned → fine-tuned: ROUGE-1 0.330 → 0.418, ROUGE-2 0.112 → 0.221, ROUGE-L 0.174 → 0.234,
    BERTScore 0.817 → 0.852.
  * IN-Ext with gold roles: the decision field covers 98% of the experts' judgement sentences from 3% of the
    text, and the facts field covers 99.6% from 26%.
  * NLI rejection: 37.0% of the generated sentences in query-focused summaries, against 48.0% of generic
    summaries' sentences.
  * Human evaluation (2 raters, 25 blinded pairs): generic summaries are preferred 29 to 18 (3 ties); the
    two styles score about the same (relevance 1.32 vs 1.24, usefulness 2.24 vs 2.22, faithfulness 3.64 vs
    3.80, query-focused first). Raters agree on preference in 19 of 25 pairs (kappa 0.56), but hardly on
    faithfulness (weighted kappa 0.10). An earlier AI-judge stand-in also preferred generic summaries.
* **LLM-only baseline** (gpt-oss-120b, 5 demo queries): of 33 precedents named, 8 (24%) are confirmed to exist
  in our graph; 24 of 33 citations have the form "(year) N SCC 1"; the LLM calls 32 of 33 good law. None
  overlap our top 10. Details: `eval/RESULTS_B.md` §4.1.

## 4. Changes to Member A's files

Kept minimal. Each one was needed for B's stages to run on A's data.

| file | change | why |
|---|---|---|
| `stage0_corpus/loaders.py` | SC metadata join: strip the `_EN` suffix from PDF stems, add an `nc_display` alias; `bench_from_header()` (SCR `[X AND Y, JJ.]`); `iso_date()` | The join never matched, so the full corpus had no metadata and the graph had 0 edges |
| `config.yaml` | `paths.judgments` → `data/processed/judgments.jsonl` (full corpus, not in git); B's `stage4`/`stage5` blocks; `use_mock`; `stage5.api_mode` | The committed `data/judgments.jsonl` holds only the 50 Bhattacharya documents |
| `tests/fixtures/make_fixture.py` | one line: point `paths.judgments` at the fixture | A's pipeline test broke after the path change |
| `common/config.py` | `use_mock` redirect of hand-off paths to `data/mock` (Phase 0 merge) | The brief requires mock mode |
| `.gitignore`, `requirements.txt` | B's outputs, `.env` and model weights ignored; B's dependencies pinned | |
| `logs/corpus_build.json` | regenerated by A's own corpus build after the loader fix | |

**For Member A:**
* Corpus ids for SC judgments are `sc:<CNR>`.
* The full corpus (7,949 judgments, 727 MB) is not in git.
* The BM25 corpus index was built locally with A's `build_index --corpus --no-dense`.
* The cross-encoder (`models/cross_encoder`) is still untrained, so `/search` ranks by BM25 + PageRank only.

## 5. Limitations

* **Validity:**
  * Only 27.8% of cited cases get a verdict, because the full-text corpus covers 2018-2023 and a case needs 3
    citing judgments. Recent cases in the demo mostly show "insufficient evidence".
  * The classifier is weak (dev 3-class macro-F1 0.25) and cannot recognise overruling. Overruled verdicts
    depend on the editors' labels.
  * Bench sizes are often unknown.
  * The AI review (not human-verified; see `eval/RESULTS_B.md` §2.3) suggests that many editorial "overruled"
    entries sit in judgments that only report an overruling by another bench. The verdicts may still be
    right, but the bench-weighting then applies to the wrong judgment.
  * The human gold review started from AI pre-labels, so it is not fully independent of an LLM.
  * A zero-shot LLM classifies treatment far better than the fine-tuned classifier (0.778 vs 0.354), but the
    validity verdicts are still built from the classifier and editorial labels.
* **Graph:**
  * About two thirds of citation mentions stay unresolved, mostly SCC and AIR citations without a nearby
    party name, and bare names.
  * Only 30 extraction rows and no resolution rows are human-checked. Reporter citations look reliable. Bare
    case names do not: page headers get read as citations.
  * About 0.9% of edges come from page headers or an editorial reference list the front-matter filter missed.
    A fix is proposed in `eval/RESULTS_B.md` §1 but not applied.
* **Summaries:**
  * The NLI filter rejects about 37% of generated sentences, and 14 of 25 summaries lose a whole field. Some
    rejections are real errors (wrong section number, negation flip, wrong party). Others are faithful
    paraphrases the 3-sentence premise does not cover. Its precision is unmeasured.
  * The filter can keep a sentence that contradicts itself.
  * SCR page-margin letters ("A B C D E F G H") leak into the text.
  * Generation takes about 20-35 s per case on a laptop.
* **Evaluation:**
  * IN-Ext coverage measures the role pools rather than the generated text.
  * A's labeller is trained on the IN-Ext documents, so its coverage row is optimistic.
  * The human evaluation has 2 raters and 25 pairs; the 5-field format cannot be blinded, and faithfulness
    ratings barely agree between raters.
* **Retrieval quality** (A's stages) caps everything downstream. Without the cross-encoder some top results
  are off-topic (e.g. a compassionate-appointment case for a gratuity query).

## 6. Still to do (needs a person)

1. **Human citation check.** Resolution is unchecked (0 of 150) and extraction is checked on 30 of 150. A
   person should finish `eval/citation_precision_sample.csv` (guide: `eval/CITATION_CHECK_GUIDE.md`), then run:
   ```
   python -m stage0_graph.precision --provenance "<who checked>"
   ```
2. **NLI filter precision.** Hand-check a sample of kept and rejected summary sentences (§3.3 of
   `eval/RESULTS_B.md`).

Done since the first version of this report: human gold labels, human ratings, the Stage 4 LLM baseline and the
LLM-only baseline.

## 7. How to run

```bash
python -m stage0_graph.build                      # citation graph + PageRank
python -m stage4_validity.dataset                 # weak labels
python -m stage4_validity.classifier train && python -m stage4_validity.classifier predict
python -m stage4_validity.aggregate               # outputs/validity.json
python -m stage4_validity.evaluate                # eval/stage4_results.json
python -m stage5_summary.eval_inext               # IN-Ext role coverage
python -m stage5_summary.eval_pairs && python -m stage5_summary.eval_pairs --sheet
streamlit run ui/app.py [-- --mock | -- --api http://127.0.0.1:8000]
pytest -q
```
