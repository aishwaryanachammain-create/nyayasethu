# Phase 0 Inspection Report (Member B)

Date: 2026-10-16. Branch: `member-b`. Every number below came from a command run during this inspection, unless it is marked as quoted from A's files.

---

## 0. Git reconciliation

* Your local `main` had `a9a43e9` ("Phase 0: structure, contract, config, downloads, mocks", 19:53). A's `origin/main` has `dca8b8a` ("Member A: Stages 0-3, FastAPI backend, retrieval evaluation", 19:14, K Deepika). The two have **no common ancestor**. A's commit has no parent, so A pushed a fresh history that replaced yours on GitHub.
* What I did, as you approved: kept `a9a43e9` on local `main` and on a new branch `backup-local-phase0` (neither is pushed). Created `member-b` from `origin/main` and merged `a9a43e9` with `--allow-unrelated-histories` as merge commit `48e9a66`.
* Conflict resolution (add/add in 4 files):
  * `common/config.py`: kept A's `cfg()`/`path()` and added the `use_mock` redirect to `paths.mock`.
  * `config.yaml`: kept A's file and added `use_mock` (set to **false**), `paths.mock/retrieval/validity/summary`, `models.summarizer/nli`, and the `stage4`/`stage5` blocks.
  * `requirements.txt`: combined both, keeping A's version floors.
  * `.gitignore`: combined both, so `.env`, `outputs/` and model weights are ignored. A's version did **not** ignore `.env`.
  * Also `scripts/make_mocks.py`: `CFG` → `cfg()`.
* `scripts/download_sc_aws.py` (your untracked file) is still untracked and not committed.
* `docs/NyayaSethu_Review_1.docx` wasn't in the repo. I copied it from `~/Downloads` (not committed yet).

## 1. What exists and what works (verified)

| Component | Status | Evidence |
|---|---|---|
| Test suite | **9/9 pass** on the merged tree (102 s) | `pytest -q`. I first had to install `pytest`, `sentence-transformers`, `faiss-cpu` and `pypdf` into `.venv`; they're in A's requirements but weren't installed |
| `stage0_corpus.download --skip-pdfs` | Works | Fetched AILA (20.7 MB), Bhattacharya (6.6 MB), and 76 years of SC metadata parquet (43,333 rows, 1950–2025) into `data/raw/` |
| SC PDFs on disk | **4,994 PDFs, 2018–2023** in `data/raw/sc_aws/year=Y/pdf/` (from your script) | per year: 2018: 795, 2019: 1,050, 2020: 571, 2021: 708, 2022: 1,017, 2023: 854. A's loader expects `data/raw/sc-judgments/pdf/`. I didn't move anything; the test run used a symlink in scratch |
| `stage0_corpus.build_corpus` on real SC PDFs | **Runs, but metadata never attaches** (see §3.1) | 300-PDF run in scratch: 300 judgments, 70,141 sentences; date from metadata 0%; bench from metadata 0%; titles 0/300; aliases 0/300; **graph edges 0** |
| Committed `data/judgments.jsonl` | Only the **50 Bhattacharya judgments** (9,380 sentences) | `logs/corpus_build.json`: by_source = Bhattacharya2019: 50, edges = 0. The AWS SC corpus has not been built or committed |
| Committed `data/segments.parquet` | Bhattacharya **gold** roles for those 50 docs, `conf = 1.0` everywhere | 9,380 rows; roles identical to the gold roles in `judgments.jsonl`. No Stage 1 model output exists yet |
| Stage 1 | LSA-feature BiLSTM-CRF, 5-fold CV on Bhattacharya: macro-F1 0.530 ± 0.029 (quoted from `logs/stage1_cv_lsa.json` / README; I didn't re-run it) | No `models/stage1/` checkpoint in the repo. The InLegalBERT run is pending on Kaggle. **No Issue role yet** |
| Stages 2–3, FastAPI | Code complete; tested only on the fictional fixtures (`tests/fixtures/`) | No real index or models; `app.engine.Engine` raises an error without `models/index_corpus/` |
| `app/b_interface.py` | A's hook for B: `from validity import get_validity`, `from summary import summarize`; returns `{"status": "pending"}` until those exist | Top-level packages `validity/` and `summary/`, **not** `stage4_validity/`/`stage5_summary/` |
| `stage0_corpus/citations.py`, `graph.py` | A already has a citation regex (AIR, SCC ×2, SCR, SCC OnLine, INSC, HC neutral, JT, SCALE, MANU, ILR, CriLJ), a case-name regex, an exact-key resolver, a networkx graph with PageRank and stats | Overlaps B's Stage 0 graph work; see §6 |

## 2. Dataset choices A made (vs Review 1)

| Review 1 | A's replacement | A's stated reason | Effect on B |
|---|---|---|---|
| LegalSeg (7,120 docs, 7 roles incl. Issue) | Bhattacharya 2019: 50 SC docs, 7 roles, MIT; plus about 20 hand-annotated SC docs (not done yet) | HF gated | Role names differ (§3.3). **No Issue role** in the open data, so Stage 5 `issue` has no source until A's annotation and Stage 1 run |
| IL-PCR (IL-TUR) | SC-PCR, built by `build_sc_pcr.py` from the SC corpus (citation-masked, year-split, disjoint pools) | HF gated | SC-PCR query texts are masked; **B must build the graph from `judgments.jsonl`**, not from SC-PCR. Not comparable to published IL-PCR |
| AILA 2019 | unchanged | already open | none |
| ILDC (~35K) | AWS Indian Supreme Court Judgments, CC BY 4.0, PDFs + metadata | access form | Main corpus. Its metadata has title, petitioner, respondent, judge, SCR citation, INSC neutral citation (`nc_display`), CNR, decision date |
| OpenNyAI (precedent-treatment roles for Stage 4) | none; A's docs suggest keyword weak labels plus 300–500 hand labels | gated | As in your brief. **But see §5.1**: the SCR PDFs contain editor-assigned treatment labels |
| IN-Abs / IN-Ext | unchanged (Zenodo 10.5281/zenodo.7152317) | open | none |

## 3. Schema mismatches: A's actual files vs `data/CONTRACT.md`

`CONTRACT.md` came from your Phase 0 and A never saw it; A's contract is `common/schema.py` plus `data/README.md`. Following your rule, B follows **A's actual schema**, and I'll update `CONTRACT.md` at the start of Phase 1 and note the change.

| File | CONTRACT.md | A's actual | Resolution |
|---|---|---|---|
| judgments.jsonl | `{case_id, court, date YYYY-MM-DD, bench_size, text, sentences:[str]}` | `{case_id, source, court, date, bench_size, title, aliases, text, sentences:[{idx, text, role, conf}]}` | Adopt A's. `court` is `"SC"`, not "Supreme Court". `date` is **year only** ("1953") for Bhattacharya, and only partly ISO for SC (§3.1) |
| segments.parquet | `case_id, sent_idx, text, role, conf` | same columns | matches; **role values differ** (§3.3) |
| case_id | `<COURT>_<YEAR>_<SEQ>`, e.g. `SC_1963_001` | `bhatt:<stem>`, `aila:<stem>`, `sc:<cnr or pdf stem>` | Adopt A's. See §4 |
| pagerank.csv | `case_id, score` | `stage3_rerank/authority.py` reads `r["pagerank"]` (and `case_id`/`doc_id`) | **Conflict.** A `score`-only file would crash A's code (KeyError). B will write `case_id,pagerank`; contract to be updated |
| edges.csv / pagerank.csv location | `data/edges.csv`, `data/pagerank.csv` (your brief) | `data/from_b/edges.csv`, `data/from_b/pagerank.csv`, **gitignored** by A | Kept A's paths so A's code works without changes. **Decision needed** (§7, Q3) |
| validity.json / summary.json | defined in CONTRACT.md | A consumes these only via `b_interface` (dict per call); no file schema in A's code | Keep CONTRACT.md's schemas |
| data/mock/* | `SC_1963_001` IDs, `court: "Supreme Court"`, `sentences: [str]`, contract roles | n/a | **Mocks don't match A's real schema.** Regenerate in Phase 1 (A's ID format, sentence dicts, Bhattacharya roles) |

### 3.1 Bug in A's SC loader (the main blocker for B)

`stage0_corpus/loaders.py`: `load_sc_metadata` keys metadata by `Path(path).stem`, which gives `2018_10_1005_1038`. `load_sc_judgments` looks it up with the PDF stem, which is `2018_10_1005_1038_EN`. The `_EN` suffix means **no lookup ever succeeds**. Measured on 300 PDFs: 0/300 matched as written, and **300/300 match once `_EN` is stripped**.

Effects as written:
* `title` and `aliases` are empty, so A's resolver can't resolve citations, and the graph has **0 edges**.
* `case_id` falls back to the PDF stem (`sc:2018_10_1005_1038_EN`) instead of the CNR (`sc:ESCR010000012018`).
* Dates come from A's text regex instead: found for only 51.7%, and some are wrong. For example, 2018 judgments got `2016-09-27` and `1975-09-03`, which are dates of lower-court orders or cited cases.

Two more problems would show up once the lookup is fixed:
* Metadata `decision_date` is `DD-MM-YYYY` (100% of 43,333 rows). A takes `[:10]` as-is, so `date` would be non-ISO.
* Bench size from metadata `judge` gives 1 for 285 of 300 docs. The `judge` field lists more than one name in only 11.7% of all rows, so it is mostly a single judge, not the bench. Because metadata is filled first, A's regex fallback never runs. The SCR PDF header carries the bench (`[R. F. NARIMAN AND INDU MALHOTRA, JJ.]`), found in 300/300 docs: 2 judges in 227, 3 in 31, 4 in 38, 6 in 3, 5 in 1. Some of the 4s and 6s are probably parse errors from my quick regex; to be checked in Phase 1.

### 3.2 Other observations
* `nc_display` (INSC neutral citation, e.g. `2018INSC696`) is in 99.6% of metadata rows but isn't used as an alias by A's loader.
* About 8.3k "SUPP." SCR citations (`[1959] SUPP. 1 S.C.R. 1`) don't parse with A's SCR regex.
* The SCR PDFs start with a **headnote** (293/300 have "held:" in the first 6,000 characters), then Case Law Reference, counsel and only then the judgment. A splits this front matter into sentences as if it were judgment text.

### 3.3 Role names

| Contract / Stage 5 field | A's roles (Bhattacharya + planned Issue) |
|---|---|
| Facts | Facts (+ Ruling by Lower Court as procedural history) |
| Issue | Issue: **only from A's planned hand-annotation**; not in any data yet |
| Arguments of Petitioner / Respondent | Argument (not split by party) |
| Reasoning | Ratio of the decision (+ Precedent, Statute as supporting) |
| Decision | Ruling by Present Court |

The proposed Stage 5 mapping is in §6.

## 4. case_id format (B must use the same)

* Bhattacharya: `bhatt:<file stem>`, e.g. `bhatt:1953_L_1`
* AILA: `aila:<doc stem>`, e.g. `aila:C1`
* AWS SC: `sc:<cnr>` when metadata matches, otherwise `sc:<pdf stem>`. **Because of §3.1, the IDs A's code produces today are `sc:2018_10_1005_1038_EN`; after a fix they would be `sc:ESCR010000012018`.** The format isn't stable until A fixes the loader and rebuilds. B's code will take every ID from `judgments.jsonl`, so it follows whatever A produces. But edges, validity and pagerank must be regenerated after any rebuild.

## 5. Deviations from Review 1

| # | Deviation | Reason |
|---|---|---|
| D1 | LegalSeg → Bhattacharya 2019 (50 docs) plus own annotation | LegalSeg gated |
| D2 | IL-PCR → SC-PCR (own, citation-masked) | gated; numbers not comparable to published IL-PCR |
| D3 | ILDC → AWS SC judgments (CC BY 4.0) | access form |
| D4 | OpenNyAI distant supervision → corpus-derived weak labels plus hand-labelled gold | gated |
| D5 | Stage 1 majority baseline is 38.6% (Ratio), not Review 1's 50% (LegalSeg None share) | different dataset (A's measurement) |
| D6 | No Issue label in open data; Stage 1 gate "not passed" | Bhattacharya has no Issue role |
| D7 | Corpus so far: SC only (no High Courts) for 2018–2023 PDFs, with metadata back to 1950. Review 1 scope included HC | AWS dataset is SC only |
| D8 | Indian Kanoon API baseline for Stage 4 (Review 1 §3.4) isn't planned by A or in your brief | needs an API key and paid tiers; I'll mark it out of scope unless you want it |
| D9 | Stage 5 training on Kaggle (no local GPU) | hardware |

### 5.1 An opportunity that changes Stage 4

216 of the 300 sampled SCR PDFs carry an editorial **"Case Law Reference"** block. Each entry is a cited citation, a treatment and a paragraph, for example `[2011] 11 SCR 870 / relied on / Para 37`. Treatment counts in the sample: referred to 1,008; relied on 472; followed 43; held inapplicable 40; considered 27; distinguished 26; approved 23; applied 16; **overruled 13**.

These are editor-assigned treatment labels. They are a far closer substitute for OpenNyAI's Precedent Relied / Not Relied / Overruled than keyword cues. The paragraph number links each label to the citing context.

Proposal: use them as the **primary weak-label source** (Review 1's distant-supervision design comes back), keep the cue rules as baseline (a) and a second label source, and still hand-label gold from the citing contexts. Risk: "referred to" (the majority) isn't followed/distinguished/overruled. It maps to neutral and gets dropped, which is the same as the "drop ambiguous" rule in your brief. The editorial labels mustn't be used as model input for the gold set, or there's leakage; I'll strip the Case Law Reference block from contexts.

## 6. Risks for B's stages

| Risk | Stage | Severity | Mitigation |
|---|---|---|---|
| R1. A's corpus isn't built and its SC loader drops all metadata (§3.1), so there are 0 edges, unstable case_ids, no titles or aliases | 0g, 4, 5 | **Blocker** | A fixes `loaders.py` (strip `_EN`, ISO-format `decision_date`, use `nc_display` as alias, bench from the SCR header) and rebuilds; or B makes a minimal listed fix with A's agreement. **Q1** |
| R2. Resolution ceiling: SCC (2,992 of 5,946 mentions in the 300-doc sample) and AIR citations can't be resolved from metadata, which carries only SCR and INSC. Exact-key resolution resolves **32.2%** of mentions (SCR 76.5%, everything else 0%) | 0g, 4 | High | Fuzzy party-name matching with rapidfuzz against metadata titles; case names usually appear next to SCC cites ("X v. Y, (2009) 6 SCC 121"); the Case Law Reference block maps SCC to SCR for the same case in some docs. Report the rate honestly |
| R3. Most cited precedents are older than 2018 and not in the 2018–2023 PDF corpus | 0g, 4 | High | Make graph nodes = all 43,333 metadata cases (the IDs follow A's `sc:<cnr>` rule), so 2018–2023 citing contexts can label older precedents. Retrieval stays on the downloaded corpus. Validity is reported for cited cases, and the UI shows it when a retrieved case has a verdict. Cited-case bench for pre-2018 cases is only reliable from the metadata `judge` field when it lists several names. **Q2** |
| R4. Bench size is unreliable (metadata `judge` is usually a single name) | 4 | High | Use the SCR bracket header for downloaded PDFs; where the cited bench is unknown, use w_bench = 1 and treat "overruled" as requiring a known cited bench, otherwise "doubted"; report the share of unknowns |
| R5. Overruled is rare (13 of ~1.7k treatments in the sample) | 4 | High | Class weights; report per-class F1; oversample overruled in the gold export; measure coverage on the full corpus in Phase 1 |
| R6. No Issue role exists; Stage 1 isn't trained, so `segments.parquet` holds only the 50 Bhattacharya docs | 5 | High | `issue` falls back to an "Issue" cue regex ("the question ... is whether") over Facts/Ratio sentences until A's model exists, and this is documented. Segments for SC docs depend on A's Kaggle run |
| R7. SCR headnotes are inside `text`; summarizing them leaks the editorial summary (and ROUGE inflates) | 5 | Medium | Strip the front matter (headnote, Case Law Reference, counsel) before summarization; ask A to do the same in Stage 0, or B filters by position |
| R8. `b_interface` imports top-level `validity`/`summary` | 4–5 int. | Low | Phase 4: add thin `validity/__init__.py` and `summary/__init__.py` that re-export from `stage4_validity`/`stage5_summary`. No change to A's files |
| R9. `pagerank.csv` column name (§3) | 0g / 3 | Low | B writes `case_id,pagerank` |
| R10. A's graph code duplicates B's scope | 0g | Low | B will **reuse** A's `extract_citations`, `CitationResolver` and `CitationGraph` (importing, not copying), and add INSC/SUPP. handling, fuzzy resolution, sentence-level context and B's outputs in `stage0_graph/`. A's files stay unchanged |
| R11. Corpus size: 4,994 PDFs is about 1.2M sentences; full InLegalBERT training is CPU-bound locally | 4 | Medium | Train the classifier on Kaggle as well (notebook), or a smaller sample locally |
| R12. Legal-LED fine-tuning requires Kaggle; IN-Abs length is 8k+ tokens | 5 | Medium | as planned (notebook + local inference) |

## 7. Decisions I need from you before Phase 1

* **Q1, the loader fix (R1).** Either (a) you or A fix `stage0_corpus/loaders.py` and rebuild `judgments.jsonl`, or (b) I make a minimal fix on `member-b`, listed in the final report: strip `_EN`, ISO date, `nc_display` alias, SCR-header bench. Then I build the SC corpus locally from the 4,994 PDFs, with `case_id = sc:<cnr>`, and you tell A. I recommend (b), confirmed with A, because nothing in B works otherwise.
* **Q2, graph scope (R3).** Should the nodes include all 43,333 metadata cases (citing edges only from downloaded PDFs)? I recommend yes.
* **Q3, hand-off paths.** Keep `data/from_b/edges.csv` and `data/from_b/pagerank.csv` (A's paths, gitignored), or un-ignore them so they're shared through git? `edges.csv` may run to tens of MB. I recommend keeping A's paths and committing only `pagerank.csv` (small).
* **Q4, Stage 4 labels (§5.1).** Use the editorial Case Law Reference treatments as the primary weak labels, with cue rules as baseline and secondary source? I recommend yes.
* **Q5.** Should I also commit `docs/NyayaSethu_Review_1.docx` and `scripts/download_sc_aws.py`? (Your raw PDFs stay in `data/raw/sc_aws/`, gitignored.)
