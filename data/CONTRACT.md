# Data Contract — do not change without both members agreeing

Paths come from `config.yaml` via `common.config.path(key)`. With `use_mock: true`, every file below
resolves to its fictional stand-in in `data/mock/` (regenerate with `python scripts/make_mocks.py`).

| File (`path` key) | Owner | Schema |
|---|---|---|
| judgments.jsonl (`judgments`) | A | one object per line: `{case_id, source, court, date, bench_size, title, aliases, text, sentences: [{idx, text, role, conf}]}` (see `common/schema.py`) |
| segments.parquet (`segments`) | A | `case_id, sent_idx (int), text, role, conf (float)` |
| edges.csv (`edges`) | B | `citing_id, cited_id, citing_sent_idx, context` + `cite_text, method, score` |
| pagerank.csv (`pagerank`) | B | `case_id, pagerank, in_degree, in_corpus` |
| retrieval.json (`retrieval`) | A | `[{query_id, results: [{case_id, score, matched_segments: [{sent_idx, role}]}]}]` |
| validity.json (`validity`) | B | `[{case_id, label, confidence, n_citing, evidence: [{citing_id, sent, treatment, weight}]}]` |
| summary.json (`summary`) | B | `[{query_id, case_id, issue, material_facts, ratio, decision, relevance_to_query, rejected: [str]}]` |

**Field notes**

* `court`: `SC`, `HC` or `DC`.
* `date`: ISO `YYYY-MM-DD`, or only `YYYY` when the source gives just the year (Bhattacharya).
* `bench_size`: an int, or `null` when unknown.
* `aliases`: reporter citations that refer to this case, e.g. `[2019] 10 S.C.R. 133` or `2019INSC696`.
* `context`: the citing sentence ±2 sentences, with SCR page furniture removed.
* `method`: one of `key`, `key_range`, `name_year`, `name_only` (see `stage0_graph/resolve.py`).
* `score`: the match score from 0 to 100.

**case_id** (Member A's rule, `stage0_corpus/loaders.py`):
* `sc:<CNR>` for AWS Supreme Court judgments, e.g. `sc:ESCR010000012018`.
* `bhatt:<file stem>` for Bhattacharya 2019, e.g. `bhatt:1953_L_1`.
* `aila:<doc id>` for AILA 2019, e.g. `aila:C1`.

The citation graph also has nodes for metadata-only SC cases. Those have no text and use the same `sc:<CNR>` form.

**Roles** (Member A, `config.yaml` `stage1.labels`):
* Facts
* Ruling by Lower Court
* Issue
* Argument
* Statute
* Precedent
* Ratio of the decision
* Ruling by Present Court

**Validity labels:** followed, distinguished, overruled, doubted, insufficient_evidence.

## Change log

* **2026-10-16 (Member B, Phase 1).** Aligned this file with Member A's actual files, as the rule says to follow A's real schema.
  * judgments: added `source`, `title`, `aliases`. `sentences` are objects rather than strings. `court` is a code. `date` may be year-only.
  * case_id: replaced the earlier `<COURT>_<YEAR>_<SEQ>` with A's prefixes.
  * pagerank: the column is `pagerank`, not `score`, because Stage 3 `authority.py` reads `pagerank`. Added `in_degree` and `in_corpus`.
  * edges: added `cite_text`, `method` and `score` after the four agreed columns.
  * roles: replaced the earlier role list with A's labels. The Stage 5 mapping is in `stage5_summary/`.
  * edges/pagerank live at A's paths `data/from_b/` (gitignored).
  * The full corpus is at `data/processed/judgments.jsonl`, which is not in git.
