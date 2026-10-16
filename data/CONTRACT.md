# Data Contract — do not change without both members agreeing

| File | Owner | Schema |
|---|---|---|
| judgments.jsonl | A | {case_id, court, date (YYYY-MM-DD), bench_size (int or null), text, sentences: [str]} |
| segments.parquet | A | case_id, sent_idx (int), text, role, conf (float) |
| edges.csv | B | citing_id, cited_id, citing_sent_idx, context |
| pagerank.csv | B | case_id, score |
| retrieval.json | A | [{query_id, results: [{case_id, score, matched_segments: [{sent_idx, role}]}]}] |
| validity.json | B | [{case_id, label, confidence, n_citing, evidence: [{citing_id, sent, treatment, weight}]}] |
| summary.json | B | [{query_id, case_id, issue, material_facts, ratio, decision, relevance_to_query, rejected: [str]}] |

Roles: Facts, Issue, Arguments of Petitioner, Arguments of Respondent, Reasoning, Decision, None
Validity labels: followed, distinguished, overruled, doubted, insufficient_evidence
case_id format: <COURT>_<YEAR>_<SEQ>, e.g. SC_1963_001