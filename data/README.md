# Data

Open-access only: no request forms, logins or gated downloads.

| Dataset | Licence | Source | Used for |
|---|---|---|---|
| Bhattacharya et al. 2019 — 50 Supreme Court judgments, 7 sentence roles | MIT | github.com/Law-AI/semantic-segmentation | Stage 1 training/evaluation |
| AILA 2019 (FIRE) — 50 queries, 2,914 candidates, qrels | CC BY 4.0 | Zenodo, DOI 10.5281/zenodo.4063986 | Stages 2-3 evaluation (comparable to published work) |
| Indian Supreme Court Judgments — ~35K English PDFs + metadata (1950–) | CC BY 4.0 | AWS Open Data, s3://indian-supreme-court-judgments | corpus, metadata, SC-PCR |
| SC-PCR — citation-masked prior-case-retrieval benchmark | CC BY 4.0 (derived) | built by `stage0_corpus/build_sc_pcr.py` | Stages 2-3 training + evaluation |
| InLegalBERT weights | MIT | huggingface.co/law-ai/InLegalBERT | Stage 1 features, bi-encoder, cross-encoder |

Files in this folder:

* `judgments.jsonl` — Week-2 hand-off to Member B. One judgment per line:
  `{case_id, source, court, date, bench_size, title, aliases, text, sentences:[{idx, text, role, conf}]}`
  (`role` is filled only where a dataset ships gold labels).
* `segments.parquet` — Week-4 hand-off to Member B: `{case_id, sent_idx, text, role, conf}`.
* `validation/` — hand-made files (splitter sample, role annotations).
* `raw/`, `processed/` — not in git; recreate with `python -m stage0_corpus.download` and the build scripts.

Once the full SC corpus is built, `judgments.jsonl` exceeds GitHub's 100 MB file limit. Share it
through Git LFS, a GitHub release asset or a Kaggle dataset, and keep only the code here.

<!-- AUTO-INSPECTION -->

## Dataset inspection (generated 2026-10-16)

### Bhattacharya et al. 2019 (MIT)
- judgments: 50; sentences: 9380; mean sentences/judgment: 188
- domains: {'Land & Property': 10, 'Constitutional': 9, 'Labour & Industrial Law': 8, 'Criminal': 16, 'Intellectual Property': 7}
- schema: one file per judgment, `sentence<TAB>role` per line (sentence-final periods already stripped)
- role distribution: Ratio of the decision 3624 (38.6%), Facts 2219 (23.7%), Precedent 1468 (15.7%), Argument 845 (9.0%), Statute 646 (6.9%), Ruling by Lower Court 316 (3.4%), Ruling by Present Court 262 (2.8%)
- majority-class baseline accuracy: 38.6% (Ratio of the decision)
- no Issue role, and arguments are not split by party: Issue comes from our own annotation
```json
[
 {
  "case_id": "bhatt:1953_L_1",
  "court": "SC",
  "date": "1953",
  "bench_size": null,
  "title": "",
  "n_sentences": 180,
  "text": "One Lakshminarayana Iyer, a Hindu Brahmin, who owned considerable properties in the Tirunelveli district, died on 13th December, 1924, leaving him surviving a widow Ranganayaki, and a married daughter Ramalakshmi Ramalakshmi had married the plaintiff and had a number of children from him They were a…"
 },
 {
  "case_id": "bhatt:1953_S_23",
  "court": "SC",
  "date": "1953",
  "bench_size": null,
  "title": "",
  "n_sentences": 151,
  "text": "This is an appeal on a certificate u/art 133(1) of theConstitution from a judgment and decree passed by the Appellate Bench of the High Court of Calcutta modifying on appeal the judgment and decree passed by Mr. Justice Bose on the original side of that court One Dwarka Nath Ghose was the owner of c…"
 },
 {
  "case_id": "bhatt:1954_M_25",
  "court": "SC",
  "date": "1954",
  "bench_size": null,
  "title": "",
  "n_sentences": 130,
  "text": "This appeal with special leave is directed against the judgment and order of the Labour Appellate Tribunal of India in a dispute regarding the workers' claim for bonus During the year 1948 the appellant made a profit of Rs 11,97,648-11-9 It paid 24 3 per cent dividend on ordinary shares, being the m…"
 }
]
```

