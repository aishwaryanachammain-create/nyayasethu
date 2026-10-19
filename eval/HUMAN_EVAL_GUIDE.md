# Human evaluation: generic vs query-focused summaries

`eval/human_eval_sheet.csv` has 25 rows. Each row is one user query and one judgment that Member A's engine
retrieved for it, with two summaries of that judgment, **A** and **B**, in random order:

* one is a generic summary of the whole judgment (Legal-LED fine-tuned on IN-Abs, no query);
* the other is NyayaSethu's query-focused summary (five fields, filtered by the NLI faithfulness check).

Which is which is in `eval/human_eval_key.csv`. **Raters must not open the key.** Give raters a copy of the
sheet only. The two summaries look different (one has five labelled fields), so the blinding hides the system
name, not the format; mention this in the report.

## How to rate

For each summary fill in a whole number from 1 (poor) to 5 (excellent):

| column | question |
|---|---|
| `*_faithful_1to5` | Is everything it says supported by the judgment? (open the judgment if unsure: the case title is given) |
| `*_relevant_1to5` | Does it tell this user what matters about this judgment **for their query**? |
| `*_useful_1to5` | Would it help the user decide whether to read the judgment? |

Then `preferred_A_B_tie`: write `A`, `B` or `tie`. `comments` is optional.

Leave a cell blank rather than guess. Two raters, rating independently, are better than one; save each
rater's copy under a different name (`human_eval_sheet_rater1.csv`, ...).

## Scoring

    python -m stage5_summary.eval_pairs --score eval/human_eval_sheet_rater1.csv [more sheets ...]
    # -> eval/human_eval_results.json (mean per system and criterion, preferences)

Until a sheet is filled in, the human evaluation is PENDING and nothing is reported.
