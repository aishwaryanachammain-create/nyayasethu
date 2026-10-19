# Labelling `eval/gold_to_label.csv` (Stage 4 gold set)

> **Current state (2026-10-19):** the `label` column holds values **copied from `suggested_label`** (machine
> suggestions), not human labels; 60 rows are blank. `assistant_label` holds an AI review. Neither is
> human-verified. Replace `label` with your own judgement before calling any result "gold".

There are 400 rows. Each row is one place where a judgment (`citing_id`) cites an earlier case (`cited_id`, `cited_title`).
In `sentence`, the citation is marked with `[[ ]]`. `context` is the sentence with two sentences either side.

## Reviewing the AI pre-labels (fastest way)

`assistant_label` holds a label proposed by an AI assistant (Claude) after reading each row's `sentence` and
`context`, **without looking at `suggested_label`**. `assistant_confidence` (high/low) and `assistant_note`
give its reason. **These are not gold.** Only what a person writes in `label` is scored.

For each row, read the sentence and context, then:
* if you agree, copy `assistant_label` into `label`;
* if you don't, write your own label (and a word in `notes`).

Start with rows where `check_first` is filled in: low confidence, disagreement with `suggested_label`, or a
note asking you to check. In the report, describe the gold set as "pre-annotated by an AI assistant, verified
and corrected by a human".

## Labels

Fill in the **`label`** column with exactly one of:

| label | Use it when the citing court itself… |
|---|---|
| `followed` | relies on, applies, follows, approves or affirms the cited case's rule |
| `distinguished` | says the cited case does not apply here: different facts, or a different provision or question |
| `overruled` | overrules it, or holds it is not good law, wrongly decided or per incuriam |
| `neutral` | only mentions, quotes or summarises it, or a party or the lower court relies on it, with no treatment by this court |
| `unclear` | you cannot decide from the text (these rows are left out of scoring) |

## Rules

* **Judge the citing court's own treatment, not a party's.** "Counsel relied on X" is `neutral`, unless the court then accepts and applies X.
* **Watch out for treatments of other cases.** "X was overruled in Y" means *Y* overruled X. If the marked citation is X and the citing court is only reporting this, the label is `neutral`.
* **The `suggested_label` column is a machine guess.** It comes from the SCR editors' notes or from keyword rules, and it is often wrong. Decide from the text, and don't copy it. It is never used as gold.
* **Use `notes` for anything odd**, e.g. a citation that points to the wrong case. If the citation is wrong, add a note and label the row `unclear`.
* **When you're done,** run `python -m stage4_validity.evaluate`. Gold results are computed only from rows with a valid `label`.

If two people label, give each a copy and keep both copies. Agreement (Cohen's kappa) can then be reported as well.
