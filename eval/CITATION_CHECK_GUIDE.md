# Checking the citation-graph sample (`eval/citation_precision_sample.csv`)

300 rows from the Stage 0 citation graph:

* **150 `extraction` rows:** is `extracted` a real citation of a judgment in `sentence_or_context`?
  Fill `extraction_correct` with `yes` or `no`.
  * `yes` when the string identifies a cited judgment. A footnote citation counts. So does a name with a stray
    "In"/"See" or footnote number, as long as it names the right case.
  * `no` for the citing judgment's own running page header or title line, a name cut to the wrong parties,
    a name that runs into the next case, or a proceeding that is not a cited precedent.
* **150 `resolution` rows:** is `resolved_title` / `resolved_date` the case the text actually cites? Fill
  `resolution_correct` with `yes` or `no`. Leave it blank if you cannot tell.

`assistant_correct` and `assistant_note` are AI pre-labels to speed you up. **They are never scored.**

When done:

    python -m stage0_graph.precision      # -> eval/graph_precision.json (precision with 95% intervals)
