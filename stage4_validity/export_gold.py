"""Export ~400 citing contexts for hand labelling -> eval/gold_to_label.csv.

Stratified by weak label so the rare classes are represented (the natural distribution has very few
distinguished/overruled mentions). The `label` column is BLANK and must be filled by a person.
`suggested_label` is the weak label, given only to speed labelling: it is never used as gold.
At most 2 rows per citing judgment, one row per (citing, cited) pair. Rows are shuffled so the
stratum is not visible from the order.

Training (stage4_validity.train) drops every (citing, cited) pair that appears here.

    python -m stage4_validity.export_gold
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.config import ROOT

from .dataset import out_paths

GOLD = ROOT / "eval" / "gold_to_label.csv"
STRATA = {"followed": 110, "distinguished": 100, "overruled": 60, "neutral": 70, "unlabelled": 60}


def sample(ed: pd.DataFrame, nodes: pd.DataFrame, strata=STRATA, seed: int = 13) -> pd.DataFrame:
    ed = ed[ed.sentence.notna()].copy()
    ed["stratum"] = ed.weak_label.fillna("unlabelled")
    ed = ed.sample(frac=1.0, random_state=seed)
    # prefer editorial labels inside each class stratum (they are the cleaner weak label)
    ed["pri"] = (ed.weak_source != "editorial").astype(int)
    ed = ed.sort_values("pri", kind="stable")
    picked, per_doc, pairs = [], {}, set()
    for st, n in strata.items():
        k = 0
        for r in ed[ed.stratum == st].itertuples():
            if k >= n:
                break
            if (r.citing_id, r.cited_id) in pairs or per_doc.get(r.citing_id, 0) >= 2:
                continue
            pairs.add((r.citing_id, r.cited_id))
            per_doc[r.citing_id] = per_doc.get(r.citing_id, 0) + 1
            picked.append(r.Index)
            k += 1
    g = ed.loc[picked].sample(frac=1.0, random_state=seed + 1)
    title = nodes.set_index("case_id").title
    out = pd.DataFrame({
        "gold_id": [f"g{i:03d}" for i in range(1, len(g) + 1)],
        "mention_id": g.mention_id.values,
        "citing_id": g.citing_id.values, "cited_id": g.cited_id.values,
        "cited_title": g.cited_id.map(title).values,
        "cite_text": g.cite_text.values,
        "sentence": g.sentence.values, "context": g.context.values,
        "label": "", "notes": "",
        "suggested_label": g.weak_label.fillna("").values,
        "suggestion_source": g.weak_source.fillna("").values,
    })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(GOLD))
    p.add_argument("--force", action="store_true", help="overwrite an existing file (it may hold your labels)")
    a = p.parse_args()
    from pathlib import Path
    if Path(a.out).exists() and not a.force:
        raise SystemExit(f"{a.out} exists (it may contain hand labels); pass --force to overwrite")
    mp, np_ = out_paths()
    out = sample(pd.read_parquet(mp), pd.read_parquet(np_))
    out.to_csv(a.out, index=False)
    print(f"{len(out)} rows -> {a.out}")
    print(out.suggested_label.replace("", "unlabelled").value_counts().to_string())


def gold_pairs():
    """(citing, cited) pairs in the gold file, excluded from training."""
    if not GOLD.exists():
        return set()
    g = pd.read_csv(GOLD)
    return set(zip(g.citing_id, g.cited_id))


if __name__ == "__main__":
    main()
