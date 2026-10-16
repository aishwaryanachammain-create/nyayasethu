"""Evaluate treatment classification.

Two clearly separated evaluations:
  1. weak_test: agreement with the SCR editors' labels on the held-out test split (by citing judgment).
     This measures agreement with a noisy weak label, NOT accuracy.
  2. gold: macro-F1 and confusion matrices against the hand labels in eval/gold_to_label.csv. Rows whose
     `label` is blank or not one of the four classes are ignored; with no labelled rows the gold result
     is PENDING. `suggested_label` is never read as gold.

Systems: cue rules (no cue or ambiguous -> neutral), the InLegalBERT classifier, the editorial weak label
itself (gold only, shows how noisy it is), and the zero-shot LLM (if eval/stage4_llm_preds.csv exists).
Headline metric: macro-F1 over the three treatment classes (followed, distinguished, overruled), as in the brief.

    python -m stage4_validity.evaluate      # -> eval/stage4_results.json, eval/figures/stage4_confusion_*.png
"""
from __future__ import annotations

import json
from typing import Dict, Sequence

import pandas as pd

from common.config import ROOT, path

from .dataset import out_paths
from .export_gold import GOLD
from .labels import LABELS, TREATMENTS
from .llm_baseline import OUT as LLM_PREDS

FIG = ROOT / "eval" / "figures"


def scores(y_true: Sequence[str], y_pred: Sequence[str]) -> Dict:
    from sklearn.metrics import confusion_matrix, f1_score
    y_pred = [p if p in LABELS else "neutral" for p in y_pred]
    per = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)
    return {"n": len(y_true),
            "macro_f1_3class": round(float(f1_score(y_true, y_pred, labels=TREATMENTS, average="macro", zero_division=0)), 4),
            "macro_f1_4class": round(float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)), 4),
            "per_class_f1": {l: round(float(f), 4) for l, f in zip(LABELS, per)},
            "support": {l: int(sum(t == l for t in y_true)) for l in LABELS},
            "confusion_matrix": {"labels": LABELS, "rows_true_cols_pred": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}}


def plot_cm(res: Dict, title: str, dst):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = res["confusion_matrix"]["rows_true_cols_pred"]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    for i, row in enumerate(cm):
        for j, v in enumerate(row):
            ax.text(j, i, v, ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    preds_p = out_paths()[0].parent / "validity_preds.parquet"
    preds = pd.read_parquet(preds_p) if preds_p.exists() else None
    ed = pd.read_parquet(out_paths()[0])
    out = {"note": "weak_test = agreement with editorial weak labels (not accuracy); gold = hand labels"}

    # 1. weak test split
    t = ed[(ed.split == "test") & ed.editorial.notna()]
    wt = {"cue_rules": scores(t.editorial.tolist(), t.cue.fillna("neutral").tolist())}
    if preds is not None:
        pm = preds.set_index("mention_id").pred
        tt = t[t.mention_id.isin(pm.index)]
        wt["classifier"] = scores(tt.editorial.tolist(), pm.loc[tt.mention_id].tolist())
    out["weak_test"] = wt
    for k, v in wt.items():
        plot_cm(v, f"{k} vs editorial labels (weak test, n={v['n']})", FIG / f"stage4_confusion_weak_{k}.png")

    # 2. gold
    if not GOLD.exists():
        out["gold"] = "PENDING: eval/gold_to_label.csv not exported"
    else:
        g = pd.read_csv(GOLD, keep_default_na=False)
        g["label"] = g.label.str.strip().str.lower()
        lab = g[g.label.isin(LABELS)]
        if lab.empty:
            out["gold"] = f"PENDING: 0 of {len(g)} rows hand-labelled"
        else:
            m = ed.set_index("mention_id")
            gd = {"n_labelled": len(lab), "n_rows": len(g), "gold_by_class": lab.label.value_counts().to_dict(),
                  "cue_rules": scores(lab.label.tolist(), m.loc[lab.mention_id].cue.fillna("neutral").tolist())}
            e = lab[lab.mention_id.map(m.editorial).notna()]
            if len(e):
                gd["editorial_weak_label"] = scores(e.label.tolist(), m.loc[e.mention_id].editorial.tolist())
            if preds is not None:
                pm = preds.set_index("mention_id").pred
                gd["classifier"] = scores(lab.label.tolist(), pm.reindex(lab.mention_id).fillna("neutral").tolist())
            if LLM_PREDS.exists():
                lp = pd.read_csv(LLM_PREDS, keep_default_na=False).set_index("gold_id")
                ll = lab[lab.gold_id.isin(lp.index)]
                if len(ll):
                    gd["llm_zero_shot"] = scores(ll.label.tolist(), lp.loc[ll.gold_id].llm_label.tolist()) | {
                        "model": lp.model.iloc[0]}
            for k in ("cue_rules", "classifier", "llm_zero_shot", "editorial_weak_label"):
                if k in gd:
                    plot_cm(gd[k], f"{k} vs gold (n={gd[k]['n']})", FIG / f"stage4_confusion_gold_{k}.png")
            out["gold"] = gd

    agg = path("logs") / "stage4_aggregate.json"
    out["validity_coverage"] = json.loads(agg.read_text()) if agg.exists() else "PENDING: aggregate not run"
    (ROOT / "eval" / "stage4_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: (v if isinstance(v, str) else {s: (r.get("macro_f1_3class") if isinstance(r, dict) else r)
                                                         for s, r in v.items()}) for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
