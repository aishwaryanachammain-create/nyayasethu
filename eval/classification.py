"""Classification metrics. Macro-F1 is the headline (half of LegalSeg is 'None', so the
majority-class accuracy baseline is ~50%); every table carries the per-class breakdown."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def classification_report(gold: Sequence[str], pred: Sequence[str], labels: List[str]) -> Dict:
    present = [l for l in labels if l in set(gold) | set(pred)]
    p, r, f, s = precision_recall_fscore_support(gold, pred, labels=present, zero_division=0)
    per = {l: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
           for i, l in enumerate(present)}
    gold_present = [l for l in present if per[l]["support"] > 0]
    macro = sum(per[l]["f1"] for l in gold_present) / max(len(gold_present), 1)
    maj_label, maj_n = Counter(gold).most_common(1)[0] if gold else ("", 0)
    return {
        "macro_f1": macro,
        "accuracy": sum(g == q for g, q in zip(gold, pred)) / max(len(gold), 1),
        "majority_baseline_accuracy": maj_n / max(len(gold), 1),
        "majority_label": maj_label,
        "per_class": per,
        "confusion": {"labels": present, "matrix": confusion_matrix(gold, pred, labels=present).tolist()},
        "n": len(gold),
    }


def acceptance_gate(report: Dict, min_issue_recall: float, min_reasoning_recall: float) -> Dict:
    """Stage-2 prerequisite: Issue and Reasoning recall must clear thresholds."""
    pc = report["per_class"]
    ir = pc.get("Issue", {}).get("recall", 0.0)
    rr = pc.get("Reasoning", {}).get("recall", 0.0)
    return {"issue_recall": ir, "reasoning_recall": rr,
            "passed": ir >= min_issue_recall and rr >= min_reasoning_recall}


def format_report(rep: Dict) -> str:
    lines = [f"macro-F1 {rep['macro_f1']:.4f} | accuracy {rep['accuracy']:.4f} | "
             f"majority baseline ({rep['majority_label']}) {rep['majority_baseline_accuracy']:.4f} | n={rep['n']}",
             f"{'class':28s} {'P':>6s} {'R':>6s} {'F1':>6s} {'n':>6s}"]
    for l, m in rep["per_class"].items():
        lines.append(f"{l:28s} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f} {m['support']:6d}")
    return "\n".join(lines)


def acceptance_gate_classes(report: Dict, classes, min_recall: float) -> Dict:
    """Step 17 gate: every listed class must reach `min_recall`. Classes absent from the gold
    labels cannot pass (they are reported as 'not measurable', never silently skipped)."""
    pc = report["per_class"]
    res = {}
    for c in classes:
        if c not in pc or pc[c]["support"] == 0:
            res[c] = {"recall": None, "status": "not measurable: no gold examples"}
        else:
            r = pc[c]["recall"]
            res[c] = {"recall": round(r, 4), "status": "pass" if r >= min_recall else "FAIL"}
    return {"threshold": min_recall, "classes": res, "passed": all(v["status"] == "pass" for v in res.values())}
