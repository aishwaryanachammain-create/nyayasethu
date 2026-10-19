"""IN-Ext role coverage: do our role-based field sources contain what legal experts extracted?

IN-Ext (Shukla et al. 2022) has, for 50 judgments (the same 50 as Bhattacharya et al. 2019, our
`bhatt:<stem>`), extractive summaries by two experts (A1, A2) split by segment: facts, analysis, argument,
judgement, statute. Field <-> expert segment:

    material_facts <-> facts      ratio <-> analysis      decision <-> judgement

(argument and statute have no field of their own; reported as not covered by design.)

Per expert segment we report
  coverage  share of the expert's sentences that are in our field's source sentences (each expert sentence is
            matched to the judgment sentence with the highest token Jaccard, >= MATCH);
  pool_share  share of the judgment's sentences in that source pool. Coverage only means something next to
            it: a pool of the whole judgment covers everything;
  rouge     ROUGE-1/2/L F1 of our extractive field text (ExtractiveGenerator, word budget) vs the expert segment.

Roles come from (a) the gold Bhattacharya labels (data/segments.parquet), the upper bound, and (b) the
position fallback used when no labeller is available. Member A's labeller (models/stage1) is trained on these
same 50 judgments, so its scores here would be in-sample; they are reported separately and marked as such.

    python -m stage5_summary.eval_inext          # -> eval/stage5_inext_results.json
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from statistics import mean

from common.config import ROOT, path

from .fields import field_sources
from .generate import ExtractiveGenerator

SEG_FIELD = {"facts": "material_facts", "analysis": "ratio", "judgement": "decision"}
MATCH = 0.5
OUT = ROOT / "eval" / "stage5_inext_results.json"


def toks(s: str):
    return re.findall(r"[a-z0-9]+", s.lower())


def expert_sentences(text: str):
    return [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if len(toks(s)) >= 3]


def match(expert: list, doc_toks: list) -> list:
    """Index of the best-matching judgment sentence per expert sentence (or None below MATCH)."""
    out = []
    for e in expert:
        et = set(toks(e))
        best, bj = None, 0.0
        for i, dt in enumerate(doc_toks):
            if not dt:
                continue
            j = len(et & dt) / len(et | dt)
            if j > bj:
                best, bj = i, j
        out.append(best if bj >= MATCH else None)
    return out


def position_roles(n: int):
    facts_end, dec_start = max(1, n // 4), max(n - 5, n // 4 + 1)
    return ["Facts" if i < facts_end else "Ruling by Present Court" if i >= dec_start else "Ratio of the decision"
            for i in range(n)]


def evaluate(role_sets: dict, root=None) -> dict:
    """role_sets: {name: {case_id: [(text, role)]}} -> results."""
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    root = root or path("raw") / "in_abs_ext" / "dataset" / "IN-Ext" / "summary" / "segment-wise"
    gen = ExtractiveGenerator()
    res = {}
    for name, docs in role_sets.items():
        cov, rouge, n_exp, n_matched = defaultdict(list), defaultdict(list), defaultdict(int), defaultdict(int)
        share = defaultdict(list)
        for cid, sents in docs.items():
            stem = cid.split(":", 1)[1]
            src, _ = field_sources(sents, "")
            doc_toks = [set(toks(t)) for t, _ in sents]
            for seg in ("facts", "analysis", "judgement", "argument", "statute"):
                fld = SEG_FIELD.get(seg)
                pool = {i for i, (t, _) in enumerate(sents) if fld and t in set(src[fld])}
                if fld:
                    share[seg].append(len(pool) / max(len(sents), 1))
                for ann in ("A1", "A2"):
                    f = root / ann / seg / f"{stem}.txt"
                    if not f.exists() or not f.read_text(errors="ignore").strip():
                        continue
                    exp = expert_sentences(f.read_text(errors="ignore"))
                    m = [i for i in match(exp, doc_toks) if i is not None]
                    n_exp[seg] += len(exp)
                    n_matched[seg] += len(m)
                    if m and fld:
                        cov[seg].append(sum(i in pool for i in m) / len(m))
                    if fld:
                        ours = gen.generate(fld, src[fld])
                        sc = rs.score(" ".join(exp), ours)
                        rouge[seg].append({k: v.fmeasure for k, v in sc.items()})
        res[name] = {seg: {"field": SEG_FIELD.get(seg, "(none: not covered by design)"),
                           "coverage": round(mean(cov[seg]), 4) if cov[seg] else None,
                           "pool_share": round(mean(share[seg]), 4) if share[seg] else None,
                           "n_summaries": len(cov[seg]),
                           "expert_sentences_matched": f"{n_matched[seg]}/{n_exp[seg]}",
                           **({f"{k}_f1": round(mean(r[k] for r in rouge[seg]), 4) for k in ("rouge1", "rouge2", "rougeL")}
                              if rouge[seg] else {})}
                     for seg in ("facts", "analysis", "judgement", "argument", "statute")}
    return res


def main():
    import pandas as pd
    from .docs import _labeller, get_judgment
    seg = pd.read_parquet(path("segments"))
    seg = seg[seg.case_id.str.startswith("bhatt:")]
    gold, pos, lab = {}, {}, {}
    labeller = _labeller()
    for cid, g in seg.groupby("case_id"):
        g = g.sort_values("sent_idx")
        gold[cid] = list(zip(g.text, g.role))
        pos[cid] = list(zip(g.text, position_roles(len(g))))
        if labeller is not None:
            d = get_judgment(cid)
            if d is not None:
                lab[cid] = [(s.text, s.role) for s in labeller.label([d])[0].sentences]
    sets = {"gold_roles": gold, "position_fallback": pos}
    if lab:
        sets["stage1_labeller_IN_SAMPLE"] = lab
    out = {"dataset": "IN-Ext segment-wise (A1, A2), 50 judgments", "match_jaccard": MATCH,
           "n_docs": len(gold), **evaluate(sets),
           "note": "stage1_labeller_IN_SAMPLE: Member A's labeller was trained on these 50 judgments; optimistic."}
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
