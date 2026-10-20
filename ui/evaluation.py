"""Evaluation page of the demo UI: every number comes from a committed log or eval file (paths shown under each)."""
from __future__ import annotations

from typing import Dict, List

from common.config import ROOT

INK, BAR = "#1f3a5f", "#3b5b8a"


def _load(rel: str) -> Dict:
    from ui.core import _load as load
    return load(rel)


def _bars(st, rows: List[Dict], x: str, y: str, fmt: str = ".2f", domain=(0, 1), height: int = 220):
    """Single-series horizontal bars: one hue, value labels at the bar end, tooltip on hover."""
    import altair as alt
    import pandas as pd
    df = pd.DataFrame(rows)
    base = alt.Chart(df).encode(y=alt.Y(f"{y}:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
                                x=alt.X(f"{x}:Q", title=None, scale=alt.Scale(domain=list(domain)),
                                        axis=alt.Axis(format=fmt, grid=True, gridOpacity=.35, tickCount=5)),
                                tooltip=[alt.Tooltip(f"{y}:N"), alt.Tooltip(f"{x}:Q", format=fmt)])
    bars = base.mark_bar(color=BAR, cornerRadiusEnd=4, height=16)
    text = base.mark_text(align="left", dx=4, color=INK, fontSize=12).encode(text=alt.Text(f"{x}:Q", format=fmt))
    st.altair_chart((bars + text).properties(height=height).configure_view(strokeWidth=0)
                    .configure_axis(domainColor="#d1d5db", labelColor="#4b5563"), width="stretch")


def _src(st, *files: str):
    st.caption("Source: " + ", ".join(f"`{f}`" for f in files))


def render(st):
    st.markdown('<div class="ns-page-title">How well does it work?</div>'
                '<div class="ns-page-sub">Every number on this page is read from the committed evaluation files.</div>',
                unsafe_allow_html=True)
    s1, s4 = _load("logs/stage1_cv_lsa.json"), _load("eval/stage4_results.json")
    inabs, nli = _load("eval/stage5_inabs_results.json"), _load("eval/stage5_nli_results.json")
    hand, prec = _load("eval/nli_check_results.json"), _load("eval/graph_precision.json")
    llm, human, inext = _load("eval/llm_only/results.json"), _load("eval/human_eval_results.json"), _load("eval/stage5_inext_results.json")
    import glob
    import json
    from pathlib import Path

    tabs = st.tabs(["Roles (Stage 1)", "Retrieval (Stages 2-3)", "Citation graph", "Validity (Stage 4)",
                    "Summaries (Stage 5)", "vs a plain LLM"])
    with tabs[0]:
        if s1:
            c = st.columns(3)
            c[0].metric("Macro-F1", f"{s1['macro_f1_mean']:.3f}", help=f"± {s1['macro_f1_std']:.3f} over {s1['cv_folds']} folds")
            c[1].metric("Accuracy", f"{s1['accuracy']:.3f}")
            c[2].metric("Majority baseline", f"{s1['majority_baseline_accuracy']:.3f}", help=s1["majority_label"])
            st.markdown(f"**F1 per rhetorical role**: BiLSTM-CRF on {s1['features'].upper()} features, "
                        f"{s1['n']:,} sentences from {s1['n_docs']} judgments")
            _bars(st, [{"role": k, "F1": v["f1"]} for k, v in s1["per_class"].items()], "F1", "role")
            _src(st, "logs/stage1_cv_lsa.json")
        else:
            st.info("logs/stage1_cv_lsa.json not found.")
    with tabs[1]:
        files = sorted(glob.glob(str(ROOT / "logs" / "results" / "*.json")))
        rows = []
        for f in files:
            m = (json.loads(Path(f).read_text()).get("metrics") or {})
            if m:
                rows.append({"run": Path(f).stem, **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}})
        if rows:
            st.markdown("**Retrieval runs**")
            st.dataframe(rows, hide_index=True, width="stretch")
            _src(st, *[f"logs/results/{Path(f).name}" for f in files])
        abl = ROOT / "logs" / "results" / "ablation.md"
        if abl.exists():
            st.markdown("**Ablation**")
            st.markdown(abl.read_text(encoding="utf-8"))
            _src(st, "logs/results/ablation.md")
        if not rows and not abl.exists():
            st.info("No retrieval results yet: run `python -m eval.ablation`.")
    with tabs[2]:
        if prec:
            e, r = prec.get("extraction") or {}, prec.get("resolution") or {}
            c = st.columns(3)
            c[0].metric("Citation extraction precision", f"{e.get('precision', 0):.1%}", help=f"{e.get('correct')}/{e.get('checked')} checked by hand")
            if r.get("precision") is not None:
                c[1].metric("Resolution precision", f"{r['precision']:.1%}", help=f"{r.get('correct')}/{r.get('checked')} checked by hand")
            bc = (e.get("bare_case_names") or {}).get("precision")
            if bc is not None:
                c[2].metric("Bare case names", f"{bc:.1%}", help="reporter citations: " + f"{(e.get('reporter_citations') or {}).get('precision', 0):.0%}")
            _src(st, "eval/graph_precision.json")
        img = ROOT / "eval" / "figures" / "indegree_distribution.png"
        if img.exists():
            st.image(str(img), caption="How often cases are cited (in-degree distribution)", width=560)
    with tabs[3]:
        gold = (s4 or {}).get("gold") or {}
        names = {"llm_zero_shot": "LLM zero-shot", "editorial_weak_label": "Editorial weak label",
                 "cue_rules": "Cue rules", "classifier": "Fine-tuned classifier"}
        rows = [{"method": names.get(k, k), "macro-F1": v["macro_f1_4class"], "key": k}
                for k, v in gold.items() if isinstance(v, dict) and "macro_f1_4class" in v]
        if rows:
            rows.sort(key=lambda r: -r["macro-F1"])
            st.markdown(f"**Treatment classification on the {gold.get('n_labelled')}-row human-reviewed gold set** "
                        "(4-class macro-F1)")
            _bars(st, [{k: v for k, v in r.items() if k != "key"} for r in rows], "macro-F1", "method", height=170)
            pick = st.selectbox("Confusion matrix", [r["method"] for r in rows])
            key = next(r["key"] for r in rows if r["method"] == pick)
            fig = ROOT / "eval" / "figures" / f"stage4_confusion_gold_{key}.png"
            if fig.exists():
                st.image(str(fig), width=460)
            _src(st, "eval/stage4_results.json", "eval/figures/")
        cov = (s4 or {}).get("validity_coverage") or {}
        if cov:
            st.markdown("**Verdicts across the citation graph**")
            bl = cov["by_label"]
            c = st.columns(4)
            for col, lab in zip(c, ("followed", "distinguished", "overruled", "insufficient_evidence")):
                col.metric(lab.replace("_", " ").capitalize(), f"{bl.get(lab, 0):,}")
    with tabs[4]:
        if inabs:
            st.markdown(f"**Legal-LED on IN-Abs ({inabs['n_docs']} test judgments): before and after fine-tuning**")
            c = st.columns(4)
            for col, m in zip(c, ("rouge1", "rouge2", "rougeLsum", "bertscore_f1")):
                ft, un = inabs["finetuned"][m], inabs["untuned"][m]
                col.metric(m.replace("rouge", "ROUGE-").replace("bertscore_f1", "BERTScore F1"), f"{ft:.3f}", f"{ft - un:+.3f}")
            _src(st, "eval/stage5_inabs_results.json")
        if nli:
            qf = nli.get("query_focused") or {}
            st.markdown("**Faithfulness filter (NLI)**")
            c = st.columns(3)
            c[0].metric("Generated sentences removed", f"{qf.get('rejection_rate', 0):.0%}", help=f"{qf.get('rejected')} of {qf.get('kept', 0) + qf.get('rejected', 0)}")
            if hand:
                c[1].metric("Removed ones truly unsupported", f"{hand['rejection_precision']['precision']:.0%}", help="blinded hand check")
                c[2].metric("Kept ones truly supported", f"{hand['kept_precision']['precision']:.0%}", help="blinded hand check")
            pf = qf.get("per_field") or {}
            if pf:
                _bars(st, [{"field": k.replace("_", " "), "rejection rate": v["rejection_rate"]} for k, v in pf.items()],
                      "rejection rate", "field", fmt=".0%", domain=(0, 1), height=180)
            _src(st, "eval/stage5_nli_results.json", "eval/nli_check_results.json")
        if (inext or {}).get("gold_roles"):
            st.markdown("**Role coverage against two experts' segment-wise summaries (IN-Ext)**")
            _bars(st, [{"expert segment": k, "coverage": v["coverage"]} for k, v in inext["gold_roles"].items()
                       if isinstance(v, dict) and "coverage" in v], "coverage", "expert segment", fmt=".0%", height=150)
            _src(st, "eval/stage5_inext_results.json")
        if human:
            st.markdown("**Human evaluation (mean rating, two raters, 50 summaries each)**")
            m = human["mean"]
            st.dataframe([{"summary": k, "faithful": m.get(f"{k}/faithful"), "relevant": m.get(f"{k}/relevant"),
                           "useful": m.get(f"{k}/useful"), "preferred": human["preferred"].get(k)}
                          for k in ("generic", "query_focused")], hide_index=True, width="stretch")
            _src(st, "eval/human_eval_results.json")
    with tabs[5]:
        from ui.llm_compare import totals
        t = totals()
        if llm and t:
            c = st.columns(3)
            c[0].metric("Cases a plain LLM named", t["named"], help=f"{llm['model']}, {llm['queries']} queries")
            c[1].metric("Confirmed to exist", f"{t['confirmed']} ({t['confirmed'] / t['named']:.0%})", help="after the hand check")
            c[2].metric("Labelled good law by the LLM", llm["llm_status"].get("good_law", 0))
            st.markdown("Without retrieval, the LLM names landmark cases from memory: most citations point to page 1 "
                        "of a volume, one query repeats the same case five times, and it calls almost everything good "
                        "law. NyayaSethu returns only judgments it holds in full text, with a validity verdict drawn "
                        "from how later judgments cite them.")
            st.caption("Not found is not proof of invention: the resolver cannot match every High Court or SCC citation.")
            _src(st, "eval/llm_only/results.json", "eval/llm_only/cases.csv", "eval/RESULTS_B.md §4.1")
