"""NyayaSethu demo UI (Member B, Phase 4).

    streamlit run ui/app.py                    # real data, Member A's engine in-process
    streamlit run ui/app.py -- --mock          # contract mocks (data/mock), no models or corpus needed
    streamlit run ui/app.py -- --api http://127.0.0.1:8000     # Member A's FastAPI server

Mock mode also follows `use_mock: true` in config.yaml.
"""
import argparse
import html as _html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_p = argparse.ArgumentParser()
_p.add_argument("--mock", action="store_true")
_p.add_argument("--api")
ARGS, _ = _p.parse_known_args(sys.argv[1:])

from ui import core  # noqa: E402

if ARGS.mock:
    core.use_mock_config()

import streamlit as st  # noqa: E402

from common.config import cfg  # noqa: E402

MOCK = bool(cfg().get("use_mock"))
BACKEND = "mock" if MOCK else ("api" if ARGS.api else "engine")
ROLE_COLOURS = {"Facts": "#2563eb", "Ruling by Lower Court": "#7c3aed", "Issue": "#dc2626", "Argument": "#ea580c",
                "Statute": "#0891b2", "Precedent": "#0d9488", "Ratio of the decision": "#16a34a",
                "Ruling by Present Court": "#4b5563"}           # same colours as Member A's /case endpoint

st.set_page_config(page_title="NyayaSethu", layout="wide")


@st.cache_resource(show_spinner="Loading Member A's retrieval engine ...")
def get_engine():
    from app.engine import Engine
    return Engine(load_dense=True)


@st.cache_data(show_spinner=False, max_entries=64)
def abstractive(query: str, case_id: str):
    from stage5_summary import summarize
    return summarize(query, case_id, mode="abstractive")


@st.cache_data(show_spinner=False, max_entries=32)
def judgment_roles(case_id: str):
    from stage5_summary.docs import get_judgment, roles_for
    d = get_judgment(case_id)
    return roles_for(d) if d else ([], "not_found")


def search(query: str, k: int):
    if BACKEND == "mock":
        return core.add_member_b(core.mock_search(query, k), query)
    if BACKEND == "api":
        return core.api_search(ARGS.api, query, k)
    return core.engine_search(get_engine(), query, k)


def show_summary(s, key):
    if not s or s.get("status") != "ok":
        st.caption(f"Summary: {(s or {}).get('status', 'pending')}. {(s or {}).get('note', '')}")
        return
    for f, name in core.FIELD_NAMES.items():
        st.markdown(f"**{name}.** {s.get(f) or '_(nothing in the source for this field)_'}")
    meta = f"generator: {s.get('generator')} · roles: {s.get('roles_source')} · NLI: {s.get('nli')}"
    st.caption(meta)
    if s.get("rejected"):
        with st.expander(f"{len(s['rejected'])} generated sentence(s) removed by the NLI faithfulness filter"):
            for r in s.get("rejected_detail") or [{"field": "", "sentence": x} for x in s["rejected"]]:
                p = f" (P(entail) {r['p_entail']:.2f})" if "p_entail" in r else ""
                st.markdown(f"- *{r.get('field', '')}*: {r['sentence']}{p}")


def show_card(i, r, query):
    v = r.get("validity")
    text, colour, meaning = core.badge(v)
    with st.container(border=True):
        head, right = st.columns([5, 2])
        head.markdown(f"#### {i}. {r.get('title') or r['case_id']}")
        bits = [r.get("court"), r.get("date"), f"bench of {r['bench_size']}" if r.get("bench_size") else None,
                r["case_id"], f"score {r['score']}" if r.get("score") is not None else None]
        head.caption(" · ".join(str(b) for b in bits if b))
        with right:
            st.badge(text, color=colour)
            conf = (v or {}).get("confidence")
            st.caption(meaning + (f" (confidence {conf:.2f})" if isinstance(conf, (int, float)) else "")
                       + (f"; {v['n_citing']} citing judgments" if v and v.get("n_citing") is not None else ""))
        tabs = st.tabs(["Summary", "Validity evidence", "Judgment"])
        with tabs[0]:
            key = f"abs::{r['case_id']}"
            if key in st.session_state:
                show_summary(st.session_state[key], key)
                st.divider()
                st.caption("Extractive summary (role sentences copied from the judgment):")
            show_summary(r.get("summary"), r["case_id"])
            if key not in st.session_state and st.button("Abstractive summary (Legal-LED + NLI filter)",
                                                         key=f"btn::{r['case_id']}",
                                                         help="Runs Legal-LED on this judgment; minutes on a laptop"):
                with st.spinner("Generating with Legal-LED and checking each sentence with NLI ..."):
                    st.session_state[key] = abstractive(query, r["case_id"])
                st.rerun()
        with tabs[1]:
            if v and v.get("votes"):
                st.caption("Weighted votes: " + ", ".join(f"{k} {w:.2f}" for k, w in v["votes"].items()))
            rows = core.evidence_rows(v)
            if not rows:
                st.caption("No citing sentences in the citation graph.")
            for e in rows:
                src = f", {e['source']}" if e.get("source") else ""
                st.markdown(f"- **{e.get('treatment')}**{src} · weight {e.get('weight')} · `{e.get('citing_id')}`  \n"
                            f"  {e['sent_md']}")
        with tabs[2]:
            if st.toggle("Show role-labelled text", key=f"jt::{r['case_id']}"):
                sents, src = judgment_roles(r["case_id"])
                st.caption(f"roles: {src}")
                html = " ".join(f'<span style="border-left:3px solid {ROLE_COLOURS.get(role, "#9ca3af")};'
                                f'padding-left:4px" title="{role}">{_html.escape(t)}</span>' for t, role in sents[:400])
                st.markdown(f'<div style="line-height:1.7">{html}</div>', unsafe_allow_html=True)


with st.sidebar:
    st.title("NyayaSethu")
    st.caption("Validity-aware precedent retrieval for Indian case law")
    st.markdown(f"**Mode:** {'mock data' if MOCK else 'real data'}  \n**Retrieval:** {BACKEND}")
    k = st.slider("Results", 3, 20, 10)
    st.markdown("**Validity badges**")
    for lab, (t, c, m) in core.BADGES.items():
        st.badge(t, color=c)
        st.caption(m)
    if MOCK:
        st.info("Mock mode: retrieval returns the same fixed list for any query.")

st.header("Find precedents for your situation")
query = st.text_area("Describe your legal problem in plain language", height=100,
                     placeholder="My landlord kept my entire security deposit without showing any damage ...")
if st.button("Search", type="primary") and query.strip():
    with st.spinner("Searching ..."):
        try:
            st.session_state["result"] = search(query.strip(), k)
            st.session_state["query"] = query.strip()
        except FileNotFoundError as e:
            st.error(f"{e}. Or run the UI with `-- --mock`.")
        except Exception as e:            # API down, etc.
            st.error(f"Search failed: {e}")

out = st.session_state.get("result")
if out:
    comp = out.get("components") or {}
    st.caption(" · ".join(f"{a}: {b}" for a, b in comp.items()) +
               (f" · {out['timings']['total']}s" if (out.get("timings") or {}).get("total") else ""))
    if not out["results"]:
        st.warning("No results.")
    for i, r in enumerate(out["results"], 1):
        show_card(i, r, st.session_state["query"])
