"""NyayaSethu demo UI (Member B, Phase 4).

    streamlit run ui/app.py                    # real data, Member A's engine in-process
    streamlit run ui/app.py -- --mock          # contract mocks (data/mock), no models or corpus needed
    streamlit run ui/app.py -- --api http://127.0.0.1:8000     # Member A's FastAPI server

Mock mode also follows `use_mock: true` in config.yaml. Before a demo, `python scripts/build_demo_cache.py`
saves results for the example queries: they are the fallback if the engine or API fails mid-demo.
"""
import argparse
import datetime as _dt
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
VALIDITY_STYLE = {"green": ("#dcfce7", "#166534", "#16a34a"), "orange": ("#ffedd5", "#9a3412", "#ea580c"),
                  "red": ("#fee2e2", "#991b1b", "#dc2626"), "gray": ("#f1f5f9", "#475569", "#94a3b8")}
FETCH = 30                                   # hits fetched per search; sidebar filters then keep the top k

st.set_page_config(page_title="NyayaSethu", page_icon="⚖️", layout="wide")
st.html(f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>")


@st.cache_resource(show_spinner="Loading Member A's retrieval engine ...")
def get_engine():
    from app.engine import Engine
    return Engine(load_dense=True)


@st.cache_resource(show_spinner="Loading the case resolver ...")
def get_resolver():
    from baseline_llm.run import load_resolver
    return load_resolver()


@st.cache_data(show_spinner=False)
def demo_cache():
    return core.demo_cache()


@st.cache_data(show_spinner=False, max_entries=64)
def abstractive(query: str, case_id: str):
    hit = core.cached_abstractive(query, case_id, demo_cache())
    if hit:
        return hit
    from stage5_summary import summarize
    return summarize(query, case_id, mode="abstractive")


@st.cache_data(show_spinner=False, max_entries=32)
def judgment_roles(case_id: str):
    from stage5_summary.docs import get_judgment, roles_for
    d = get_judgment(case_id)
    return roles_for(d) if d else ([], "not_found")


@st.cache_data(show_spinner=False, max_entries=128)
def neighbourhood(case_id: str, _validity):
    from ui.graph import neighbourhood as nb
    return nb(case_id, _validity)


def live_search(query: str, k: int):
    if BACKEND == "mock":
        return core.add_member_b(core.mock_search(query, k), query)
    if BACKEND == "api":
        return core.api_search(ARGS.api, query, k)
    return core.engine_search(get_engine(), query, k)


def search(query: str, k: int):
    """Live backend first; if it fails, the saved demo results, flagged in the UI (never a dead screen)."""
    try:
        return live_search(query, k)
    except Exception as e:                       # engine files missing, API down, ...
        out = core.cached_search(query, k, demo_cache())
        if out is None:
            raise
        out["fallback_error"] = f"{type(e).__name__}: {e}"
        return out


def abs_key(query: str, case_id: str) -> str:
    return f"abs::{query}::{case_id}"


# ---------------------------------------------------------------- pieces
def hero():
    tiles = "".join(f'<div class="ns-tile"><div class="ns-tile-v">{_html.escape(t["value"])}</div>'
                    f'<div class="ns-tile-l">{_html.escape(t["label"])}</div>'
                    f'<div class="ns-tile-s">{_html.escape(t["sub"])}</div></div>' for t in core.project_stats())
    st.markdown(f"""<div class="ns-hero"><div class="ns-hero-kicker">न्याय सेतु · a bridge to justice</div>
<div class="ns-hero-title">NyayaSethu</div>
<div class="ns-hero-sub">Describe your problem in plain language. We find the Indian precedents that match it,
tell you whether each one is <b>still good law</b>, and summarise only what the judgment actually says.</div>
<div class="ns-tiles">{tiles}</div></div>""", unsafe_allow_html=True)


def validity_block(v):
    text, colour, meaning = core.badge(v)
    bg, ink, bar = VALIDITY_STYLE.get(colour, VALIDITY_STYLE["gray"])
    conf = (v or {}).get("confidence")
    meter = (f'<div class="ns-meter"><div style="width:{conf * 100:.0f}%;background:{bar}"></div></div>'
             f'<div class="ns-meter-l">confidence {conf:.2f}</div>') if isinstance(conf, (int, float)) else ""
    n = (v or {}).get("n_citing")
    cites = f'<div class="ns-meter-l">{n} citing judgment{"s" if n != 1 else ""}</div>' if n is not None else ""
    icon = {"green": "✓", "red": "✕", "orange": "!", "gray": "?"}.get(colour, "?")
    return (f'<div class="ns-validity" style="background:{bg};color:{ink}"><div class="ns-validity-h">'
            f'<span class="ns-validity-i" style="background:{bar}">{icon}</span>{_html.escape(text)}</div>'
            f'<div class="ns-validity-m">{_html.escape(meaning)}</div>{meter}{cites}</div>')


def warning_strip(v):
    o = core.overruling(v)
    label = (v or {}).get("label")
    if label not in ("overruled", "doubted"):
        return
    by = ""
    if o:
        from ui.graph import node_info, short_title
        t, d = node_info(o["citing_id"])
        by = f" by <b>{_html.escape(short_title(t, 60))}</b>{f' ({str(d)[:4]})' if d else ''}"
    if label == "overruled":
        st.markdown(f'<div class="ns-warn ns-warn-red">⚠ <b>NOT GOOD LAW.</b> Overruled{by}. Do not rely on this '
                    f'judgment without reading the later one.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ns-warn ns-warn-amber">⚠ <b>Doubted.</b> A later judgment{by} disagreed with it, '
                    f'but its bench was smaller or unknown, so we could not confirm an overruling.</div>',
                    unsafe_allow_html=True)


def show_summary(s):
    if not s or s.get("status") != "ok":
        st.caption(f"Summary: {(s or {}).get('status', 'pending')}. {(s or {}).get('note', '')}")
        return
    rows = "".join(f'<div class="ns-field"><div class="ns-field-k">{name}</div><div class="ns-field-v">'
                   f'{_html.escape(core.tidy(s.get(f) or "")) or "<i>(nothing in the source for this field)</i>"}</div></div>'
                   for f, name in core.FIELD_NAMES.items())
    st.markdown(f'<div class="ns-fields">{rows}</div>', unsafe_allow_html=True)
    meta = f"generator: {Path(str(s.get('generator'))).name} · roles: {s.get('roles_source')} · NLI: {s.get('nli')}"
    if s.get("precomputed"):
        meta += f" · precomputed for the demo ({s.get('seconds') or '?'} s to generate)"
    st.caption(meta)
    if s.get("rejected"):
        with st.expander(f"🛡 {len(s['rejected'])} generated sentence(s) removed by the NLI faithfulness filter"):
            st.caption("Legal-LED wrote these, but the judgment's own sentences do not entail them, so they are not shown "
                       "in the summary above.")
            for r in s.get("rejected_detail") or [{"field": "", "sentence": x} for x in s["rejected"]]:
                p = f" · P(entail) {r['p_entail']:.2f}" if "p_entail" in r else ""
                st.markdown(f'<div class="ns-rejected"><span>{_html.escape(str(r.get("field", "")).replace("_", " "))}{p}'
                            f'</span>{_html.escape(r["sentence"])}</div>', unsafe_allow_html=True)


def summary_tab(r, query):
    key = abs_key(query, r["case_id"])
    if key not in st.session_state:              # precomputed Legal-LED summary: show it straight away
        hit = core.cached_abstractive(query, r["case_id"], demo_cache())
        if hit:
            st.session_state[key] = hit
    if key in st.session_state:
        st.markdown("**Abstractive summary** (Legal-LED, every sentence checked against the judgment)")
        show_summary(st.session_state[key])
        with st.expander("Extractive summary (role sentences copied from the judgment)"):
            show_summary(r.get("summary"))
    else:
        show_summary(r.get("summary"))
    if key not in st.session_state:
        ready = core.cached_abstractive(query, r["case_id"], demo_cache()) is not None
        label = "✨ Abstractive summary (Legal-LED + NLI filter)" + (" · instant" if ready else "")
        if st.button(label, key=f"btn::{r['case_id']}",
                     help="Precomputed for this query" if ready else "Runs Legal-LED on this judgment; minutes on a laptop"):
            with st.spinner("Generating with Legal-LED and checking each sentence with NLI ..."):
                st.session_state[key] = abstractive(query, r["case_id"])
            st.rerun()


def evidence_tab(v):
    if v and v.get("votes"):
        total = sum(v["votes"].values()) or 1
        bars = "".join(f'<div class="ns-vote"><span>{_html.escape(k)}</span><div class="ns-meter"><div style="width:'
                       f'{w / total * 100:.0f}%;background:{VALIDITY_STYLE[core.BADGES.get(k, ("", "gray"))[1]][2]}">'
                       f'</div></div><b>{w:.2f}</b></div>' for k, w in sorted(v["votes"].items(), key=lambda x: -x[1]))
        st.markdown(f'<div class="ns-votes"><div class="ns-field-k">Weighted votes of citing judgments</div>{bars}</div>',
                    unsafe_allow_html=True)
    rows = core.evidence_rows(v)
    if not rows:
        st.caption("No citing sentences in the citation graph.")
    from ui.graph import node_info
    for e in rows:
        src = f" · {e['source']}" if e.get("source") else ""
        title, date = node_info(e.get("citing_id") or "")
        when = f" ({str(date)[:4]})" if date else ""
        st.markdown(f"- **{e.get('treatment')}** in *{core.display_title(title)}*{when}{src} · weight "
                    f"{e.get('weight') or 0:.2f}  \n  {core.tidy(e['sent_md'])}")


def graph_tab(r):
    from ui.graph import graph_svg
    try:
        nb = neighbourhood(r["case_id"], r.get("validity"))
    except Exception as e:                       # noqa: BLE001  edges file missing (API mode on another machine)
        st.caption(f"Citation graph unavailable here: {e}")
        return
    st.html(graph_svg(r["case_id"], r.get("title") or r["case_id"], nb, (r.get("validity") or {}).get("label")))
    st.caption("Hover a case to see the sentence that cites it. Showing the strongest links: up to 14 citing, 10 cited.")
    rows = ([{"direction": "cited by", "case": n["title"], "year": n["year"], "treatment / verdict": n["kind"]}
             for n in nb["citing"]] +
            [{"direction": "cites", "case": n["title"], "year": n["year"], "treatment / verdict": n["kind"].replace("_", " ")}
             for n in nb["cited"]])
    if rows:
        with st.expander("Connected cases as a table"):
            st.dataframe(rows, hide_index=True, width="stretch")


def trace_tab(r, out):
    steps = core.ranking_trace(r, out.get("alpha"), out.get("components") or {})
    if not steps:
        st.caption("No ranking trace for this backend.")
        return
    boxes = '<div class="ns-arrow">→</div>'.join(
        f'<div class="ns-step"><div class="ns-step-k">{_html.escape(s["stage"])}</div><div class="ns-step-v">'
        f'{_html.escape(s["value"])}</div><div class="ns-step-n">{_html.escape(s["note"])}</div></div>' for s in steps)
    st.markdown(f'<div class="ns-trace">{boxes}</div>', unsafe_allow_html=True)
    for key, label in (("matched_facts", "Closest facts sentence"), ("matched_reasoning", "Closest reasoning sentence")):
        m = r.get(key)
        if m:
            st.markdown(f"**{label}** ({m.get('role')}): {m.get('text')}")


def judgment_tab(r, query):
    sents, src = judgment_roles(r["case_id"])
    if not sents:
        st.caption("Full text not available for this judgment.")
        return
    present = [x for x in ROLE_COLOURS if any(role == x for _, role in sents)]
    chips = "".join(f'<span class="ns-chip" style="--c:{ROLE_COLOURS[x]}">{x}</span>' for x in present)
    st.markdown(f'<div class="ns-chips"><span class="ns-chips-k">Role colours</span>{chips}'
                f'<span class="ns-chip ns-chip-hit">★ matches your query</span></div>', unsafe_allow_html=True)
    focus = st.pills("Focus on (the full text dims the other roles)", present, selection_mode="multi", default=present,
                     key=f"roles::{r['case_id']}")
    hits = set(core.query_matches(sents[:400], query))
    if hits:
        with st.expander(f"★ {len(hits)} passages closest to your query", expanded=True):
            for i in sorted(hits):
                t, role = sents[i]
                st.markdown(f'<div class="ns-hit" style="--c:{ROLE_COLOURS.get(role, "#9ca3af")}"><span>{role or "?"}</span>'
                            f'{_html.escape(t)}</div>', unsafe_allow_html=True)
    if st.toggle("Show the full role-labelled judgment", key=f"jt::{r['case_id']}"):
        st.caption(f"roles: {src}")
        parts = []
        for i, (t, role) in enumerate(sents[:400]):
            cls = "ns-s" + (" ns-s-hit" if i in hits else "") + ("" if role in (focus or present) else " ns-s-dim")
            parts.append(f'<span class="{cls}" style="--c:{ROLE_COLOURS.get(role, "#9ca3af")}" title="{role}">'
                         f'{_html.escape(t)}</span>')
        st.markdown(f'<div class="ns-judgment">{" ".join(parts)}</div>', unsafe_allow_html=True)


def show_card(i, r, query, out):
    v = r.get("validity")
    with st.container(border=True, key=f"card-{i}"):
        warning_strip(v)
        head, right = st.columns([5, 2])
        head.markdown(f"#### {i}. {core.display_title(r.get('title')) or r['case_id']}")
        bits = [r.get("court"), r.get("date"), f"bench of {r['bench_size']}" if r.get("bench_size") else None,
                r["case_id"], f"score {r['score']}" if r.get("score") is not None else None]
        head.caption(" · ".join(str(b) for b in bits if b))
        right.markdown(validity_block(v), unsafe_allow_html=True)
        s = st.session_state.get(abs_key(query, r["case_id"])) or core.cached_abstractive(query, r["case_id"], demo_cache())
        held = core.snippet(s) or core.snippet(r.get("summary"))
        if held:
            head.markdown(f'<div class="ns-held"><span>Ratio</span>{_html.escape(held)}</div>', unsafe_allow_html=True)
        open_ = i == 1 or (v or {}).get("label") in ("overruled", "doubted")
        if st.toggle("Details: summary · validity evidence · citation graph · ranking · judgment", value=open_,
                     key=f"open::{query}::{r['case_id']}"):
            card_tabs(r, query, out)


def card_tabs(r, query, out):
    v = r.get("validity")
    tabs = st.tabs(["📄 Summary", "⚖️ Validity evidence", "🕸 Citation graph", "🔎 How it was ranked", "📜 Judgment"])
    with tabs[0]:
        summary_tab(r, query)
    with tabs[1]:
        evidence_tab(v)
    with tabs[2]:
        graph_tab(r)
    with tabs[3]:
        trace_tab(r, out)
    with tabs[4]:
        judgment_tab(r, query)


def llm_panel(query, results):
    from ui import llm_compare
    key = f"llm::{query}"
    cmp = st.session_state.get(key) or llm_compare.saved(query, demo_cache())
    if cmp is None:
        st.caption("The comparison was saved for the example queries. For this query, ask the LLM now "
                   "(same prompt as the evaluation baseline; its cases are resolved against our citation graph).")
        if st.button("Ask the plain LLM now (about 15 s)", key="llm_live"):
            try:
                with st.spinner("Asking the LLM and checking every case it names ..."):
                    st.session_state[key] = llm_compare.live(query, get_resolver())
                st.rerun()
            except SystemExit as e:              # no key in .env
                st.warning(str(e))
            except Exception as e:               # noqa: BLE001  network, provider error
                st.warning(f"LLM call failed: {e}")
        return
    h = llm_compare.headline(cmp)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="ns-col-h ns-col-ours">NyayaSethu: grounded retrieval</div>', unsafe_allow_html=True)
        for j, r in enumerate(results[:10], 1):
            text, colour, _ = core.badge(r.get("validity"))
            st.markdown(f'<div class="ns-row"><b>{j}.</b> {_html.escape(core.display_title(r.get("title")) or r["case_id"])}'
                        f'<span class="ns-pill ns-pill-{colour}">{_html.escape(text)}</span></div>', unsafe_allow_html=True)
        st.caption("Every hit is a judgment we hold in full text; validity comes from the judgments that cite it.")
    with right:
        st.markdown(f'<div class="ns-col-h ns-col-llm">Plain LLM, no retrieval ({_html.escape(str(cmp.get("model")))})</div>',
                    unsafe_allow_html=True)
        colours = {"green": "green", "blue": "gray", "orange": "orange", "red": "red"}
        for r in cmp["rows"]:
            vt, vc, vm = llm_compare.VERDICTS[r["verdict"]]
            flags = []
            if r["repeated"]:
                flags.append("repeated")
            if r["page_one"]:
                flags.append("cites page 1")
            if r["our_label"] in ("overruled", "doubted") and r["llm_status"] == "good_law":
                flags.append(f"LLM says good law; ours: {r['our_label']}")
            match = (f'<div class="ns-row-sub">matched: {_html.escape(r["matched_title"])}</div>'
                     if r.get("matched_title") and r["verdict"] in ("mismatch", "uncertain", "resolved") else "")
            st.markdown(f'<div class="ns-row"><b>{r["rank"]}.</b> {_html.escape(str(r["title"]))} '
                        f'<span class="ns-cite">{_html.escape(str(r.get("citation") or ""))}</span>'
                        f'<span class="ns-pill ns-pill-{colours[vc]}" title="{_html.escape(vm)}">{_html.escape(vt)}</span>'
                        f'{"".join(f"<span class=ns-flag>{_html.escape(f)}</span>" for f in flags)}{match}</div>',
                        unsafe_allow_html=True)
        st.caption(f"{h['exist']} of {h['named']} named cases verified in our graph · {h['good_law']} labelled good law "
                   f"by the LLM · {h['page_one']} citations point to page 1"
                   + ("" if cmp.get("hand_checked") else " · automatic matching, not hand-checked")
                   + ". Not found is not proof of invention: the resolver cannot match every citation.")


def results_view(out, query, k, filters):
    hits = core.apply_filters(out["results"], **filters)[:k]
    if out.get("fallback_error"):
        c = out.get("cached") or {}
        st.warning(f"The live backend is unavailable ({out['fallback_error']}). Showing results saved for "
                   + ("this query." if c.get("exact") else f"the closest example query: “{c.get('query')}”."))
    labels = [(r.get("validity") or {}).get("label") for r in hits]
    t = (out.get("timings") or {}).get("total")
    summary = [f"<b>{len(hits)}</b> precedents"] + [
        f'<span class="ns-pill ns-pill-{core.BADGES[lab][1]}">{labels.count(lab)} {core.BADGES[lab][0].lower()}</span>'
        for lab in ("overruled", "doubted", "distinguished", "followed") if labels.count(lab)]
    if t:
        summary.append(f"{t}s")
    comp = out.get("components") or {}
    bar, b1, b2 = st.columns([6, 1.3, 1.3])
    bar.markdown(f'<div class="ns-summary">{" · ".join(summary)}</div>', unsafe_allow_html=True)
    bar.caption(" · ".join(f"{a}: {b}" for a, b in comp.items()))
    abstr = {r["case_id"]: st.session_state[abs_key(query, r["case_id"])] for r in hits
             if abs_key(query, r["case_id"]) in st.session_state}
    today = _dt.date.today().strftime("%d %B %Y")
    sub = {**out, "results": hits}
    b1.download_button("⬇ Markdown", core.brief_markdown(sub, query, abstr, today), "nyayasethu_brief.md",
                       "text/markdown", width="stretch")
    b2.download_button("⬇ PDF brief", core.brief_html(sub, query, abstr, today), "nyayasethu_brief.html",
                       "text/html", width="stretch", help="Open it, then Print → Save as PDF")
    if st.toggle("🆚 Compare with a plain LLM (no retrieval)", key="cmp_toggle"):
        with st.container(border=True, key="llm-panel"):
            llm_panel(query, hits)
    if not hits:
        st.warning("No results" + (" after the filters in the sidebar." if out["results"] else "."))
    for i, r in enumerate(hits, 1):
        show_card(i, r, query, out)


# ---------------------------------------------------------------- pages
def pick_example(q):
    st.session_state["q"] = q
    st.session_state["go"] = True


def search_page():
    shared = st.query_params.get("q")                # ?q=... runs a query on load (bookmarkable demo links)
    if shared and st.session_state.get("shared") != shared:
        st.session_state["shared"] = shared
        pick_example(shared)
    brand()
    hero()
    st.markdown('<div class="ns-label">Try an example</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for j, d in enumerate(core.DEMO_QUERIES):
        cols[j % 4].button(("🔴 " if d.get("overruled") else "") + d["label"], key=f"ex::{j}", on_click=pick_example,
                           args=(d["query"],), width="stretch",
                           help="Results include an overruled precedent" if d.get("overruled") else d["query"])
    query = st.text_area("Describe your legal problem in plain language", height=100, key="q",
                         placeholder="My landlord kept my entire security deposit without showing any damage ...")
    go = st.button("Search precedents", type="primary") or st.session_state.pop("go", False)
    if go and query.strip():
        with st.spinner("Searching, checking validity and summarising ..."):
            try:
                st.session_state["result"] = search(query.strip(), FETCH)
                st.session_state["query"] = query.strip()
            except FileNotFoundError as e:
                st.error(f"{e}. Or run the UI with `-- --mock`.")
            except Exception as e:            # API down and no saved results
                st.error(f"Search failed: {e}")
    sidebar()                                        # after the search, so the filters fit the new results
    out = st.session_state.get("result")
    if out:
        results_view(out, st.session_state["query"], st.session_state.get("k", 10), current_filters(out))


def current_filters(out):
    q = st.session_state.get("query", "")
    return {"courts": st.session_state.get(f"f_courts::{q}") or None, "years": st.session_state.get(f"f_years::{q}"),
            "hide_overruled": st.session_state.get("f_hide", False), "min_bench": st.session_state.get("f_bench", 1)}


def brand():
    with st.sidebar:
        st.markdown('<div class="ns-side-title">⚖️ NyayaSethu</div>', unsafe_allow_html=True)
        st.caption("Validity-aware precedent retrieval for Indian case law")
        st.markdown(f"**Mode:** {'mock data' if MOCK else 'real data'} · **Retrieval:** {BACKEND}")


def sidebar():
    with st.sidebar:
        out = st.session_state.get("result")
        st.markdown("##### Filters")
        st.slider("Results", 3, 20, 10, key="k")
        courts = sorted({r["court"] for r in (out or {}).get("results", []) if r.get("court")})
        if len(courts) > 1:
            st.multiselect("Court", courts, key=f"f_courts::{st.session_state.get('query', '')}")
        years = [y for y in (core.year_of(r) for r in (out or {}).get("results", [])) if y]
        if years and min(years) < max(years):
            st.slider("Year of decision", min(years), max(years), (min(years), max(years)),   # ranges depend on the
                      key=f"f_years::{st.session_state.get('query', '')}")                    # results: one per query
        st.toggle("Hide overruled judgments", key="f_hide")
        st.slider("Minimum bench size", 1, 7, 1, key="f_bench", help="Judgments with an unknown bench size are kept")
        st.markdown("##### Validity badges")
        for lab, (t, c, m) in core.BADGES.items():
            st.badge(t, color=c)
            st.caption(m)
        if MOCK:
            st.info("Mock mode: retrieval returns the same fixed list for any query.")


def eval_page():
    brand()
    from ui import evaluation
    evaluation.render(st)


st.navigation([st.Page(search_page, title="Search", icon="🔎", default=True),
               st.Page(eval_page, title="Evaluation", icon="📊", url_path="evaluation")]).run()
