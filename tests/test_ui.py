"""Member B, Phase 4: re-exports for A's API, UI logic, and the Streamlit app in mock mode. Offline."""
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def mock_mode(monkeypatch):
    from common import config
    from stage4_validity import _table
    from stage5_summary import docs
    monkeypatch.delenv("NYAYASETHU_CONFIG", raising=False)
    caches = (config.cfg, _table, docs._segments, docs.offsets)
    for c in caches:
        c.cache_clear()
    from ui import core
    core.use_mock_config()
    yield core
    os.environ.pop("NYAYASETHU_CONFIG", None)
    for c in caches:
        c.cache_clear()
    for p in (REPO / "data" / "mock").glob("*.offsets.json"):
        p.unlink()


def test_reexports_reach_a_interface(mock_mode):
    from app import b_interface
    assert b_interface.available() == {"get_validity": True, "summarize": True}
    assert b_interface.get_validity("sc:MOCK1971002")["label"] == "overruled"
    s = b_interface.summarize("deposit forfeited", "sc:MOCK1963001")
    assert s["status"] == "ok" and s["generator"] == "extractive"     # stage5.api_mode: fast for every hit


def test_mock_search_and_badges(mock_mode):
    core = mock_mode
    out = core.add_member_b(core.mock_search("anything", k=3), "anything")
    assert [r["case_id"] for r in out["results"]] == ["sc:MOCK1963001", "sc:MOCK1971002", "sc:MOCK2003014"]
    labels = [core.badge(r["validity"])[:2] for r in out["results"]]
    assert labels[0] == ("Followed", "green") and labels[1] == ("Overruled", "red")
    assert core.badge({"status": "pending", "label": None, "note": "x"}) == ("Validity pending", "gray", "x")
    ev = core.evidence_rows(out["results"][0]["validity"], limit=2)
    assert len(ev) == 2 and ev[0]["weight"] >= ev[1]["weight"]
    assert core.evidence_rows({"evidence": [{"sent": "see [[X v. Y]]", "weight": 1}]})[0]["sent_md"] == "see **X v. Y**"


def test_streamlit_app_mock(mock_mode):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO / "ui" / "app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    at.text_area[0].input("landlord kept my deposit")
    at.button[0].click().run()
    assert not at.exception and not at.error
    heads = [m.value for m in at.markdown if m.value.startswith("#### ")]
    assert len(heads) == 7 and heads[0].startswith("#### 1. Ramesh Traders v. Kavita Builders")
    for t in at.toggle:                               # open every card and the LLM panel: all tabs must render
        if t.key and (t.key.startswith("open::") or t.key == "cmp_toggle"):
            t.set_value(True)
    at.run()
    assert not at.exception and not at.error
    at.toggle(key="f_hide").set_value(True).run()     # sidebar filter: the overruled mock case disappears
    heads = [m.value for m in at.markdown if m.value.startswith("#### ")]
    assert len(heads) == 6 and not any("Sunder Lal" in h for h in heads)


def test_example_button_and_eval_page(mock_mode):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO / "ui" / "app.py"), default_timeout=60)
    at.run()
    at.button(key="ex::0").click().run()              # an example query fills the box and searches
    assert not at.exception
    assert at.text_area(key="q").value == mock_mode.DEMO_QUERIES[0]["query"]
    assert [m.value for m in at.markdown if m.value.startswith("#### 1.")]


def _eval_page():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path.cwd()))
    import streamlit as st
    from ui import evaluation
    evaluation.render(st)


def test_evaluation_page():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(_eval_page, default_timeout=60)
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Macro-F1"] == "0.530" and metrics["Confirmed to exist"] == "8 (24%)"


def test_filters_matches_trace():
    from ui import core
    rs = [{"case_id": "a", "court": "SC", "date": "2019-01-01", "bench_size": 2, "validity": {"label": "followed"}},
          {"case_id": "b", "court": "HC", "date": "1995-05-05", "bench_size": 5, "validity": {"label": "overruled"}},
          {"case_id": "c", "court": None, "date": None, "bench_size": None, "validity": None}]
    ids = lambda xs: [r["case_id"] for r in xs]
    assert ids(core.apply_filters(rs)) == ["a", "b", "c"]
    assert ids(core.apply_filters(rs, hide_overruled=True)) == ["a", "c"]
    assert ids(core.apply_filters(rs, courts={"SC"})) == ["a", "c"]          # unknown court is kept
    assert ids(core.apply_filters(rs, years=(2000, 2023))) == ["a", "c"]
    assert ids(core.apply_filters(rs, min_bench=3)) == ["b", "c"]
    sents = [("The tenant paid rent.", "Facts"), ("Security deposit forfeited without proof of loss.", "Issue"),
             ("Appeal allowed.", "Ruling by Present Court")]
    assert core.query_matches(sents, "landlord kept my security deposit", n=1) == [1]
    steps = core.ranking_trace({"surfaced_by": {"bm25:facts": 3, "bm25:reasoning": 1}, "ce": 0.0, "prior": 0.3,
                                "score": 0.06}, 0.8, {"cross-encoder": "missing (prior-only rerank)"})
    assert [s["stage"] for s in steps] == ["BM25 · reasoning", "BM25 · facts", "RRF fusion", "Cross-encoder",
                                           "Authority prior", "Final score"]
    assert steps[3]["value"] == "off" and "0.8·CE" in steps[-1]["note"]


def test_display_helpers():
    from ui import core
    assert core.display_title("DANAMMA @ SUMAN SURPUR & ANR. versus AMAR AND ORS.") == "Danamma @ Suman Surpur & Anr. v. Amar and Ors."
    assert core.display_title("Ramesh Traders v. Kavita") == "Ramesh Traders v. Kavita"      # mixed case untouched
    assert core.tidy("A B C D E F G H SUPREME COURT REPORTS [2019] 7 S.C.R. The appeal [Para 23][46-G-H; 47-A] fails.") == "The appeal fails."
    s = {"status": "ok", "ratio": "word " * 100}
    assert core.snippet(s, 40).endswith("…") and len(core.snippet(s, 40)) <= 42


def test_brief_and_overruling(mock_mode):
    core = mock_mode
    out = core.add_member_b(core.mock_search("deposit", k=3), "deposit")
    md = core.brief_markdown(out, "deposit", today="1 Jan 2026")
    assert md.startswith("# NyayaSethu research brief") and "## 2. Sunder Lal" in md and "Not good law" in md
    html = core.brief_html(out, "deposit <b>", today="1 Jan 2026")
    assert "deposit &lt;b&gt;" in html and html.count("<section>") == 3
    o = core.overruling(out["results"][1]["validity"])
    assert o["citing_id"] == "sc:MOCK2022400"


def test_citation_graph_mock(mock_mode):
    from app import b_interface
    from ui import graph
    graph._edges.cache_clear(); graph._nodes.cache_clear(); graph.node_info.cache_clear()
    v = b_interface.get_validity("sc:MOCK1963001")
    nb = graph.neighbourhood("sc:MOCK1963001", v)
    assert nb["n_citing"] == 4 and {n["kind"] for n in nb["citing"]} == {"followed", "distinguished"}
    svg = graph.graph_svg("sc:MOCK1963001", "Ramesh Traders", nb, v["label"])
    assert svg.startswith("<svg") and svg.count('class="nd"') == len(nb["citing"]) + len(nb["cited"])
    graph._edges.cache_clear(); graph._nodes.cache_clear(); graph.node_info.cache_clear()


def test_demo_caches_and_llm_comparison():
    from ui import core, llm_compare
    q1 = core.DEMO_QUERIES[0]["query"]
    s = core.cached_abstractive(q1, "sc:ESCR010005702019", cache={})               # from eval/stage5_pairs.jsonl
    assert s and s["status"] == "ok" and s["precomputed"] and s["rejected_detail"]
    assert core.cached_abstractive("an unseen query", "sc:ESCR010005702019", cache={}) is None
    cache = {"search": {core._norm(d["query"]): {"query": d["query"], "results": [{"case_id": str(i)} for i in range(5)]}
                        for d in core.DEMO_QUERIES[:2]}}
    exact = core.cached_search(q1.upper(), 3, cache)
    assert exact["cached"]["exact"] and len(exact["results"]) == 3
    near = core.cached_search("police recorded the confession of the accused", 10, cache)
    assert not near["cached"]["exact"] and near["query"] == core.DEMO_QUERIES[1]["query"]
    assert core.cached_search("x", 3, {}) is None
    assert llm_compare.totals() == {"named": 33, "confirmed": 8}                   # eval/RESULTS_B.md 4.1
    cmp = llm_compare.saved(core.DEMO_QUERIES[2]["query"])                          # q3: the same case five times
    h = llm_compare.headline(cmp)
    assert h["named"] == 5 and h["repeated"] == 4 and h["exist"] == 0
    rows = llm_compare.saved(core.DEMO_QUERIES[1]["query"])["rows"]
    assert rows[0]["verdict"] == "mismatch" and rows[4]["verdict"] == "confirmed"
