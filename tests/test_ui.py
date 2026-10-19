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
    heads = [m.value for m in at.markdown if m.value.startswith("####")]
    assert len(heads) == 7 and heads[0].startswith("#### 1. RAMESH TRADERS")
