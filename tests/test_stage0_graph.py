"""Member B, Stage 0 graph: extraction, resolution, build. Offline, on data/mock (fictional)."""
from pathlib import Path

from stage0_graph.build import body_start, build, clean, graph_stats
from stage0_graph.extract import alias_keys, mentions
from stage0_graph.nodes import Node, scr_range
from stage0_graph.resolve import Resolver

REPO = Path(__file__).resolve().parents[1]
MOCK = REPO / "data" / "mock" / "judgments.jsonl"


def test_reporter_forms_and_names():
    t = ("In Ramesh Traders v. Kavita Builders, (1963) 3 SCR 515 : AIR 1963 SC 1405, and see [1959] SUPP. 1 S.C.R. 28, "
         "1964 (1) SCR 515, 2019 SCC OnLine SC 12, 2023 INSC 456 and (2009) 6 SCC 121.")
    ms = mentions(t)
    kinds = [m.kind for m in ms]
    assert {"SCR", "AIR", "SCRSUPP", "SCCOL", "INSC", "SCC"} <= set(kinds)
    assert ms[0].name == "Ramesh Traders v Kavita Builders" and ms[1].name == ms[0].name   # parallel citation
    assert alias_keys(["[1959] SUPP. 1 S.C.R. 28", "2018INSC696"]) == ["INSC|2018||696", "SCRSUPP|1959|1|28"]


def test_case_name_strips_lead_words():
    ms = mentions("This Court in Sunder Lal v. State of Mysore held so. Whether Anil Kumar v. Northern Mills applies.")
    assert [m.name for m in ms] == ["Sunder Lal v State of Mysore", "Anil Kumar v Northern Mills"]


def test_scr_page_furniture_removed():
    s = ("the Bench. M/S. GOEL GANGA DEVELOPERS v. U.O.I.THR. SECY. MINISTRY OF ENVIRONMENT [DEEPAK GUPTA, J.] "
         "A B C D E F G H SUPREME COURT REPORTS [2018] 10 S.C.R. impractical")
    assert clean(s) == "the Bench. impractical"
    assert mentions(clean(s)) == []


def test_resolution_methods_and_time_order():
    nodes = {
        "sc:A": Node("sc:A", "FATEH LAL versus BALKISHAN DASS", "FATEH LAL", "BALKISHAN DASS", "1963-05-01", 5,
                     keys=["SCR|1964|1|515"], scr_range=("SCR", "1964", "1", 515, 530)),
        "sc:B": Node("sc:B", "STATE OF UTTAR PRADESH versus RAJ NARAIN", "STATE OF UTTAR PRADESH", "RAJ NARAIN",
                     "1975-01-24", 4),
        "sc:C": Node("sc:C", "MUNNA versus STATE (N.C.T. OF DELHI)", "MUNNA", "STATE (N.C.T. OF DELHI)", "2003-08-27", 2),
    }
    r = Resolver(nodes)
    m = lambda t: mentions(t)[0]
    assert r.resolve(m("(1964) 1 SCR 515"), 2000)[:2] == ("sc:A", "key")
    assert r.resolve(m("(1964) 1 SCR 520"), 2000)[:2] == ("sc:A", "key_range")          # pinpoint page
    assert r.resolve(m("(1964) 1 SCR 515"), 1960)[0] is None                              # cited after citing
    assert r.resolve(m("State of U.P. v. Raj Narain, (1975) 4 SCC 428"), 1990)[:2] == ("sc:B", "name_year")
    assert r.resolve(m("State of U.P. v. Raj Narain, (1985) 4 SCC 428"), 1990)[0] is None  # wrong year
    assert r.resolve(m("as held in Raj Narain v. State of U.P. earlier"), 1990)[:2] == ("sc:B", "name_only")
    # subset names must not match ("munna" inside a longer name)
    assert r.resolve(m("Udaikumar Pandharinath Jadhav alias Munna v. State of Maharashtra"), 2010)[0] is None


def test_scr_range_from_metadata_path():
    assert scr_range("2019_10_133_142") == ("SCR", "2019", "10", 133, 142)
    assert scr_range("S_1959_1_28_38") == ("SCRSUPP", "1959", "1", 28, 38)
    assert scr_range("1999_compressed") is None


def test_build_on_mock(tmp_path):
    nodes, _, edges, stats, by_kind, citing, _, _ = build(MOCK, tmp_path / "no_metadata")
    pairs = {(e["citing_id"], e["cited_id"]) for e in edges}
    assert ("sc:MOCK2022400", "sc:MOCK1971002") in pairs and ("hc:MOCK2019088", "sc:MOCK1963001") in pairs
    assert all(e["citing_id"] != e["cited_id"] for e in edges)
    assert {"citing_id", "cited_id", "citing_sent_idx", "context"} <= set(edges[0])
    g, st = graph_stats(nodes, edges)
    assert st["edges_judgment_level"] == 5 and st["nodes_with_>=3_citing"] == 1


def test_body_start_skips_front_matter():
    from common.schema import Judgment, Sentence
    d = Judgment("sc:x", sentences=[Sentence(i, t) for i, t in enumerate(
        ["Headnote held: x.", "Case Law Reference [2011] 11 SCR 870 relied on.", "Advs. for the appearing parties.",
         "The Judgment of the Court was delivered by X, J.", "1. Body.", "2. More.", "3. End.", "4. End.", "5. End."])])
    assert body_start(d) == 3


def test_precision_scoring_ignores_assistant_labels():
    import pandas as pd
    from stage0_graph.precision import score
    df = pd.DataFrame({"check": ["extraction"] * 3 + ["resolution"] * 2, "kind": ["SCC", "NAME", "NAME", "", ""],
                       "extraction_correct": ["yes", "no", "", "", ""], "resolution_correct": ["", "", "", "", ""],
                       "assistant_correct": ["yes", "yes", "yes", "yes", "no"]})
    r = score(df)
    assert r["extraction"]["checked"] == 2 and r["extraction"]["precision"] == 0.5
    assert r["extraction"]["reporter_citations"]["precision"] == 1.0
    assert r["extraction"]["bare_case_names"]["precision"] == 0.0
    assert r["resolution"].startswith("PENDING")
