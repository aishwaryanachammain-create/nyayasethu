"""Member B, Phase 4: LLM-only baseline harness (parsing, resolution, scoring). Offline, no API calls."""
from baseline_llm.run import parse_cases, resolve_case, score
from stage0_graph.nodes import Node
from stage0_graph.resolve import Resolver

NODES = {"sc:A": Node("sc:A", title="Fateh Chand v. Balkishan Das", petitioner="Fateh Chand",
                      respondent="Balkishan Das", date="1963-03-11", keys=["SCR|1964|1|515"]),
         "sc:B": Node("sc:B", title="Surya Dev Rai v. Ram Chander Rai", petitioner="Surya Dev Rai",
                      respondent="Ram Chander Rai", date="2003-08-07")}


def test_parse_cases():
    raw = 'Sure:\n[{"title": "A v. B", "status": "good_law"}, {"no_title": 1}, "x"]\nHope this helps.'
    assert parse_cases(raw) == [{"title": "A v. B", "status": "good_law"}]
    assert parse_cases("no json here") == [] and parse_cases("[broken") == [] and parse_cases(None) == []


def test_resolve_case():
    res = Resolver(NODES)
    assert resolve_case({"title": "Fateh Chand v. Balkishan Das", "citation": "(1964) 1 SCR 515"}, res) == ("sc:A", "key")
    cid, method = resolve_case({"title": "Surya Dev Rai v. Ram Chander Rai", "citation": "(2003) 6 SCC 675",
                                "year": 2003}, res)
    assert cid == "sc:B" and method == "name_year"                # SCC is not in metadata: names near the year
    assert resolve_case({"title": "Imaginary Traders v. Nonexistent Builders", "year": 1999}, res)[0] is None


def test_score():
    rows = [{"query_id": "q1", "case_id": "sc:A", "method": "key", "full_text": True, "llm_status": "good_law",
             "our_label": "followed", "in_our_topk": True, "nli_sentences": 4, "nli_kept": 1},
            {"query_id": "q1", "case_id": "sc:B", "method": "name_year", "full_text": False, "llm_status": "good_law",
             "our_label": "overruled", "in_our_topk": False, "nli_sentences": 0, "nli_kept": 0},
            {"query_id": "q1", "case_id": "", "method": "unresolved", "full_text": False, "llm_status": "",
             "our_label": "", "in_our_topk": False, "nli_sentences": 0, "nli_kept": 0}]
    s = score(rows)
    assert s["exists_in_graph"] == 2 and s["exists_rate"] == round(2 / 3, 4)
    assert s["validity_compared"] == 2 and s["validity_agreement"] == 0.5
    assert s["in_our_topk"] == 1 and s["nli_entailed_rate"] == 0.25
