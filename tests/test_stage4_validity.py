"""Member B, Stage 4 validity: weak labels, editorial parsing, aggregation rules, API. Offline, fictional data."""
import math

import pandas as pd

from stage4_validity.aggregate import aggregate, judgment_treatment
from stage4_validity.editorial import parse_block
from stage4_validity.evaluate import scores
from stage4_validity.labels import cue_label, editorial_label

S4 = {"min_citing": 3, "overrule_conf": 0.7, "recency_tau": 15, "bench_cap": 2.0, "default_bench": 2,
      "use_editorial": True}


def test_editorial_phrases():
    cases = {"relied on": "followed", "eferred to": "neutral", "held inapplicable": "distinguished",
             "not approved": "overruled", "held not good law": "overruled", "per incuriam": "overruled",
             "partly overruled": "overruled", "approved": "followed", "needs reconsideration": None, "ench": None}
    assert {k: editorial_label(k) for k in cases} == cases


def test_cue_rules():
    assert cue_label("We respectfully follow the view in [[X]].") == "followed"
    assert cue_label("[[X]] is clearly distinguishable on facts.") == "distinguished"
    assert cue_label("[[X]] stands overruled by the larger Bench.") == "overruled"
    assert cue_label("[[X]] was not followed.") is None                  # negated: dropped
    assert cue_label("[[X]] was referred to by counsel.") is None         # no cue
    assert cue_label("We follow [[X]], though [[Y]] is distinguishable.") is None   # two classes: ambiguous


def test_parse_editorial_block():
    blk = ("(2017) 9 SCC 499 referred to Para 20 [1997] 1 Suppl. SCR 87 relied on Para 19 "
           "2005 (3) Bom.C.R. 485 referred to Para 5 [1971] 2 SCR 40 overruled Para 7, 9")
    es = parse_block(blk)
    assert [(e.keys, e.label, e.paras) for e in es] == [
        (["SCC|2017|9|499"], "neutral", [20]), (["SCRSUPP|1997|1|87"], "followed", [19]),
        (["SCR|1971|2|40"], "overruled", [7, 9])]


def _nodes():
    return pd.DataFrame([
        {"case_id": "sc:OLD", "title": "OLD v STATE", "year": 1971, "bench_size": 2},
        {"case_id": "sc:A", "title": "A", "year": 2020, "bench_size": 2},
        {"case_id": "sc:B", "title": "B", "year": 2022, "bench_size": 2},
        {"case_id": "sc:C", "title": "C", "year": 2022, "bench_size": 3},
        {"case_id": "sc:BIG", "title": "BIG", "year": 2022, "bench_size": 5},
        {"case_id": "sc:ONE", "title": "ONE", "year": 2022, "bench_size": 1},
    ])


def _pred(citing, treatment, conf=0.9, editorial=None, cited="sc:OLD"):
    p = {l: 0.0 for l in ("followed", "distinguished", "overruled", "neutral")}
    p[treatment] = conf
    return {"mention_id": f"{citing}|1|{cited}", "citing_id": citing, "cited_id": cited, "citing_sent_idx": 1,
            "sentence": f"{citing} on [[OLD]]", "editorial": editorial, "cue": None, "split": "train",
            **{f"p_{k}": v for k, v in p.items()}, "pred": treatment, "conf": conf}


def _verdict(rows):
    return aggregate(pd.DataFrame(rows), _nodes(), S4)[0]


def test_overruled_needs_equal_or_larger_bench():
    assert _verdict([_pred("sc:BIG", "overruled", 0.95)])["label"] == "overruled"
    assert _verdict([_pred("sc:ONE", "overruled", 0.95)])["label"] == "doubted"      # smaller bench
    low = _verdict([_pred("sc:BIG", "overruled", 0.6)] + [_pred(c, "followed") for c in ("sc:A", "sc:B", "sc:C")])
    assert low["label"] == "followed"                                                  # conf below 0.7


def test_majority_and_insufficient():
    v = _verdict([_pred("sc:A", "followed"), _pred("sc:B", "distinguished"), _pred("sc:C", "distinguished")])
    assert v["label"] == "distinguished" and v["n_citing"] == 3
    # weights: bench ratio (3/2 for sc:C) and recency (ref year 2022)
    assert math.isclose(v["votes"]["distinguished"], 1.0 + 1.5, rel_tol=1e-3)
    assert math.isclose(v["votes"]["followed"], math.exp(-2 / 15), rel_tol=1e-3)
    assert _verdict([_pred("sc:A", "followed"), _pred("sc:B", "followed")])["label"] == "insufficient_evidence"
    only_refs = _verdict([_pred(c, "neutral") for c in ("sc:A", "sc:B", "sc:C")])
    assert only_refs["label"] == "insufficient_evidence"


def test_editorial_label_overrides_classifier():
    v = _verdict([_pred("sc:BIG", "followed", 0.9, editorial="overruled")])
    assert v["label"] == "overruled" and v["evidence"][0]["source"] == "editorial"


def test_judgment_merges_its_mentions():
    rows = [{"treatment": "neutral", "t_conf": 0.99, "sentence": "a", "source": "classifier"},
            {"treatment": "followed", "t_conf": 0.6, "sentence": "b", "source": "classifier"}]
    assert judgment_treatment(rows, 0.7)["treatment"] == "followed"


def test_scores_macro_f1():
    r = scores(["followed", "distinguished", "overruled", "neutral"], ["followed", "followed", "overruled", None])
    assert r["per_class_f1"]["distinguished"] == 0.0 and r["macro_f1_3class"] == round((2 / 3 + 0 + 1) / 3, 4)


def test_get_validity_on_mock(monkeypatch):
    import stage4_validity
    from common import config
    real = config.cfg()
    monkeypatch.setattr(config, "cfg", lambda path=None: {**real, "use_mock": True})
    stage4_validity._table.cache_clear()
    try:
        assert stage4_validity.get_validity("sc:MOCK1971002")["label"] == "overruled"
        assert stage4_validity.get_validity("sc:NOPE")["label"] == "insufficient_evidence"
    finally:
        stage4_validity._table.cache_clear()
