"""Member B, Phase 5: IN-Ext role coverage and the human-eval sheet writer. Offline (tiny fixtures)."""
import csv
import json

from stage5_summary import eval_inext, eval_pairs

DOC = [("The tenant paid an advance of Rs 50,000 to the landlord.", "Facts"),
       ("The landlord kept the advance after the tenant left.", "Facts"),
       ("Forfeiture without proof of loss is a penalty under section 74.", "Ratio of the decision"),
       ("Counsel argued that the deposit was earnest money.", "Argument"),
       ("The appeal is allowed with costs.", "Ruling by Present Court")]


def test_inext_coverage(tmp_path):
    for ann in ("A1", "A2"):
        for seg, text in {"facts": "the tenant paid an advance of rs 50000 to the landlord.",
                          "judgement": "the appeal is allowed with costs.",
                          "argument": "counsel argued that the deposit was earnest money."}.items():
            (tmp_path / ann / seg).mkdir(parents=True, exist_ok=True)
            (tmp_path / ann / seg / "X.txt").write_text(text)
    res = eval_inext.evaluate({"gold": {"bhatt:X": DOC}}, root=tmp_path)["gold"]
    assert res["facts"]["coverage"] == 1.0 and res["facts"]["pool_share"] == 0.4
    assert res["judgement"]["coverage"] == 1.0 and res["judgement"]["rouge1_f1"] == 1.0
    assert res["argument"]["coverage"] is None and res["analysis"]["n_summaries"] == 0
    assert eval_inext.match(["something entirely different here"], [set(eval_inext.toks(t)) for t, _ in DOC]) == [None]


def test_sheet_is_blinded_and_rates_counted(tmp_path, monkeypatch):
    for k, name in (("PAIRS", "pairs.jsonl"), ("SHEET", "sheet.csv"), ("KEY", "key.csv"), ("NLI_OUT", "nli.json")):
        monkeypatch.setattr(eval_pairs, k, tmp_path / name)
    recs = []
    for i in range(6):
        qf = {f: (f"{f} text." if f != "decision" else "") for f in eval_pairs.FIELDS}
        recs.append({"query_id": f"q{i}", "query": "q", "rank": 1, "case_id": f"c{i}", "title": "T",
                     "generator": "led", "nli_model": "nli", "generic": "GENERIC SUMMARY.", "generic_sentences": 2,
                     "generic_rejected": ["x"], "query_focused": qf,
                     "qf_kept_sentences": {f: (1 if qf[f] else 0) for f in eval_pairs.FIELDS},
                     "qf_rejected": [{"field": "decision", "sentence": "y", "p_entail": 0.1}],
                     "provenance": {}, "roles_source": "stage1_labeller", "seconds_generic": 1, "seconds_query_focused": 1})
    (tmp_path / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    eval_pairs.write_outputs()
    sheet = list(csv.DictReader(open(tmp_path / "sheet.csv")))
    key = {r["pair_id"]: r for r in csv.DictReader(open(tmp_path / "key.csv"))}
    assert len(sheet) == 6 and "query_focused" not in open(tmp_path / "sheet.csv").read()
    for r in sheet:
        qf_col = "summary_A" if key[r["pair_id"]]["A"] == "query_focused" else "summary_B"
        assert r[qf_col].startswith("Issue: ") and r["preferred_A_B_tie"] == ""
    assert {k["A"] for k in key.values()} == {"query_focused", "generic"}
    nli = json.loads((tmp_path / "nli.json").read_text())
    assert nli["query_focused"]["rejection_rate"] == round(6 / 30, 4)
    assert nli["query_focused"]["per_field"]["decision"]["rejection_rate"] == 1.0
    assert nli["generic_whole_judgment"]["rejection_rate"] == 0.5


def test_score_sheets(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_pairs, "KEY", tmp_path / "key.csv")
    monkeypatch.setattr(eval_pairs, "ROOT", tmp_path)
    (tmp_path / "eval").mkdir()
    (tmp_path / "key.csv").write_text("pair_id,query_id,case_id,A,B\np01,q1,c1,query_focused,generic\n"
                                      "p02,q1,c2,generic,query_focused\n")
    cols = "pair_id,A_faithful_1to5,A_relevant_1to5,A_useful_1to5,B_faithful_1to5,B_relevant_1to5,B_useful_1to5,preferred_A_B_tie\n"
    blank = tmp_path / "blank.csv"
    blank.write_text(cols + "p01,,,,,,,\n")
    assert eval_pairs.score_sheets([blank])["status"].startswith("PENDING")
    filled = tmp_path / "r1.csv"
    filled.write_text(cols + "p01,5,4,4,3,2,2,A\np02,3,,2,4,5,5,B\n")
    r = eval_pairs.score_sheets([filled])
    assert r["mean"]["query_focused/relevant"] == 4.5 and r["mean"]["generic/relevant"] == 2.0
    assert r["n"]["generic/relevant"] == 1 and r["preferred"] == {"query_focused": 2}
