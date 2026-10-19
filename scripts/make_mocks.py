"""Write FICTIONAL stand-ins for every hand-off file into paths.mock (used when config `use_mock: true`).

Schemas follow Member A's real files (common/schema.py) and data/CONTRACT.md. Parties, citations and
holdings are invented; they are not law.

    python scripts/make_mocks.py
"""
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import ROOT, cfg

OUT = ROOT / cfg()["paths"]["mock"]
OUT.mkdir(parents=True, exist_ok=True)

F, I, P, R, D = "Facts", "Issue", "Precedent", "Ratio of the decision", "Ruling by Present Court"

# (case_id, court, date, bench, title, aliases, [(sentence, role)])
DOCS = [
    ("sc:MOCK1963001", "SC", "1963-01-10", 5, "RAMESH TRADERS versus KAVITA BUILDERS", ["[1963] 3 S.C.R. 515", "1963INSC901"], [
        ("The appellant agreed to purchase land and paid Rs 25,000 as earnest money.", F),
        ("The vendor forfeited the amount when the sale deed was not executed.", F),
        ("The question is whether a party may forfeit money without proving loss.", I),
        ("Section 74 of the Contract Act limits recovery to reasonable compensation.", R),
        ("Forfeiture beyond reasonable compensation is not permissible.", R),
        ("The appeal is partly allowed.", D)]),
    ("sc:MOCK1971002", "SC", "1971-03-02", 2, "SUNDER LAL versus STATE OF MYSORE", ["[1971] 2 S.C.R. 40", "1971INSC902"], [
        ("The licence fee was demanded without any service being rendered.", F),
        ("Whether a fee can be levied without a quid pro quo.", I),
        ("A fee without any element of quid pro quo is a tax in disguise.", R),
        ("The demand is quashed.", D)]),
    ("sc:MOCK2003014", "SC", "2003-04-17", 2, "ANIL KUMAR versus NORTHERN MILLS LTD.", ["[2003] 1 S.C.R. 210", "2003INSC903"], [
        ("The employer withheld gratuity alleging misconduct.", F),
        ("Whether gratuity can be forfeited absent proof of loss.", I),
        ("Following Ramesh Traders v. Kavita Builders, [1963] 3 SCR 515, actual loss must be shown before forfeiture.", P),
        ("The withholding is set aside.", D)]),
    ("sc:MOCK2015220", "SC", "2015-07-02", 3, "MEERA DEVI versus SURAJ ESTATES", ["[2015] 6 S.C.R. 88", "2015INSC904"], [
        ("The buyer defaulted on payment under an agreement to sell.", F),
        ("Whether the earnest money clause permits forfeiture of the full deposit.", I),
        ("The ratio in Ramesh Traders v. Kavita Builders, [1963] 3 SCR 515 is distinguishable as the deposit here was nominal.", P),
        ("The appeal is dismissed.", D)]),
    ("hc:MOCK2019088", "HC", "2019-11-21", 1, "K. SELVAM versus R. LAKSHMI", [], [
        ("The tenant paid an advance which the landlord retained.", F),
        ("Whether retention of advance requires proof of damage.", I),
        ("Relying on Ramesh Traders v. Kavita Builders, [1963] 3 SCR 515, the landlord must prove loss.", P),
        ("The petition is allowed.", D)]),
    ("sc:MOCK2021310", "SC", "2021-02-05", 2, "VIKRAM INFRA versus CITY CORPORATION", ["[2021] 4 S.C.R. 77", "2021INSC905"], [
        ("The contractor's security deposit was encashed.", F),
        ("Whether liquidated damages require proof of actual loss.", I),
        ("We follow the principle in Ramesh Traders v. Kavita Builders, [1963] 3 SCR 515.", P),
        ("The appeal is allowed.", D)]),
    ("sc:MOCK2022400", "SC", "2022-09-12", 5, "UNION OF INDIA versus HARI PRASAD", ["[2022] 9 S.C.R. 300", "2022INSC906"], [
        ("The levy was challenged as lacking any service in return.", F),
        ("Whether Sunder Lal v. State of Mysore lays down the correct law.", I),
        ("Sunder Lal v. State of Mysore, [1971] 2 SCR 40 is overruled; a fee need not be matched by a specific service.", P),
        ("The appeal is allowed.", D)]),
]

with open(OUT / "judgments.jsonl", "w") as f:
    for cid, court, date, bench, title, aliases, sents in DOCS:
        f.write(json.dumps({"case_id": cid, "source": "MOCK", "court": court, "date": date, "bench_size": bench,
                            "title": title, "aliases": aliases, "text": " ".join(s for s, _ in sents),
                            "sentences": [{"idx": i, "text": s, "role": r, "conf": 1.0}
                                          for i, (s, r) in enumerate(sents)]}) + "\n")

pd.DataFrame([{"case_id": cid, "sent_idx": i, "text": s, "role": r, "conf": 1.0}
              for cid, *_, sents in DOCS for i, (s, r) in enumerate(sents)]
             ).to_parquet(OUT / "segments.parquet", index=False)

E = [("sc:MOCK2003014", "sc:MOCK1963001", 2), ("sc:MOCK2015220", "sc:MOCK1963001", 2),
     ("hc:MOCK2019088", "sc:MOCK1963001", 2), ("sc:MOCK2021310", "sc:MOCK1963001", 2),
     ("sc:MOCK2022400", "sc:MOCK1971002", 2)]
sent = {(cid, i): s for cid, *_, sents in DOCS for i, (s, _) in enumerate(sents)}
ctx = lambda cid, i: " ".join(sent[(cid, j)] for j in range(i - 2, i + 3) if (cid, j) in sent)
cite = {"sc:MOCK1963001": "[1963] 3 SCR 515", "sc:MOCK1971002": "[1971] 2 SCR 40"}
edges = pd.DataFrame([{"citing_id": a, "cited_id": b, "citing_sent_idx": i, "context": ctx(a, i),
                       "cite_text": cite[b], "method": "key", "score": 100.0} for a, b, i in E])
edges.to_csv(OUT / "edges.csv", index=False)

ids = [d[0] for d in DOCS]
indeg = edges.cited_id.value_counts().to_dict()
pd.DataFrame({"case_id": ids, "pagerank": [0.30, 0.16, 0.09, 0.09, 0.09, 0.09, 0.18],
              "in_degree": [indeg.get(c, 0) for c in ids], "in_corpus": 1}).to_csv(OUT / "pagerank.csv", index=False)

json.dump([{"query_id": "q1",
            "results": [{"case_id": c, "score": round(1 - i * 0.1, 2),
                         "matched_segments": [{"sent_idx": 0, "role": "Facts"}]}
                        for i, c in enumerate(ids)]}],
          open(OUT / "retrieval.json", "w"), indent=2)

tr = lambda t: "distinguished" if "distinguishable" in t else ("overruled" if "overruled" in t else "followed")
json.dump([{"case_id": "sc:MOCK1963001", "label": "followed", "confidence": 0.82, "n_citing": 4,
            "evidence": [{"citing_id": r.citing_id, "sent": sent[(r.citing_id, r.citing_sent_idx)],
                          "treatment": tr(sent[(r.citing_id, r.citing_sent_idx)]), "weight": 1.0}
                         for r in edges[edges.cited_id == "sc:MOCK1963001"].itertuples()]},
           {"case_id": "sc:MOCK1971002", "label": "overruled", "confidence": 0.9, "n_citing": 1,
            "evidence": [{"citing_id": "sc:MOCK2022400", "sent": sent[("sc:MOCK2022400", 2)],
                          "treatment": "overruled", "weight": 2.0}]}],
          open(OUT / "validity.json", "w"), indent=2)

json.dump([{"query_id": "q1", "case_id": "sc:MOCK1963001",
            "issue": "Whether earnest money can be forfeited without proof of loss.",
            "material_facts": "Buyer paid earnest money; vendor forfeited it on non-execution of the sale deed.",
            "ratio": "Under Section 74, only reasonable compensation is recoverable; forfeiture beyond it is barred.",
            "decision": "Appeal partly allowed.",
            "relevance_to_query": "Your landlord retained your deposit without showing loss, which is the same point.",
            "rejected": []}],
          open(OUT / "summary.json", "w"), indent=2)

print("Mock files written to", OUT)
for p in sorted(OUT.iterdir()):
    print(" ", p.name, p.stat().st_size, "bytes")
