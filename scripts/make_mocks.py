import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import ROOT, CFG

OUT = ROOT / CFG["paths"]["mock"]
OUT.mkdir(parents=True, exist_ok=True)

DOCS = [
    ("SC_1963_001", "Supreme Court", "1963-01-10", 5, [
        ("The appellant agreed to purchase land and paid Rs 25,000 as earnest money.", "Facts"),
        ("The vendor forfeited the amount when the sale deed was not executed.", "Facts"),
        ("The question is whether a party may forfeit money without proving loss.", "Issue"),
        ("Section 74 of the Contract Act limits recovery to reasonable compensation.", "Reasoning"),
        ("Forfeiture beyond reasonable compensation is not permissible.", "Reasoning"),
        ("The appeal is partly allowed.", "Decision")]),
    ("SC_2003_014", "Supreme Court", "2003-04-17", 2, [
        ("The employer withheld gratuity alleging misconduct.", "Facts"),
        ("Whether gratuity can be forfeited absent proof of loss.", "Issue"),
        ("Following (1963) 3 SCR 515, actual loss must be shown before forfeiture.", "Reasoning"),
        ("The withholding is set aside.", "Decision")]),
    ("SC_2015_220", "Supreme Court", "2015-07-02", 3, [
        ("The buyer defaulted on payment under an agreement to sell.", "Facts"),
        ("Whether the earnest money clause permits forfeiture of the full deposit.", "Issue"),
        ("The ratio in (1963) 3 SCR 515 is distinguishable as the deposit here was nominal.", "Reasoning"),
        ("The appeal is dismissed.", "Decision")]),
    ("HC_2019_088", "Madras High Court", "2019-11-21", 1, [
        ("The tenant paid an advance which the landlord retained.", "Facts"),
        ("Whether retention of advance requires proof of damage.", "Issue"),
        ("Relying on AIR 1963 SC 1405, the landlord must prove loss.", "Reasoning"),
        ("The petition is allowed.", "Decision")]),
    ("SC_2021_310", "Supreme Court", "2021-02-05", 2, [
        ("The contractor's security deposit was encashed.", "Facts"),
        ("Whether liquidated damages require proof of actual loss.", "Issue"),
        ("We follow the principle in (1963) 3 SCR 515.", "Reasoning"),
        ("The appeal is allowed.", "Decision")]),
]

with open(OUT / "judgments.jsonl", "w") as f:
    for cid, court, date, bench, sents in DOCS:
        f.write(json.dumps({"case_id": cid, "court": court, "date": date, "bench_size": bench,
                            "text": " ".join(s for s, _ in sents),
                            "sentences": [s for s, _ in sents]}) + "\n")

pd.DataFrame([{"case_id": cid, "sent_idx": i, "text": s, "role": r, "conf": 1.0}
              for cid, _, _, _, sents in DOCS for i, (s, r) in enumerate(sents)]
             ).to_parquet(OUT / "segments.parquet", index=False)

edges = pd.DataFrame([
    ("SC_2003_014", "SC_1963_001", 2, "Following (1963) 3 SCR 515, actual loss must be shown before forfeiture."),
    ("SC_2015_220", "SC_1963_001", 2, "The ratio in (1963) 3 SCR 515 is distinguishable as the deposit here was nominal."),
    ("HC_2019_088", "SC_1963_001", 2, "Relying on AIR 1963 SC 1405, the landlord must prove loss."),
    ("SC_2021_310", "SC_1963_001", 2, "We follow the principle in (1963) 3 SCR 515."),
], columns=["citing_id", "cited_id", "citing_sent_idx", "context"])
edges.to_csv(OUT / "edges.csv", index=False)

ids = [d[0] for d in DOCS]
pd.DataFrame({"case_id": ids, "score": [0.40, 0.15, 0.15, 0.15, 0.15]}).to_csv(OUT / "pagerank.csv", index=False)

json.dump([{"query_id": "q1",
            "results": [{"case_id": c, "score": round(1 - i * 0.1, 2),
                         "matched_segments": [{"sent_idx": 0, "role": "Facts"}]}
                        for i, c in enumerate(ids)]}],
          open(OUT / "retrieval.json", "w"), indent=2)

json.dump([{"case_id": "SC_1963_001", "label": "followed", "confidence": 0.82, "n_citing": 4,
            "evidence": [{"citing_id": r.citing_id, "sent": r.context,
                          "treatment": "distinguished" if "distinguishable" in r.context else "followed",
                          "weight": 1.0} for r in edges.itertuples()]}],
          open(OUT / "validity.json", "w"), indent=2)

json.dump([{"query_id": "q1", "case_id": "SC_1963_001",
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
    