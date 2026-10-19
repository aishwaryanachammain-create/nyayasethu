"""Query-judgment pairs for the human evaluation and the NLI rejection rate.

Pairs: each query in app/latency_test.py x Member A's top `--per-query` results (engine in-process).
Per pair, with the fine-tuned Legal-LED (models/legal_led_inabs):
  generic        one summary of the whole judgment, no query (what IN-Abs fine-tuning teaches), not filtered
  query_focused  our 5-field summary (summarize(), abstractive, NLI filter on)
The NLI filter is also run, for measurement only, on the generic summary against the whole judgment.

    python -m stage5_summary.eval_pairs              # -> eval/stage5_pairs.jsonl (resumable)
    python -m stage5_summary.eval_pairs --sheet      # -> eval/human_eval_sheet.csv, eval/human_eval_key.csv,
                                                     #    eval/stage5_nli_results.json
    python -m stage5_summary.eval_pairs --score eval/human_eval_sheet_rater1.csv   # after rating (eval/HUMAN_EVAL_GUIDE.md)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

from common.config import ROOT, cfg

from .fields import FIELDS

PAIRS = ROOT / "eval" / "stage5_pairs.jsonl"
SHEET = ROOT / "eval" / "human_eval_sheet.csv"
KEY = ROOT / "eval" / "human_eval_key.csv"
NLI_OUT = ROOT / "eval" / "stage5_nli_results.json"
NAMES = {"issue": "Issue", "material_facts": "Material facts", "ratio": "Ratio", "decision": "Decision",
         "relevance_to_query": "Relevance to the query"}
GENERIC_MAX_NEW = 512


def generate_pairs(per_query: int):
    from app.engine import Engine
    from app.latency_test import QUERIES
    from stage0_corpus.splitter import split_sentences
    from stage5_summary import _generator, _scorer, summarize
    from .docs import get_judgment
    from .nli import filter_text

    done = set()
    if PAIRS.exists():
        done = {(r["query_id"], r["case_id"]) for r in map(json.loads, open(PAIRS, encoding="utf-8"))}
    eng = Engine(load_dense=False)
    gen, scorer = _generator("abstractive"), _scorer()
    import torch
    with open(PAIRS, "a", encoding="utf-8") as f:
        for qi, q in enumerate(QUERIES, 1):
            qid = f"q{qi}"
            for rank, r in enumerate(eng.search(q, per_query)["results"], 1):
                if (qid, r["case_id"]) in done:
                    continue
                t0 = time.time()
                doc = get_judgment(r["case_id"])
                text = " ".join(s.text for s in doc.sentences)
                enc = gen.tok(text, truncation=True, max_length=gen.max_input, return_tensors="pt").to(gen.device)
                gam = torch.zeros_like(enc["input_ids"])
                gam[:, 0] = 1
                with torch.no_grad():
                    out = gen.model.generate(**enc, global_attention_mask=gam, max_new_tokens=GENERIC_MAX_NEW,
                                             num_beams=gen.num_beams, no_repeat_ngram_size=3, early_stopping=True)
                generic = gen.tok.decode(out[0], skip_special_tokens=True).strip()
                kept, rej = filter_text(generic, [s.text for s in doc.sentences], scorer)
                t1 = time.time()
                qf = summarize(q, r["case_id"], mode="abstractive")
                n_out = {fl: len(split_sentences(qf[fl])) if qf[fl] else 0 for fl in FIELDS}
                rec = {"query_id": qid, "query": q, "rank": rank, "case_id": r["case_id"], "title": r["title"],
                       "generator": gen.name.replace(str(ROOT) + "/", ""), "nli_model": scorer.name,
                       "generic": generic, "generic_sentences": len(split_sentences(generic)) if generic else 0,
                       "generic_rejected": [x["sentence"] for x in rej],
                       "query_focused": {fl: qf[fl] for fl in FIELDS},
                       "qf_kept_sentences": n_out, "qf_rejected": qf["rejected_detail"],
                       "provenance": qf["provenance"], "roles_source": qf["roles_source"],
                       "seconds_generic": round(t1 - t0, 1), "seconds_query_focused": round(time.time() - t1, 1)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{qid} #{rank} {r['case_id']}: generic {rec['seconds_generic']}s, 5-field "
                      f"{rec['seconds_query_focused']}s, rejected {len(rec['qf_rejected'])}", flush=True)


def qf_text(qf: dict) -> str:
    return "\n".join(f"{NAMES[f]}: {qf[f] or '(nothing kept)'}" for f in FIELDS)


def write_outputs(seed: int = 13):
    recs = [json.loads(l) for l in open(PAIRS, encoding="utf-8")]
    rng = random.Random(seed)
    rng.shuffle(recs)
    with open(SHEET, "w", newline="", encoding="utf-8") as fs, open(KEY, "w", newline="", encoding="utf-8") as fk:
        ws, wk = csv.writer(fs), csv.writer(fk)
        ws.writerow(["pair_id", "query", "case_title", "summary_A", "summary_B",
                     "A_faithful_1to5", "A_relevant_1to5", "A_useful_1to5",
                     "B_faithful_1to5", "B_relevant_1to5", "B_useful_1to5", "preferred_A_B_tie", "comments"])
        wk.writerow(["pair_id", "query_id", "case_id", "A", "B"])
        for i, r in enumerate(recs, 1):
            pid = f"p{i:02d}"
            a_is_qf = rng.random() < 0.5
            a, b = (qf_text(r["query_focused"]), r["generic"]) if a_is_qf else (r["generic"], qf_text(r["query_focused"]))
            ws.writerow([pid, r["query"], r["title"], a, b] + [""] * 8)
            wk.writerow([pid, r["query_id"], r["case_id"], "query_focused" if a_is_qf else "generic",
                         "generic" if a_is_qf else "query_focused"])
    per_field = defaultdict(lambda: {"kept": 0, "rejected": 0})
    for r in recs:
        for f in FIELDS:
            per_field[f]["kept"] += r["qf_kept_sentences"][f]
        for x in r["qf_rejected"]:
            per_field[x["field"]]["rejected"] += 1
    rate = lambda d: round(d["rejected"] / (d["kept"] + d["rejected"]), 4) if d["kept"] + d["rejected"] else None
    tot = {"kept": sum(v["kept"] for v in per_field.values()), "rejected": sum(v["rejected"] for v in per_field.values())}
    gen_tot = {"kept": sum(r["generic_sentences"] - len(r["generic_rejected"]) for r in recs),
               "rejected": sum(len(r["generic_rejected"]) for r in recs)}
    res = {"pairs": len(recs), "queries": len({r["query_id"] for r in recs}), "generator": recs[0]["generator"],
           "nli_model": recs[0]["nli_model"], "threshold": cfg()["stage5"]["nli_threshold"],
           "query_focused": {"rejection_rate": rate(tot), **tot,
                             "per_field": {f: {"rejection_rate": rate(per_field[f]), **per_field[f]} for f in FIELDS},
                             "pairs_with_an_empty_field": sum(any(not r["query_focused"][f] for f in FIELDS) for r in recs)},
           "generic_whole_judgment": {"rejection_rate": rate(gen_tot), **gen_tot,
                                      "note": "premises from the whole judgment, so this filter is more lenient than the per-field one"},
           "seconds_per_pair_mean": round(sum(r["seconds_generic"] + r["seconds_query_focused"] for r in recs) / len(recs), 1),
           "roles_source": dict(Counter(r["roles_source"] for r in recs))}
    NLI_OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def score_sheets(sheets, out=None, provenance: str = "human raters") -> dict:
    """Mean rating per system and criterion over filled cells; preference counts. Blank cells are skipped."""
    key = {r["pair_id"]: r for r in csv.DictReader(open(KEY, encoding="utf-8"))}
    vals, pref = defaultdict(list), Counter()
    for sh in sheets:
        for r in csv.DictReader(open(sh, encoding="utf-8")):
            k = key[r["pair_id"]]
            for side in ("A", "B"):
                for crit in ("faithful", "relevant", "useful"):
                    v = (r.get(f"{side}_{crit}_1to5") or "").strip()
                    if v:
                        vals[(k[side], crit)].append(int(v))
            p = (r.get("preferred_A_B_tie") or "").strip().upper()
            if p in ("A", "B"):
                pref[k[p]] += 1
            elif p == "TIE":
                pref["tie"] += 1
    if not vals and not pref:
        return {"status": "PENDING: no ratings filled in"}
    res = {"provenance": provenance, "sheets": [Path(s).name for s in sheets],
           "mean": {f"{sys}/{crit}": round(sum(v) / len(v), 3) for (sys, crit), v in sorted(vals.items())},
           "n": {f"{sys}/{crit}": len(v) for (sys, crit), v in sorted(vals.items())},
           "preferred": dict(pref)}
    Path(out or ROOT / "eval" / "human_eval_results.json").write_text(json.dumps(res, indent=2))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-query", type=int, default=5)
    p.add_argument("--sheet", action="store_true")
    p.add_argument("--score", nargs="+", help="filled-in copies of the sheet")
    p.add_argument("--out", help="results file (default eval/human_eval_results.json)")
    p.add_argument("--provenance", default="human raters", help="who filled the sheets")
    a = p.parse_args()
    if a.score:
        print(json.dumps(score_sheets(a.score, a.out, a.provenance), indent=2))
    elif a.sheet:
        write_outputs()
    else:
        generate_pairs(a.per_query)


if __name__ == "__main__":
    main()
