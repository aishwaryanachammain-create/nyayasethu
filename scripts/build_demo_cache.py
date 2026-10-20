"""Save results for the demo UI's example queries (ui/demo_cache.json): the fallback if the engine or API
fails during a demo, plus Legal-LED summaries and LLM comparisons the UI can show instantly.

    python scripts/build_demo_cache.py                     # engine results for every example query
    python scripts/build_demo_cache.py --abstractive 3     # + Legal-LED/NLI summaries of the top 3 and any overruled hit
    python scripts/build_demo_cache.py --llm               # + live LLM comparison for queries outside eval/llm_only

Queries q1-q5 already have Legal-LED summaries (eval/stage5_pairs.jsonl) and LLM replies (eval/llm_only/), so
--abstractive and --llm only run the other example queries.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui import core  # noqa: E402

KEEP_EVIDENCE = 40                    # strongest evidence rows kept per hit, to keep the file small


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--abstractive", type=int, default=0, help="summarise this many top hits per new query")
    ap.add_argument("--llm", action="store_true")
    a = ap.parse_args()
    from app.engine import Engine
    from ui import llm_compare
    eng = Engine(load_dense=True)
    cache = core.demo_cache()
    cache.setdefault("abstractive", {})
    cache.setdefault("llm", {})
    cache["search"] = {}
    resolver = None
    for d in core.DEMO_QUERIES:
        q, key = d["query"], core._norm(d["query"])
        out = core.engine_search(eng, q, a.k)
        for r in out["results"]:
            v = r.get("validity") or {}
            if v.get("evidence"):
                v["evidence"] = sorted(v["evidence"], key=lambda e: -(e.get("weight") or 0))[:KEEP_EVIDENCE]
        cache["search"][key] = out
        labels = [(r.get("validity") or {}).get("label") for r in out["results"][:10]]
        print(f"{d['label']}: {len(out['results'])} hits, top-10 labels {labels}")
        if d["qid"]:
            continue
        if a.abstractive:
            from stage5_summary import summarize
            want = [r["case_id"] for r in out["results"][:a.abstractive]]
            want += [r["case_id"] for r in out["results"][:10]
                     if (r.get("validity") or {}).get("label") in ("overruled", "doubted") and r["case_id"] not in want]
            for cid in want:
                if cid in cache["abstractive"].get(key, {}):
                    continue
                import time
                t0 = time.time()
                s = summarize(q, cid, mode="abstractive")
                s["seconds"] = round(time.time() - t0, 1)
                cache["abstractive"].setdefault(key, {})[cid] = s
                print(f"  abstractive {cid}: {s['seconds']} s, {len(s.get('rejected') or [])} sentence(s) rejected")
        if a.llm and (cache["llm"].get(key) or {}).get("rows") in (None, []):
            if resolver is None:
                from baseline_llm.run import load_resolver
                resolver = load_resolver()
            cmp = llm_compare.live(q, resolver)
            h = llm_compare.headline(cmp)
            if cmp.get("raw_empty"):
                print("  LLM: empty reply three times, not cached")
                continue
            cache["llm"][key] = cmp
            print(f"  LLM: {h['named']} cases named, {h['exist']} found in the graph")
    core.DEMO_CACHE.write_text(json.dumps(cache, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"-> {core.DEMO_CACHE} ({core.DEMO_CACHE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
