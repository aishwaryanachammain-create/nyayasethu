"""Step 28 — Review 2 demo: query in -> top 10 with scores and matched segments out.

    python -m stage3_rerank.demo_cli "My employer withheld my gratuity because I resigned without notice..."
    python -m stage3_rerank.demo_cli --json "..."
"""
import argparse
import json
import textwrap

from app.engine import Engine


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-dense", action="store_true")
    a = p.parse_args()
    eng = Engine(load_dense=not a.no_dense)
    out = eng.search(a.query, a.k)
    if a.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return
    print(f"components: {out['components']}")
    print(f"query facts: {out['facts'][:200]}\n")
    w = lambda s: textwrap.fill(s, 100, initial_indent="      ", subsequent_indent="      ")
    for i, r in enumerate(out["results"], 1):
        print(f"{i:2d}. {r['case_id']}  {r['title'][:70]}  ({r['court']}, {r['date']}, bench {r['bench_size']})")
        print(f"    score {r['score']}  = ce {r['ce']} / prior {r['prior']}   surfaced by {r['surfaced_by']}")
        for key in ("matched_facts", "matched_reasoning"):
            m = r.get(key)
            if m:
                print(f"    {key.split('_')[1]} [s{m['sent_idx']} {m['role']}]:")
                print(w(m["text"][:400]))
    print(f"\ntimings (s): {out['timings']}")


if __name__ == "__main__":
    main()
