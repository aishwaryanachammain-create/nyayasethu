"""Step 32 — run 5 queries end-to-end through the API and log latency (target: under ~10 s each).

    python -m app.latency_test                 # starts the app in-process (no server needed)
    python -m app.latency_test --url http://127.0.0.1:8000
"""
import argparse
import json
import statistics
import time

from common.config import path

QUERIES = [
    "My employer withheld my entire gratuity because I resigned without serving the notice period. Can they keep it without proving any loss?",
    "The accused was convicted on the basis of a confession recorded by a police officer. Is such a confession admissible?",
    "A tenant was evicted on the ground that the landlord bona fide required the premises for his son. Is the requirement valid?",
    "The government cancelled a mining lease without giving the lessee a hearing. Was natural justice violated?",
    "A trademark owner claims a rival's similar label for the same goods deceives customers. Is this passing off?",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url")
    a = p.parse_args()
    if a.url:
        import httpx
        call = lambda q: httpx.post(f"{a.url}/search", json={"query": q}, timeout=120).json()
    else:
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        client.__enter__()                                  # run startup (model loading)
        call = lambda q: client.post("/search", json={"query": q}).json()
    rows = []
    for q in QUERIES:
        t0 = time.time()
        out = call(q)
        dt = time.time() - t0
        rows.append({"query": q[:60], "seconds": round(dt, 3), "results": len(out.get("results", [])),
                     "timings": out.get("timings")})
        print(f"{dt:6.2f}s  {len(out.get('results', []))} results  {q[:70]}")
    secs = [r["seconds"] for r in rows]
    summary = {"median_s": round(statistics.median(secs), 3), "max_s": round(max(secs), 3),
               "under_10s": all(s < 10 for s in secs), "runs": rows}
    path("logs").mkdir(parents=True, exist_ok=True)
    (path("logs") / "latency.json").write_text(json.dumps(summary, indent=2))
    print(f"median {summary['median_s']}s, max {summary['max_s']}s, all under 10 s: {summary['under_10s']}")


if __name__ == "__main__":
    main()
