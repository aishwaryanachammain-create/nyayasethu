"""End-to-end test of every Member-A step on the fictional fixture (offline, ~2 minutes on CPU).

Numbers from the fixture are meaningless (11 invented judgments, random tiny models); the test checks
that each step runs and produces the hand-off files in the agreed format.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    pytest.importorskip("torch")
    pytest.importorskip("torchcrf")
    root = tmp_path_factory.mktemp("ws")
    sys.path.insert(0, str(REPO / "tests" / "fixtures"))
    from make_fixture import build
    cfg = build(root)
    return root, {**os.environ, "NYAYASETHU_CONFIG": str(cfg), "PYTHONPATH": str(REPO)}


def run(env, *args):
    r = subprocess.run([sys.executable, "-m", *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"{args} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def test_member_a_pipeline(ws):
    root, env = ws
    # Stage 1: CV training + final model + segments.parquet hand-off
    run(env, "stage1_roles.train", "--cv", "2")
    rep = json.loads((root / "logs" / "stage1_cv_lsa.json").read_text())
    assert {"macro_f1", "per_class", "majority_baseline_accuracy", "gate"} <= set(rep)
    run(env, "stage1_roles.label_corpus")
    import pandas as pd
    seg = pd.read_parquet(root / "data" / "segments.parquet")
    assert list(seg.columns) == ["case_id", "sent_idx", "text", "role", "conf"] and len(seg) > 50

    # Stage 0: SC-PCR benchmark (queries/candidates disjoint, split by year)
    run(env, "stage0_corpus.build_sc_pcr", "--min-cited", "1", "--test-frac", "0.34", "--dev-frac", "0.17")
    pcr = root / "data" / "processed" / "sc-pcr"
    q = [json.loads(l) for l in open(pcr / "test_queries.jsonl")]
    cands = {json.loads(l)["id"] for l in open(pcr / "candidates.jsonl")}
    assert q and not ({x["id"] for x in q} & cands)
    assert "<CITATION>" in q[0]["text"] or "<CASE>" in q[0]["text"]

    # Stage 2: BM25 baseline + overlap, bi-encoder, indexes
    out = run(env, "stage2_retrieval.bm25_baseline", "--bench", "aila", "--split", "all")
    assert "MAP" in out and "lexical overlap" in out
    run(env, "stage2_retrieval.train_biencoder", "--epochs", "1")
    run(env, "stage2_retrieval.build_index", "--corpus")
    assert (root / "models" / "index_corpus" / "dense_facts.faiss").exists()

    # Stage 3: cross-encoder, alpha tuning, ablation, error analysis, CLI demo
    run(env, "stage3_rerank.cross_encoder", "train")
    out = run(env, "stage3_rerank.rerank", "--bench", "sc-pcr")
    assert "alpha (tuned on dev)" in out
    out = run(env, "eval.ablation", "--bench", "sc-pcr")
    assert "BM25 (full text)" in out and "+ role partitioning" in out and "+ PageRank" in out
    run(env, "eval.error_analysis", "--bench", "sc-pcr", "--n", "1")
    out = run(env, "stage3_rerank.demo_cli", "gratuity withheld after resignation without notice", "--k", "3")
    assert "score" in out

    # Backend: /search, /case, latency
    out = run(env, "app.latency_test")
    assert "all under 10 s: True" in out


def test_metrics_agree_with_ranx():
    pytest.importorskip("ranx")
    from eval.metrics import evaluate_run, score
    run = {"q1": ["a", "b", "c", "d"], "q2": ["x", "y", "z"]}
    qr = {"q1": {"b": 1, "d": 1}, "q2": {"z": 1, "w": 1}}
    a, b = score(run, qr), evaluate_run(run, qr)
    for k in ("MAP", "nDCG@10", "P@10", "MRR", "Recall@100"):
        assert abs(a[k] - b[k]) < 1e-9, k
