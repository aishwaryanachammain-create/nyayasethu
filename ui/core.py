"""UI logic without Streamlit (tested in tests/test_ui.py): mode setup, search backends, badges.

Backends, chosen in ui/app.py:
  mock    the contract stand-ins in paths.mock (fixed retrieval list, mock validity, mock judgments)
  engine  Member A's retrieval engine in-process (app/engine.py; needs models/index_corpus)
  api     Member A's FastAPI server (POST /search with Member B's validity + summary per hit)
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from common.config import ROOT

FIELD_NAMES = {"issue": "Issue", "material_facts": "Material facts", "ratio": "Ratio", "decision": "Decision",
               "relevance_to_query": "Relevance to your query"}

# validity label -> (badge text, Streamlit badge colour, one-line meaning)
BADGES = {
    "followed": ("Followed", "green", "Later judgments mostly follow or apply it"),
    "distinguished": ("Distinguished", "orange", "Later judgments mostly distinguish it on facts or law"),
    "doubted": ("Doubted", "orange", "An overruling we could not confirm (smaller or unknown bench)"),
    "overruled": ("Overruled", "red", "Overruled by a bench at least as large: not good law"),
    "insufficient_evidence": ("Insufficient evidence", "gray", "Fewer than 3 citing judgments with a treatment"),
}


def badge(v: Optional[Dict]):
    if not v or v.get("status") == "pending" or not v.get("label"):
        return "Validity pending", "gray", (v or {}).get("note", "Stage 4 output not available")
    return BADGES.get(v["label"], (v["label"], "gray", ""))


def use_mock_config() -> Path:
    """Point common.config at a copy of config.yaml with use_mock: true (must run before cfg() is first used)."""
    c = yaml.safe_load((ROOT / "config.yaml").read_text())
    c["use_mock"] = True
    p = Path(tempfile.gettempdir()) / "nyayasethu_config_mock.yaml"
    p.write_text(yaml.safe_dump(c))
    os.environ["NYAYASETHU_CONFIG"] = str(p)
    from common import config
    config.cfg.cache_clear()
    return p


def mock_search(query: str, k: int = 10) -> Dict:
    """Mock retrieval is a fixed list (paths.mock/retrieval.json): it ignores the query."""
    from common.config import path
    from stage5_summary.docs import get_judgment
    runs = json.loads(path("retrieval").read_text(encoding="utf-8"))
    results = []
    for r in (runs[0]["results"] if runs else [])[:k]:
        d = get_judgment(r["case_id"])
        results.append({"case_id": r["case_id"], "score": r.get("score"), "title": d.title if d else "",
                        "court": d.court if d else None, "date": d.date if d else None,
                        "bench_size": d.bench_size if d else None, "matched_segments": r.get("matched_segments")})
    return {"query": query, "results": results, "components": {"retrieval": "mock (fixed list, ignores the query)"}}


def add_member_b(out: Dict, query: str) -> Dict:
    """Attach validity + (fast, API-mode) summary to each hit, the way A's /search does."""
    from app import b_interface
    for r in out["results"]:
        r.setdefault("validity", b_interface.get_validity(r["case_id"]))
        r.setdefault("summary", b_interface.summarize(query, r["case_id"]))
    return out


def engine_search(engine, query: str, k: int) -> Dict:
    return add_member_b(engine.search(query, k), query)


def api_search(url: str, query: str, k: int) -> Dict:
    import httpx
    r = httpx.post(f"{url.rstrip('/')}/search", json={"query": query, "k": k, "with_b": True},
                   timeout=httpx.Timeout(600, connect=5))          # fail fast if the server is down: the UI falls back
    r.raise_for_status()
    return r.json()


def evidence_rows(v: Optional[Dict], limit: int = 10) -> List[Dict]:
    """Evidence sentences for the expander, strongest first; the cited span is bolded."""
    ev = sorted((v or {}).get("evidence") or [], key=lambda e: -(e.get("weight") or 0) * (e.get("confidence") or 1))
    return [{**e, "sent_md": (e.get("sent") or "").replace("[[", "**").replace("]]", "**")} for e in ev[:limit]]


# ---------------------------------------------------------------- demo queries and caches
# The five queries of the LLM-only baseline (app/latency_test.py, eval/llm_only/), plus two whose results include
# an overruled precedent, so the red "not good law" warning shows up in the demo.
DEMO_QUERIES = [
    {"label": "Gratuity withheld", "qid": "q1", "query": "My employer withheld my entire gratuity because I resigned without serving the notice period. Can they keep it without proving any loss?"},
    {"label": "Police confession", "qid": "q2", "query": "The accused was convicted on the basis of a confession recorded by a police officer. Is such a confession admissible?"},
    {"label": "Eviction for owner's son", "qid": "q3", "query": "A tenant was evicted on the ground that the landlord bona fide required the premises for his son. Is the requirement valid?"},
    {"label": "Lease cancelled, no hearing", "qid": "q4", "query": "The government cancelled a mining lease without giving the lessee a hearing. Was natural justice violated?"},
    {"label": "Look-alike trademark", "qid": "q5", "query": "A trademark owner claims a rival's similar label for the same goods deceives customers. Is this passing off?"},
    {"label": "Daughter's property share", "qid": None, "overruled": True, "query": "My father died before the 2005 amendment to the Hindu Succession Act. Can I as a daughter claim an equal coparcenary share in our ancestral property?"},
    {"label": "Unstamped arbitration", "qid": None, "overruled": True, "query": "Our arbitration agreement is in an unstamped contract. Can the court appoint an arbitrator before the stamp duty is paid?"},
]
DEMO_CACHE = ROOT / "ui" / "demo_cache.json"      # python -m scripts.build_demo_cache
PAIRS = ROOT / "eval" / "stage5_pairs.jsonl"      # Legal-LED + NLI summaries for q1-q5 x top 5 (Stage 5 eval)


def _norm(q: str) -> str:
    return " ".join((q or "").lower().split())


def demo_cache() -> Dict:
    try:
        return json.loads(DEMO_CACHE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def cached_search(query: str, k: int, cache: Optional[Dict] = None) -> Optional[Dict]:
    """Saved results for a demo query; for any other query, those of the closest demo query, marked as such."""
    runs = (cache if cache is not None else demo_cache()).get("search") or {}
    if not runs:
        return None
    from common.text import tokenize
    q = set(tokenize(query))
    best = runs.get(_norm(query)) and _norm(query)
    if not best:
        best = max(runs, key=lambda n: len(q & set(tokenize(runs[n]["query"]))) / (len(q | set(tokenize(runs[n]["query"]))) or 1))
    out = json.loads(json.dumps(runs[best]))
    out["results"] = out["results"][:k]
    out["cached"] = {"exact": best == _norm(query), "query": out["query"]}
    return out


def _pair_summary(p: Dict) -> Dict:
    qf = p.get("query_focused") or {}
    rej = p.get("qf_rejected") or []
    return {"status": "ok", "case_id": p["case_id"], "query": p["query"], "title": p.get("title"),
            **{f: qf.get(f, "") for f in FIELD_NAMES},
            "rejected": [f"{r.get('field')}: {r.get('sentence')}" for r in rej], "rejected_detail": rej,
            "provenance": p.get("provenance"), "roles_source": p.get("roles_source"),
            "generator": p.get("generator"), "nli": p.get("nli_model"),
            "seconds": p.get("seconds_query_focused"), "precomputed": True}


def cached_abstractive(query: str, case_id: str, cache: Optional[Dict] = None) -> Optional[Dict]:
    """A Legal-LED + NLI summary computed ahead of time (demo cache, then the Stage 5 eval pairs), or None."""
    hit = ((cache if cache is not None else demo_cache()).get("abstractive") or {}).get(_norm(query), {}).get(case_id)
    if hit:
        return {**hit, "precomputed": True}
    try:
        lines = PAIRS.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        p = json.loads(line)
        if p["case_id"] == case_id and _norm(p["query"]) == _norm(query):
            return _pair_summary(p)
    return None


# ---------------------------------------------------------------- headline numbers
def _load(rel: str) -> Dict:
    try:
        return json.loads((ROOT / rel).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def project_stats() -> List[Dict]:
    """Stat tiles from the committed build and evaluation logs; a tile whose log is missing is left out."""
    g, agg, prec = _load("logs/graph_build.json"), _load("logs/stage4_aggregate.json"), _load("eval/graph_precision.json")
    from ui.llm_compare import totals
    llm = totals()
    tiles = []
    if g.get("corpus_docs"):
        tiles.append({"value": f"{g['corpus_docs']:,}", "label": "judgments searched", "sub": "full-text judgments in the index"})
    if g.get("mentions_resolved"):
        tiles.append({"value": f"{g['mentions_resolved'] / 1000:.1f}k", "label": "citations resolved",
                      "sub": f"into a graph of {g.get('metadata_nodes', 0):,} cases"})
    if agg.get("by_label"):
        decided = sum(n for lab, n in agg["by_label"].items() if lab != "insufficient_evidence")
        tiles.append({"value": f"{decided:,}", "label": "validity verdicts",
                      "sub": f"{agg['by_label'].get('overruled', 0)} overruled cases flagged"})
    if (prec.get("extraction") or {}).get("precision"):
        tiles.append({"value": f"{prec['extraction']['precision']:.0%}", "label": "citation precision",
                      "sub": f"human check of {prec['extraction']['checked']} links"})
    if llm.get("named"):
        tiles.append({"value": f"{llm['confirmed']} of {llm['named']}", "label": "plain-LLM cases that exist",
                      "sub": "the rest could not be verified"})
    return tiles


# ---------------------------------------------------------------- filters, matches, ranking trace
def year_of(r: Dict) -> Optional[int]:
    d = str(r.get("date") or "")
    return int(d[:4]) if d[:4].isdigit() else None


def apply_filters(results: List[Dict], courts=None, years=None, hide_overruled: bool = False,
                  min_bench: int = 1) -> List[Dict]:
    """Sidebar filters. Unknown court, year or bench size never hides a result."""
    out = []
    for r in results:
        y, b = year_of(r), r.get("bench_size")
        if courts and r.get("court") and r["court"] not in courts:
            continue
        if years and y is not None and not years[0] <= y <= years[1]:
            continue
        if hide_overruled and (r.get("validity") or {}).get("label") == "overruled":
            continue
        if min_bench > 1 and b and b < min_bench:
            continue
        out.append(r)
    return out


def query_matches(sents: List, query: str, n: int = 5) -> List[int]:
    """Indices of the n sentences sharing the most query terms (length-normalised, as in app/engine.py)."""
    from common.text import tokenize
    q = set(tokenize(query))
    scored = []
    for i, (t, _) in enumerate(sents):
        toks = set(tokenize(t))
        sc = len(q & toks) / (len(toks) ** 0.5 + 1) if toks else 0.0
        if sc > 0:
            scored.append((sc, i))
    return sorted(i for _, i in sorted(scored, reverse=True)[:n])


def overruling(v: Optional[Dict]) -> Optional[Dict]:
    """The strongest 'overruled' evidence row (who overruled the case), if any."""
    rows = [e for e in (v or {}).get("evidence") or [] if e.get("treatment") == "overruled"]
    return max(rows, key=lambda e: e.get("weight") or 0) if rows else None


def ranking_trace(r: Dict, alpha: Optional[float], components: Dict) -> List[Dict]:
    """Stage-by-stage view of how one hit was ranked (Stages 2-3)."""
    steps = []
    for run, rank in sorted((r.get("surfaced_by") or {}).items(), key=lambda x: x[1]):
        kind, _, part = run.partition(":")
        steps.append({"stage": f"{kind.upper() if kind == 'bm25' else kind.title()} · {part or 'full'}",
                      "value": f"#{rank}", "note": "rank in this run (top 100 go to fusion)"})
    if steps:
        steps.append({"stage": "RRF fusion", "value": f"{len(steps)} run(s)", "note": "reciprocal-rank fusion"})
    ce_missing = str(components.get("cross-encoder", "")).startswith("missing")
    steps.append({"stage": "Cross-encoder", "value": "off" if ce_missing else f"{r.get('ce', 0):.3f}",
                  "note": "not loaded: rerank by authority only" if ce_missing else "min-max normalised relevance"})
    if r.get("prior") is not None:
        steps.append({"stage": "Authority prior", "value": f"{r['prior']:.3f}",
                      "note": str(components.get("prior", "citation PageRank"))})
    if r.get("score") is not None:
        a = f"{alpha:g}" if isinstance(alpha, (int, float)) else "α"
        steps.append({"stage": "Final score", "value": f"{r['score']:.4f}", "note": f"{a}·CE + (1−{a})·prior"})
    return steps


# ---------------------------------------------------------------- case brief export
def _summary_for(r: Dict, abstractive: Optional[Dict]) -> Dict:
    s = abstractive if abstractive and abstractive.get("status") == "ok" else r.get("summary")
    return s if s and s.get("status") == "ok" else {}


def brief_markdown(out: Dict, query: str, abstractive: Optional[Dict] = None, today: str = "") -> str:
    """The research brief a user takes away: query, then each precedent with its validity and summary."""
    abstractive = abstractive or {}
    lines = ["# NyayaSethu research brief", "", f"**Your question.** {query}", ""]
    if today:
        lines += [f"*Prepared {today}. A research aid, not legal advice: read the judgments before relying on them.*", ""]
    for i, r in enumerate(out.get("results") or [], 1):
        v = r.get("validity")
        text, _, meaning = badge(v)
        meta = " · ".join(str(b) for b in (r.get("court"), r.get("date"),
                                             f"bench of {r['bench_size']}" if r.get("bench_size") else None,
                                             r["case_id"]) if b)
        lines += [f"## {i}. {display_title(r.get('title')) or r['case_id']}", "", meta, "", f"**Validity: {text}.** {meaning}"
                  + (f" (confidence {v['confidence']:.2f})" if isinstance((v or {}).get("confidence"), (int, float)) else ""), ""]
        o = overruling(v)
        if o:
            lines += [f"> **Not good law.** Overruled in `{o.get('citing_id')}`: "
                      f"{(o.get('sent') or '').replace('[[', '').replace(']]', '')}", ""]
        s = _summary_for(r, abstractive.get(r["case_id"]))
        for f, name in FIELD_NAMES.items():
            if s.get(f):
                lines.append(f"- **{name}.** {tidy(s[f])}")
        lines.append("")
    return "\n".join(lines)


def brief_html(out: Dict, query: str, abstractive: Optional[Dict] = None, today: str = "") -> str:
    """The same brief as a print-ready page (open it, then Print > Save as PDF)."""
    import html as h
    abstractive = abstractive or {}
    colour = {"green": "#15803d", "orange": "#c2410c", "red": "#b91c1c", "gray": "#4b5563"}
    parts = []
    for i, r in enumerate(out.get("results") or [], 1):
        v = r.get("validity")
        text, c, meaning = badge(v)
        meta = " · ".join(h.escape(str(b)) for b in (r.get("court"), r.get("date"),
                                                      f"bench of {r['bench_size']}" if r.get("bench_size") else None,
                                                      r["case_id"]) if b)
        o = overruling(v)
        warn = (f'<p class="warn"><b>Not good law.</b> Overruled in {h.escape(str(o.get("citing_id")))}: '
                f'{h.escape((o.get("sent") or "").replace("[[", "").replace("]]", ""))}</p>') if o else ""
        s = _summary_for(r, abstractive.get(r["case_id"]))
        fields = "".join(f"<dt>{name}</dt><dd>{h.escape(tidy(s[f]))}</dd>" for f, name in FIELD_NAMES.items() if s.get(f))
        parts.append(f'<section><h2>{i}. {h.escape(display_title(r.get("title")) or r["case_id"])}</h2><p class="meta">{meta}</p>'
                     f'<p><span class="badge" style="background:{colour.get(c, "#4b5563")}">{h.escape(text)}</span> '
                     f'{h.escape(meaning)}</p>{warn}<dl>{fields}</dl></section>')
    stamp = f"Prepared {h.escape(today)}. " if today else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>NyayaSethu research brief</title>
<style>body{{font:15px/1.55 Georgia,serif;color:#1f2937;max-width:780px;margin:40px auto;padding:0 20px}}
h1{{font-family:system-ui,sans-serif;color:#1f3a5f;margin-bottom:4px}}h2{{font-family:system-ui,sans-serif;font-size:18px;color:#1f3a5f;margin:0 0 4px}}
section{{border-top:1px solid #e5e7eb;padding:18px 0;break-inside:avoid}}.meta{{color:#6b7280;font-size:13px;margin:0 0 8px}}
.badge{{color:#fff;border-radius:999px;padding:2px 10px;font:600 12px system-ui,sans-serif}}
.warn{{background:#fef2f2;border-left:4px solid #b91c1c;padding:8px 12px}}dt{{font-weight:700;margin-top:6px}}dd{{margin:0 0 0 0}}
.q{{background:#f8fafc;border-left:4px solid #c8a24a;padding:10px 14px}}.note{{color:#6b7280;font-size:13px}}</style></head>
<body><h1>NyayaSethu research brief</h1><p class="note">{stamp}A research aid, not legal advice: read the judgments before relying on them.</p>
<p class="q"><b>Your question.</b> {h.escape(query)}</p>{''.join(parts)}</body></html>"""


_CLUTTER = [r"\bA B C D E F G H\b", r"SUPREME COURT REPORTS\s*\[\d{4}\]\s*\d+\s*S\.\s?C\.\s?R\.?",
            r"\[(?:Paras?\s[^\]]{0,40}|\d+-[A-H](?:-[A-H])?(?:[;,]\s*\d+-[A-H](?:-[A-H])?)*)\]"]


def tidy(text: str) -> str:
    """Display-only clean-up of reporter furniture in sentences copied from SCR judgments (letter rulers,
    running heads, page-margin markers). The stored summary is unchanged."""
    import re
    t = text or ""
    for pat in _CLUTTER:
        t = re.sub(pat, " ", t)
    return re.sub(r"\s{2,}", " ", t).strip(" ;:,")


def snippet(s: Optional[Dict], n: int = 260) -> str:
    """One-line 'what was held' for a collapsed card: the ratio, else the decision, cut at a word boundary."""
    if not s or s.get("status") != "ok":
        return ""
    t = tidy(s.get("ratio") or s.get("decision") or s.get("issue") or "")
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + " …"


_KEEP_UPPER = {"UOI", "LRS", "LIC", "ONGC", "NHAI", "CBI", "SBI", "RBI", "SEBI", "DDA", "UP", "MP", "AP", "II", "III", "IV"}


def display_title(title: str) -> str:
    """'VINEETA SHARMA versus RAKESH SHARMA & ORS.' -> 'Vineeta Sharma v. Rakesh Sharma & Ors.' (display only)."""
    import re
    if not title or not title.isupper() and "versus" not in title:
        return title or ""

    def word(w: str) -> str:
        core_ = re.sub(r"[^A-Za-z]", "", w)
        if core_.lower() in {"and", "of", "the", "through", "thr"}:
            return w.lower()
        if core_.upper() in _KEEP_UPPER or (len(core_) <= 2 and "." in w and core_.isupper()):
            return w                                   # initials (K. M.) and known acronyms stay upper case
        return w[:1].upper() + w[1:].lower()
    t = re.sub(r"\s+versus\s+", " v. ", title.strip(), flags=re.I)
    t = " ".join(word(w) if w != "v." else w for w in t.split())
    return t[:1].upper() + t[1:]
