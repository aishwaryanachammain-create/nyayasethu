"""Citation neighbourhood of one judgment, drawn as a self-contained SVG (no JS, no CDN: works offline).

Right of the centre: later judgments that cite it, coloured by how they treated it (Stage 4 evidence).
Left of the centre: what it cites (paths.edges), each ringed by that precedent's own validity verdict,
so a judgment resting on an overruled precedent stands out.
"""
from __future__ import annotations

import functools
import html
import math
import re
from typing import Dict, List, Optional, Tuple

TREATMENT_COLOURS = {"followed": "#16a34a", "distinguished": "#ea580c", "overruled": "#dc2626",
                     "doubted": "#d97706", "neutral": "#94a3b8", "cites": "#64748b"}
VERDICT_COLOURS = {"followed": "#16a34a", "distinguished": "#ea580c", "overruled": "#dc2626",
                   "doubted": "#d97706", "insufficient_evidence": "#cbd5e1"}
_RANK = {"overruled": 4, "doubted": 3, "distinguished": 2, "followed": 1, "neutral": 0}


@functools.lru_cache(maxsize=1)
def _edges():
    import pandas as pd
    from common.config import path
    p = path("edges")
    if not p.exists():
        return None
    return pd.read_csv(p, usecols=["citing_id", "cited_id", "cite_text"]).groupby("citing_id")


@functools.lru_cache(maxsize=1)
def _nodes() -> Dict[str, Tuple[str, Optional[str]]]:
    from common.config import cfg, path
    p = path("edges").parent / "validity_nodes.parquet"
    if cfg().get("use_mock") or not p.exists():
        return {}
    import pandas as pd
    d = pd.read_parquet(p, columns=["case_id", "title", "date"])
    return {c: (t, dt) for c, t, dt in zip(d.case_id, d.title, d.date)}


@functools.lru_cache(maxsize=4096)
def node_info(case_id: str) -> Tuple[str, Optional[str]]:
    """(title, date) from the graph's node table, else the corpus, else the bare id."""
    hit = _nodes().get(case_id)
    if hit:
        return hit[0] or case_id, hit[1]
    try:
        from stage5_summary.docs import get_judgment
        d = get_judgment(case_id)
        if d:
            return d.title or case_id, d.date
    except Exception:            # noqa: BLE001  corpus missing: the id is still a usable label
        pass
    return case_id, None


def short_title(title: str, n: int = 30) -> str:
    t = re.sub(r"\s+(versus|vs\.?|v\.)\s+", " v. ", title or "", flags=re.I)
    t = re.sub(r"\s*(&|AND)\s+(ANR|ORS|OTHERS|ANOTHER)\.?\b.*?(?= v\. |$)", "", t, flags=re.I)
    from ui.core import display_title
    t = display_title(re.sub(r"\s+", " ", t).strip().replace(" v. ", " versus "))
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def neighbourhood(case_id: str, validity: Optional[Dict], max_citing: int = 14, max_cited: int = 10) -> Dict:
    """{'citing': [...], 'cited': [...], 'n_citing', 'n_cited'}; each node has id, title, year, kind, sent."""
    by_citer: Dict[str, Dict] = {}
    for e in (validity or {}).get("evidence") or []:
        t = e.get("treatment") or "neutral"
        cur = by_citer.get(e["citing_id"])
        if cur is None or (_RANK.get(t, 0), e.get("weight") or 0) > (_RANK.get(cur["kind"], 0), cur["w"]):
            by_citer[e["citing_id"]] = {"id": e["citing_id"], "kind": t, "w": e.get("weight") or 0,
                                        "sent": (e.get("sent") or "").replace("[[", "").replace("]]", "")}
    citing = sorted(by_citer.values(), key=lambda x: (-_RANK.get(x["kind"], 0), -x["w"]))[:max_citing]
    cited: List[Dict] = []
    g = _edges()
    if g is not None and case_id in g.groups:
        rows = g.get_group(case_id).drop_duplicates("cited_id")
        from app import b_interface
        for cid, text in zip(rows.cited_id, rows.cite_text):
            lab = (b_interface.get_validity(cid) or {}).get("label") or "insufficient_evidence"
            cited.append({"id": cid, "kind": lab, "sent": str(text)})
        cited.sort(key=lambda x: -_RANK.get(x["kind"], 0) if x["kind"] in _RANK else 0)
    n_cited = len(cited)
    cited = cited[:max_cited]
    for n in citing + cited:
        title, date = node_info(n["id"])
        n["title"], n["year"] = title, (str(date)[:4] if date else "")
    return {"citing": citing, "cited": cited, "n_citing": len(by_citer), "n_cited": n_cited}


def _arc(n: int, start: float, end: float) -> List[float]:
    if n == 1:
        return [(start + end) / 2]
    return [start + (end - start) * i / (n - 1) for i in range(n)]


def graph_svg(case_id: str, title: str, nb: Dict, verdict: Optional[str] = None) -> str:
    W, H, cx, cy, rx, ry = 860, 480, 430, 225, 205, 185
    esc = html.escape
    defs = "".join(f'<marker id="ar-{k}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                   f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{c}"/></marker>'
                   for k, c in TREATMENT_COLOURS.items())
    edges, nodes = [], []

    def place(items, angles, side):
        for n, a in zip(items, angles):
            x, y = cx + rx * math.cos(math.radians(a)), cy + ry * math.sin(math.radians(a))
            if side == "citing":       # citing -> centre, coloured by treatment
                col, fill, ring = TREATMENT_COLOURS.get(n["kind"], "#94a3b8"), TREATMENT_COLOURS.get(n["kind"], "#94a3b8"), "#fff"
                x1, y1, x2, y2, mk = x, y, cx + 46 * math.cos(math.radians(a)), cy + 46 * math.sin(math.radians(a)), n["kind"]
                dash, tip = "", f"{n['kind'].upper()} by this judgment"
            else:                      # centre -> cited, grey, ringed by the cited case's own verdict
                col, fill, ring = TREATMENT_COLOURS["cites"], "#fff", VERDICT_COLOURS.get(n["kind"], "#cbd5e1")
                x1, y1 = cx + 46 * math.cos(math.radians(a)), cy + 46 * math.sin(math.radians(a))
                x2, y2, mk = x - 11 * math.cos(math.radians(a)), y - 11 * math.sin(math.radians(a)), "cites"
                dash, tip = ' stroke-dasharray="5 4"', f"Cited here. Its own verdict: {n['kind'].replace('_', ' ')}"
            w = 2.6 if n["kind"] in ("overruled", "doubted") else 1.6
            edges.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
                         f'stroke-width="{w}"{dash} marker-end="url(#ar-{mk})" opacity=".85"/>')
            anchor, dx = ("start", 16) if side == "citing" else ("end", -16)
            label = esc(short_title(n["title"]))
            sent = esc((n.get("sent") or "")[:280])
            nodes.append(f'<g class="nd"><title>{esc(n["title"])} ({esc(n["year"] or "n.d.")})&#10;{esc(tip)}&#10;&#10;{sent}</title>'
                         f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{fill}" stroke="{ring}" stroke-width="3"/>'
                         f'<text x="{x + dx:.1f}" y="{y - 1:.1f}" text-anchor="{anchor}" class="lb">{label}</text>'
                         f'<text x="{x + dx:.1f}" y="{y + 12:.1f}" text-anchor="{anchor}" class="yr">{esc(n["year"])}</text></g>')

    place(nb["citing"], _arc(len(nb["citing"]), -62, 62), "citing")
    place(nb["cited"], _arc(len(nb["cited"]), 118, 242), "cited")
    ring = VERDICT_COLOURS.get(verdict or "", "#1f3a5f")
    centre = (f'<g><title>{esc(title)}</title><circle cx="{cx}" cy="{cy}" r="40" fill="#1f3a5f" stroke="{ring}" '
              f'stroke-width="6"/><text x="{cx}" y="{cy - 2}" text-anchor="middle" class="cl">THIS</text>'
              f'<text x="{cx}" y="{cy + 13}" text-anchor="middle" class="cl">CASE</text></g>')
    hdr = (f'<text x="22" y="24" class="hd">Cites ({nb["n_cited"]})</text>'
           f'<text x="{W - 22}" y="24" text-anchor="end" class="hd">Cited by ({nb["n_citing"]})</text>')
    legend, lx = [], 22
    for k in ("followed", "distinguished", "overruled", "neutral"):
        legend.append(f'<circle cx="{lx + 6}" cy="{H - 16}" r="6" fill="{TREATMENT_COLOURS[k]}"/>'
                      f'<text x="{lx + 17}" y="{H - 12}" class="lg">{k}</text>')
        lx += 112
    legend.append(f'<line x1="{lx}" y1="{H - 16}" x2="{lx + 26}" y2="{H - 16}" stroke="#64748b" stroke-dasharray="5 4" '
                  f'stroke-width="1.6"/><text x="{lx + 32}" y="{H - 12}" class="lg">cites (ring = its verdict)</text>')
    empty = ""
    if not nb["citing"] and not nb["cited"]:
        empty = f'<text x="{cx}" y="{cy + 80}" text-anchor="middle" class="lg">No citation links in the graph for this judgment.</text>'
    style = ("<style>.lb{font:600 11.5px system-ui,sans-serif;fill:#1f2937}.yr{font:11px system-ui,sans-serif;fill:#6b7280}"
             ".cl{font:700 11px system-ui,sans-serif;fill:#fff;letter-spacing:.08em}.hd{font:700 12px system-ui,sans-serif;"
             "fill:#1f3a5f;letter-spacing:.06em;text-transform:uppercase}.lg{font:12px system-ui,sans-serif;fill:#4b5563}"
             ".nd{cursor:help}.nd:hover circle{stroke:#c8a24a}.nd:hover .lb{fill:#1f3a5f;text-decoration:underline}</style>")
    return (f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="Citation graph of {esc(title)}" '
            f'xmlns="http://www.w3.org/2000/svg" style="background:#fbfaf7;border-radius:12px">{style}<defs>{defs}</defs>'
            f'{hdr}{"".join(edges)}{centre}{"".join(nodes)}{empty}{"".join(legend)}</svg>')
