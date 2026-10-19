"""Fetch one judgment and its sentence roles without loading the whole corpus.

Judgments: a byte-offset index over paths.judgments (built once, cached next to the file, rebuilt when
the file changes). Roles, in order of preference (the source is reported as `roles_source`):
  1. paths.segments (Member A's Stage 1 hand-off; gold where available),
  2. role labels stored in the judgment itself (gold Bhattacharya roles),
  3. Member A's trained labeller (models/stage1), run on this one judgment,
  4. position fallback: first 25% Facts, last 5 sentences Ruling by Present Court, rest Ratio.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import List, Optional, Tuple

from common.config import path
from common.schema import Judgment


def _index_file(p: Path) -> Path:
    return p.with_name(p.name + ".offsets.json")


@functools.lru_cache(maxsize=2)
def offsets(p: Path) -> dict:
    st = p.stat()
    sig = f"{st.st_size}:{int(st.st_mtime)}"
    ix = _index_file(p)
    if ix.exists():
        d = json.loads(ix.read_text())
        if d.get("sig") == sig:
            return d["off"]
    off, pos = {}, 0
    with open(p, "rb") as f:
        for line in f:
            i = line.find(b'"case_id": "')
            if i >= 0:
                j = line.index(b'"', i + 12)
                off[line[i + 12:j].decode()] = pos
            pos += len(line)
    try:
        ix.write_text(json.dumps({"sig": sig, "off": off}))
    except OSError:
        pass
    return off


def get_judgment(case_id: str) -> Optional[Judgment]:
    p = path("judgments")
    o = offsets(p).get(case_id)
    if o is None:
        return None
    with open(p, "rb") as f:
        f.seek(o)
        return Judgment.from_dict(json.loads(f.readline()))


@functools.lru_cache(maxsize=1)
def _segments():
    import pandas as pd
    p = path("segments")
    if not p.exists():
        return {}
    s = pd.read_parquet(p)
    return {cid: dict(zip(g.sent_idx, g.role)) for cid, g in s.groupby("case_id")}


@functools.lru_cache(maxsize=1)
def _labeller():
    try:
        from stage1_roles.label_corpus import Labeller
        return Labeller()
    except (FileNotFoundError, ImportError, OSError):
        return None


def roles_for(doc: Judgment) -> Tuple[List[Tuple[str, str]], str]:
    """-> ([(sentence text, role)], roles_source)."""
    seg = _segments().get(doc.case_id)
    if seg:
        return [(s.text, seg.get(s.idx, "")) for s in doc.sentences], "segments"
    if doc.sentences and all(s.role for s in doc.sentences):
        return [(s.text, s.role) for s in doc.sentences], "gold_in_judgment"
    lab = _labeller()
    if lab is not None and doc.sentences:
        d = lab.label([doc])[0]
        return [(s.text, s.role) for s in d.sentences], "stage1_labeller"
    n = len(doc.sentences)
    facts_end, dec_start = max(1, n // 4), max(n - 5, n // 4 + 1)
    roles = ["Facts" if i < facts_end else "Ruling by Present Court" if i >= dec_start else "Ratio of the decision"
             for i in range(n)]
    return [(s.text, r) for s, r in zip(doc.sentences, roles)], "position_fallback"
