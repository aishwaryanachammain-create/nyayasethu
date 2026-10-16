"""Reading/writing the hand-off files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from .schema import Judgment


def write_jsonl(rows: Iterable[Dict], path: str | Path) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def save_judgments(docs: Iterable[Judgment], path: str | Path, with_roles: bool = False) -> int:
    return write_jsonl((d.to_dict(with_roles) for d in docs), path)


def load_judgments(path: str | Path) -> List[Judgment]:
    return [Judgment.from_dict(r) for r in read_jsonl(path)]


def save_segments(docs: Iterable[Judgment], path: str | Path) -> int:
    """segments.parquet: {case_id, sent_idx, text, role, conf}"""
    import pandas as pd
    rows = [{"case_id": d.case_id, "sent_idx": s.idx, "text": s.text, "role": s.role,
             "conf": float(s.conf) if s.conf is not None else None} for d in docs for s in d.sentences]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return len(rows)


def attach_segments(docs: List[Judgment], path: str | Path) -> List[Judgment]:
    """Put roles from segments.parquet back onto judgments (by case_id + sent_idx)."""
    import pandas as pd
    seg = pd.read_parquet(path)
    lookup = {(r.case_id, int(r.sent_idx)): (r.role, r.conf) for r in seg.itertuples()}
    for d in docs:
        for s in d.sentences:
            if (d.case_id, s.idx) in lookup:
                s.role, s.conf = lookup[(d.case_id, s.idx)]
    return docs
