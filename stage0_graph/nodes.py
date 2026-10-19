"""Graph nodes: every Supreme Court case in the AWS metadata (1950-2025), plus every corpus judgment.

Corpus judgments are the citing side (we have their text). Metadata-only cases can only be cited.
IDs follow Member A's rule (stage0_corpus/loaders.py): `sc:<cnr>`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from stage0_corpus.loaders import bench_from_judges, iso_date

from .extract import alias_keys


@dataclass
class Node:
    case_id: str
    title: str = ""
    petitioner: str = ""
    respondent: str = ""
    date: Optional[str] = None
    bench_size: Optional[int] = None
    in_corpus: bool = False
    keys: List[str] = field(default_factory=list)      # canonical reporter keys that refer to this case
    scr_range: Optional[tuple] = None                   # (reporter, year, vol, first page, last page) from metadata path

    @property
    def year(self) -> Optional[int]:
        try:
            return int(str(self.date)[:4]) if self.date else None
        except ValueError:
            return None


def split_title(title: str):
    parts = re.split(r"\s+(?:versus|vs\.?|v\.)\s+", title or "", maxsplit=1, flags=re.I)
    return (parts[0], parts[1]) if len(parts) == 2 else (title or "", "")


_PATH = re.compile(r"^(S_)?(\d{4})_(\d+)_(\d+)_(\d+)$")


def scr_range(p: str) -> Optional[tuple]:
    """Metadata `path` encodes the SCR location: 2019_10_133_142 or S_1959_1_28_38 (supplementary)."""
    m = _PATH.match(str(p or ""))
    if not m:
        return None
    return ("SCRSUPP" if m.group(1) else "SCR", m.group(2), m.group(3), int(m.group(4)), int(m.group(5)))


def metadata_nodes(meta_dir: str | Path) -> Dict[str, Node]:
    """One node per CNR (the parquet files repeat some rows)."""
    import pandas as pd
    files = sorted(Path(meta_dir).rglob("*.parquet"))
    if not files:
        return {}
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).drop_duplicates("cnr")
    out: Dict[str, Node] = {}
    for r in df.itertuples(index=False):
        cid = f"sc:{r.cnr}"
        pet, resp = (r.petitioner or "", r.respondent or "")
        if not pet:
            pet, resp = split_title(r.title)
        judges = bench_from_judges(r.judge)
        out[cid] = Node(cid, title=str(r.title or ""), petitioner=str(pet), respondent=str(resp),
                        date=iso_date(r.decision_date),
                        bench_size=judges if judges and judges > 1 else None,   # single name = author, not bench
                        keys=alias_keys([r.citation, r.nc_display]), scr_range=scr_range(r.path))
    return out


def merge_corpus(nodes: Dict[str, Node], docs: Iterable) -> Dict[str, Node]:
    """Corpus judgments override metadata (their bench comes from the SCR header)."""
    for d in docs:
        n = nodes.get(d.case_id)
        pet, resp = split_title(d.title)
        if n is None:
            n = nodes[d.case_id] = Node(d.case_id, title=d.title, petitioner=pet, respondent=resp)
        n.in_corpus = True
        n.date = d.date or n.date
        n.bench_size = d.bench_size or n.bench_size
        n.keys = sorted(set(n.keys) | set(alias_keys(d.aliases)))
    return nodes
