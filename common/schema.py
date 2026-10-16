"""Shared record format (Member A -> Member B contract).

judgments.jsonl, one object per line:
    {case_id, source, court, date, bench_size, title, aliases, text, sentences:[{idx, text}]}
segments.parquet, one row per sentence:
    {case_id, sent_idx, text, role, conf}
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

COURT_RANK = {"SC": 3, "HC": 2, "DC": 1}


@dataclass
class Sentence:
    idx: int
    text: str
    role: Optional[str] = None
    conf: Optional[float] = None
    citations: List[str] = field(default_factory=list)   # case_ids cited in this sentence (filled by Stage 0)


@dataclass
class Judgment:
    case_id: str
    source: str = ""
    court: Optional[str] = None          # SC / HC / DC
    date: Optional[str] = None           # ISO yyyy-mm-dd, or yyyy when only the year is known
    bench_size: Optional[int] = None
    title: str = ""
    aliases: List[str] = field(default_factory=list)    # reporter citations that refer TO this case
    text: str = ""
    sentences: List[Sentence] = field(default_factory=list)

    # ---- convenience used by graph / retrieval code
    @property
    def doc_id(self) -> str:
        return self.case_id

    @property
    def year(self) -> Optional[int]:
        try:
            return int(str(self.date)[:4]) if self.date else None
        except ValueError:
            return None

    @property
    def has_roles(self) -> bool:
        return any(s.role for s in self.sentences)

    def role_text(self, roles) -> str:
        roles = {roles} if isinstance(roles, str) else set(roles)
        return " ".join(s.text for s in self.sentences if s.role in roles)

    def to_dict(self, with_roles: bool = False) -> Dict:
        d = asdict(self)
        d["sentences"] = [{"idx": s.idx, "text": s.text, **({"role": s.role, "conf": s.conf} if with_roles else {})}
                          for s in self.sentences]
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "Judgment":
        sents = [Sentence(idx=s.get("idx", i), text=s["text"], role=s.get("role"), conf=s.get("conf"))
                 for i, s in enumerate(d.get("sentences") or [])]
        return cls(case_id=str(d["case_id"]), source=d.get("source", ""), court=d.get("court"), date=d.get("date"),
                   bench_size=d.get("bench_size"), title=d.get("title") or "", aliases=list(d.get("aliases") or []),
                   text=d.get("text") or "", sentences=sents)
