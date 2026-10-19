"""Parse the SCR editors' "Case Law Reference" block at the head of AWS Supreme Court judgments.

    Case Law Reference (2017) 9 SCC 499 referred to Para 20 [2011] 11 SCR 870 relied on Para 37 CIVIL APPELLATE ...

Each entry is `<citation> <treatment phrase> Para <n>[, <n>...]`. The block ends at the jurisdiction line
("CIVIL APPELLATE JURISDICTION") or "From the Judgment". Page furniture is removed with the graph cleaner.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

from stage0_graph.build import clean
from stage0_graph.extract import reporter_mentions

from .labels import editorial_label

_HEAD = "Case Law Reference"
_END = re.compile(r"(?:CIVIL|CRIMINAL|ORIGINAL|APPELLATE|ADVISORY|WRIT|SPECIAL|EXTRAORDINARY)[A-Z ]*JURISDICTION|"
                  r"From the (?:Judgment|Order)|Judgment of the Court", re.I)
_ENTRY = re.compile(r"(.+?)\s+Paras?\.?\s*(\d[\d.]*(?:\s*(?:,|and|&)\s*\d[\d.]*)*)", re.S)
_PHRASE = re.compile(r"((?:[a-z]+\s?){1,5})$")


@dataclass
class Entry:
    cite_text: str
    phrase: str
    label: Optional[str]
    paras: List[int] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)


def block_text(sentences, limit: int = 300) -> Optional[str]:
    txt = " ".join(s.text for s in sentences[:limit])
    i = txt.find(_HEAD)
    if i < 0:
        return None
    blk = txt[i + len(_HEAD):]
    m = _END.search(blk)
    return clean(blk[:m.start()] if m else blk[:5000])


def parse_block(blk: str) -> List[Entry]:
    out = []
    for m in _ENTRY.finditer(blk or ""):
        body = m.group(1).strip()
        ph = _PHRASE.search(body)
        if not ph:
            continue
        phrase = ph.group(1).strip()
        cite = body[:ph.start()].strip(" ,;")
        keys = sorted({x.key for x in reporter_mentions(cite) if x.key})
        if not keys:
            continue
        paras = sorted({int(float(p)) for p in re.findall(r"\d+(?:\.\d+)?", m.group(2)) if float(p) < 10000})
        out.append(Entry(cite, phrase, editorial_label(phrase), paras, keys))
    return out


def entries(doc) -> List[Entry]:
    blk = block_text(doc.sentences)
    return parse_block(blk) if blk else []
