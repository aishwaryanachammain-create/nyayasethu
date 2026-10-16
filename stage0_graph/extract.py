"""Citation mentions in a sentence: reporter citations and "X v. Y" case names.

Reporter patterns come from Member A (stage0_corpus/citations.py: AIR, SCC both forms, SCR, SCC OnLine,
INSC, HC neutral, JT, SCALE, MANU, ILR, CriLJ). Added here, since A's regexes miss them:
  * SCR supplementary volumes   "[1959] SUPP. 1 S.C.R. 1", "1959 Supp (1) SCR 1"
  * SCR year-first form         "1964 (1) SCR 515"
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

from stage0_corpus.citations import _canon, extract_citations

_EXTRA = [
    # editorial form "[1997] 1 Suppl. SCR 87" (volume before "Suppl.")
    ("SCRSUPP", re.compile(r"[\(\[](?P<year>(?:19|20)\d{2})[\)\]]\s*(?P<vol>\d{1,2})\s*Suppl?\.?\s*S\.?\s?C\.?\s?R\.?\s*(?P<page>\d{1,5})\b", re.I)),
    ("SCRSUPP", re.compile(r"[\(\[](?P<year>(?:19|20)\d{2})[\)\]]\s*Suppl?\.?\s*\(?(?P<vol>\d{1,2})?\)?\s*S\.?\s?C\.?\s?R\.?\s*(?P<page>\d{1,5})\b", re.I)),
    ("SCRSUPP", re.compile(r"\b(?P<year>(?:19|20)\d{2})\s*Suppl?\.?\s*\(?(?P<vol>\d{1,2})?\)?\s*S\.?C\.?R\.?\s*(?P<page>\d{1,5})\b", re.I)),
    ("SCR",     re.compile(r"\b(?P<year>(?:19|20)\d{2})\s*\((?P<vol>\d{1,2})\)\s*S\.?\s?C\.?\s?R\.?\s*(?P<page>\d{1,5})\b")),
]

# A party name: capitalised words, allowing lower-case joiners ("State of Uttar Pradesh", "M/s. Goel & Co.")
_PARTY = r"(?:[A-Z][\w.&'/\-]*\.?)(?:\s+(?:of|and|&|the|for|[A-Z][\w.&'/\-]*\.?|\([A-Za-z][^()]{0,30}\))){0,9}"
_VS = r"\s+(?:v\.?|vs\.?|versus)\s+"
CASE_NAME = re.compile(rf"(?P<p1>{_PARTY}){_VS}(?P<p2>{_PARTY})")
# Words that start a sentence or clause but are not part of a party name
_LEAD = re.compile(r"^(?:(?:In|See|Also|And|But|The|This|That|Thus|Hence|Further|Following|Relying|Reliance|Whether|Where|When|If|Since|While|Although|Because|"
                   r"Placed|Referring|Similarly|Likewise|Accordingly|Again|Cf|Vide|Per|As|In re|Re|Of|On|"
                   r"Court|Bench|Judgment|Decision|Case|Learned|Counsel|Mr|Ms|Dr|Sri|Shri)\b\.?\s*)+")
_TRAIL = re.compile(r"(?:\s+(?:and|&|of|the|in|for))+$")


@dataclass(frozen=True)
class Mention:
    kind: str                 # reporter name (SCR, SCC, ...) or "NAME"
    key: str                  # canonical reporter key, or "" for case names
    span: tuple
    raw: str
    year: Optional[int] = None
    name: str = ""            # "petitioner v respondent" near the mention, if found


def _year(key: str) -> Optional[int]:
    try:
        return int(key.split("|")[1])
    except (IndexError, ValueError):
        return None


def reporter_mentions(text: str) -> List[Mention]:
    found = [Mention(c.reporter, c.key, c.span, c.raw, _year(c.key)) for c in extract_citations(text)]
    for rep, pat in _EXTRA:
        for m in pat.finditer(text):
            if not any(m.start() < f.span[1] and f.span[0] < m.end() for f in found):
                key = _canon(rep, m.groupdict())
                found.append(Mention(rep, key, m.span(), m.group(0), _year(key)))
    return sorted(found, key=lambda x: x.span)


def alias_keys(aliases: Iterable) -> List[str]:
    """Canonical keys for metadata/corpus citation strings, e.g. '[2019] 10 S.C.R. 133', '2018INSC696'."""
    out = []
    for a in aliases:
        if a is None or str(a).strip() in ("", "None", "nan", "<NA>"):
            continue
        for m in reporter_mentions(str(a)):
            out.append(m.key)
    return sorted(set(out))


_INTRO = re.compile(r"^.*\b(?:in|In|see|See|vide|Vide|following|Following)\s+(?=[A-Z])")


def clean_party(p: str, left: bool = False) -> str:
    if left:                                   # "This Court in Fateh Chand" -> "Fateh Chand"
        p = _INTRO.sub("", p)
    p = _LEAD.sub("", p.strip(" ,;:("))
    return _TRAIL.sub("", p).strip(" ,;:(")


def case_names(text: str) -> List[Mention]:
    out = []
    for m in CASE_NAME.finditer(text):
        p1, p2 = clean_party(m.group("p1"), left=True), clean_party(m.group("p2"))
        if len(p1) >= 3 and len(p2) >= 3:
            out.append(Mention("NAME", "", m.span(), m.group(0), None, f"{p1} v {p2}"))
    return out


def mentions(text: str) -> List[Mention]:
    """Reporter citations, each paired with the nearest preceding "X v. Y" (or following one, for
    "(1964) 1 SCR 515 (Fateh Chand v. Balkishan Das)"), plus bare case names not attached to a citation."""
    reps = reporter_mentions(text)
    names = case_names(text)
    used = set()
    out = []
    for r in reps:
        best = None
        for i, n in enumerate(names):
            gap_before = r.span[0] - n.span[1]
            gap_after = n.span[0] - r.span[1]
            if 0 <= gap_before <= 40 and not re.search(r"[.;]\s", text[n.span[1]:r.span[0]]):
                best = i                                   # keep the closest preceding name
            elif best is None and 0 <= gap_after <= 4:
                best = i
        name = ""
        if best is not None:
            name = names[best].name
            used.add(best)
        out.append(Mention(r.kind, r.key, r.span, r.raw, r.year, name))
    out += [n for i, n in enumerate(names) if i not in used]
    return sorted(out, key=lambda x: x.span)
