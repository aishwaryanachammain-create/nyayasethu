"""Citation extraction over standard Indian reporter formats, and resolution to doc_ids."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

# Each pattern yields named groups normalised into a canonical key.
_PATTERNS = [
    ("AIR",    re.compile(r"\bA\.?I\.?R\.?\s*(?P<year>(?:18|19|20)\d{2})\s*(?P<court>S\.?C\.?|[A-Z][A-Za-z&]{1,8})\.?\s*(?P<page>\d{1,5})\b")),
    ("SCC",    re.compile(r"[\(\[](?P<year>(?:18|19|20)\d{2})[\)\]]\s*(?P<vol>\d{1,2})\s*S\.?\s?C\.?\s?C\.?\s*(?P<page>\d{1,5})\b")),
    ("SCC",    re.compile(r"\b(?P<year>(?:18|19|20)\d{2})\s*\((?P<vol>\d{1,2})\)\s*S\.?C\.?C\.?\s*(?P<page>\d{1,5})\b")),
    ("SCR",    re.compile(r"[\(\[](?P<year>(?:18|19|20)\d{2})[\)\]]\s*(?P<vol>\d{1,2})?\s*S\.?\s?C\.?\s?R\.?\s*(?P<page>\d{1,5})\b")),
    ("SCCOL",  re.compile(r"\b(?P<year>(?:19|20)\d{2})\s*SCC\s*OnLine\s*(?P<court>[A-Z][A-Za-z]{1,6})\s*(?P<page>\d{1,6})\b", re.I)),
    ("INSC",   re.compile(r"\b(?P<year>(?:19|20)\d{2})\s*INSC\s*(?P<page>\d{1,5})\b")),
    # High Court neutral citations: 2023:DHC:1234, 2024:KER:56789, 2023:BHC-AS:1234, 2023:AHC:12345-DB
    ("NEUTRAL", re.compile(r"\b(?P<year>20\d{2}):(?P<court>[A-Z]{2,6}(?:-[A-Z]{1,4})?):(?P<page>\d{1,6})(?:-DB)?\b")),
    ("JT",     re.compile(r"\bJT\s*(?P<year>(?:19|20)\d{2})\s*\((?P<vol>\d{1,2})\)\s*SC\s*(?P<page>\d{1,5})\b")),
    ("SCALE",  re.compile(r"\((?P<year>(?:19|20)\d{2})\)\s*(?P<vol>\d{1,2})\s*SCALE\s*(?P<page>\d{1,5})\b", re.I)),
    ("MANU",   re.compile(r"\bMANU/(?P<court>[A-Z]{2,4})/(?P<page>\d{3,5})/(?P<year>(?:19|20)\d{2})\b")),
    ("ILR",    re.compile(r"\bI\.?L\.?R\.?\s*(?P<year>(?:18|19|20)\d{2})\s*(?P<court>[A-Z][a-z]{2,4})\.?\s*(?P<page>\d{1,5})\b")),
    ("CRILJ",  re.compile(r"\b(?P<year>(?:19|20)\d{2})\s*Cri\.?\s*L\.?\s*J\.?\s*(?P<page>\d{1,5})\b", re.I)),
]

# "Fateh Chand v. Balkishan Das" / "State of U.P. vs Raj Narain" — party names around v./vs.
CASE_NAME = re.compile(
    r"(?P<p1>(?:[A-Z][\w.&'\-]*\s?){1,6}?(?:\s(?:of|and|&)\s(?:[A-Z][\w.&'\-]*\s?){1,4})?)"
    r"\s+(?:v\.|vs\.?|versus)\s+"
    r"(?P<p2>(?:[A-Z][\w.&'\-]*(?:\s|,|$)){1,6}(?:(?:of|and|&)\s(?:[A-Z][\w.&'\-]*\s?){1,4})?)")


# SCR running page headers ("SUPREME COURT REPORTS [2020] 1 S.C.R. 36", or "36 SUPREME COURT REPORTS [2020] 1 S.C.R.")
# name the page of the citing judgment itself; read as citations they point at whichever case starts on that page.
SCR_HEADER = re.compile(r"SUPREME\s+COURT\s+REPORTS\s*\[\d{4}\]\s*\d{1,2}\s*S\.?\s?C\.?\s?R\.?\s*\d{1,5}|"
                        r"\b\d{1,5}\s*SUPREME\s+COURT\s+REPORTS\s*\[\d{4}\]\s*\d{1,2}\s*S\.?\s?C\.?\s?R\.?", re.I)


@dataclass(frozen=True)
class Citation:
    reporter: str
    key: str           # canonical key, e.g. "SCC|2019|5|123"
    span: tuple        # (start, end) char offsets in the source text
    raw: str


def _canon(reporter: str, gd: Dict[str, Optional[str]]) -> str:
    court = (gd.get("court") or "").replace(".", "").upper()
    vol = gd.get("vol") or ""
    return "|".join([reporter, gd.get("year") or "", court or vol, gd.get("page") or ""])


def extract_citations(text: str) -> List[Citation]:
    found: Dict[tuple, Citation] = {}
    for rep, pat in _PATTERNS:
        for m in pat.finditer(text):
            c = Citation(rep, _canon(rep, m.groupdict()), m.span(), m.group(0))
            # avoid duplicates where SCC and SCR regexes overlap the same span
            if not any(_overlap(c.span, k) for k in found):
                found[c.span] = c
    return sorted(found.values(), key=lambda c: c.span)


def canonical_key(raw: str) -> Optional[str]:
    cs = extract_citations(raw)
    return cs[0].key if cs else None


def _overlap(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def normalise_case_name(name: str) -> str:
    n = name.lower()
    n = re.sub(r"\b(the|m/s|ors?|anr|others|another|and|&|etc)\b\.?", " ", n)
    n = re.sub(r"\b(versus|vs?)\b\.?", " v ", n)
    n = re.sub(r"[^a-z0-9 ]+", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def extract_case_names(text: str) -> List[str]:
    out = []
    for m in CASE_NAME.finditer(text):
        p1 = m.group("p1").strip().split()[-4:]   # last few tokens of the left party
        p2 = m.group("p2").strip().rstrip(",").split()[:4]
        out.append(normalise_case_name(" ".join(p1) + " v " + " ".join(p2)))
    return out


class CitationResolver:
    """Maps reporter keys and normalised case names to corpus doc_ids."""

    # The short "surname v first-word" key is skipped when the other party starts with one of these words: it would
    # match every "<X> Singh v. State of <Y>" citation.
    GENERIC_PARTY = {"state", "union", "govt", "government", "commissioner", "the", "central", "chief", "director",
                     "secretary", "collector", "municipal", "national", "district", "registrar"}

    def __init__(self):
        self.by_key: Dict[str, str] = {}
        self.by_name: Dict[str, str] = {}
        self._short: Dict[str, str] = {}          # short key -> doc_id; None once two judgments share it

    def register(self, doc_id: str, aliases: Iterable[str] = (), title: str = "") -> None:
        self._index_size = None                   # invalidate the first-token index used by resolve_sentence
        for a in aliases:
            k = canonical_key(a) or a.strip()
            self.by_key[k] = doc_id
        if title:
            n = normalise_case_name(title)
            if n:
                self.by_name[n] = doc_id
                # also register first-party-surname heuristic "fateh chand v balkishan", unless it is generic
                # (other party "State", "Union", ...) or ambiguous (two judgments give the same short key)
                parts = n.split(" v ")
                if len(parts) == 2 and parts[0].split() and parts[1].split() \
                        and parts[1].split()[0] not in self.GENERIC_PARTY:
                    short = parts[0].split()[-1] + " v " + parts[1].split()[0]
                    if short != n:
                        if short in self._short and self._short[short] != doc_id:
                            prev, self._short[short] = self._short[short], None
                            if prev is not None and self.by_name.get(short) == prev:
                                del self.by_name[short]         # only the short-key entry, never a full title
                        elif short not in self._short:
                            self._short[short] = doc_id
                            self.by_name.setdefault(short, doc_id)

    def resolve_sentence(self, sentence: str) -> List[str]:
        ids: List[str] = []
        for c in extract_citations(SCR_HEADER.sub(" ", sentence)):
            d = self.by_key.get(c.key)
            if d and d not in ids:
                ids.append(d)
        # Same result as testing `f" {name} " in f" {low} "` for every registered name (both sides are normalised
        # to single-spaced tokens), but only names whose first token occurs in the sentence are compared.
        toks = normalise_case_name(sentence).split()
        hits = set()
        for i, t in enumerate(toks):
            for name, ntoks in self._by_first().get(t, ()):
                if len(name) > 6 and toks[i:i + len(ntoks)] == ntoks:
                    hits.add(name)
        for name in sorted(hits, key=self._name_order.__getitem__):     # registration order, as before
            d = self.by_name[name]
            if d not in ids:
                ids.append(d)
        return ids

    def _by_first(self) -> Dict[str, list]:
        if getattr(self, "_index_size", None) is None:
            self._first, self._name_order = {}, {}
            for k, name in enumerate(self.by_name):
                self._name_order[name] = k
                ntoks = name.split()
                if ntoks:
                    self._first.setdefault(ntoks[0], []).append((name, ntoks))
            self._index_size = len(self.by_name)
        return self._first
