"""Step 8 — header metadata by regex: court, date, bench size. Missing fields stay None and
coverage is reported (never guessed)."""
from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Optional

from common.schema import Judgment

MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}
_MON = r"(january|february|march|april|may|june|july|august|september|october|november|december)"

_DATE_PATTERNS = [
    # Dated: 12.03.2019 / Decided on 12-03-2019 / DATE OF JUDGMENT: 12/03/2019
    re.compile(r"(?:dated|decided on|date of (?:the )?judgment|pronounced on|delivered on)\s*[:\-]?\s*"
               r"(\d{1,2})[./\-](\d{1,2})[./\-]((?:19|20)\d{2})", re.I),
    # DATE OF JUDGMENT: 12 March 2019 / 12th March, 2019
    re.compile(r"(?:dated|decided on|date of (?:the )?judgment|pronounced on|delivered on)\s*[:\-]?\s*"
               r"(\d{1,2})(?:st|nd|rd|th)?\s+" + _MON + r",?\s+((?:19|20)\d{2})", re.I),
    # March 12, 2019 at the end of an SC judgment ("NEW DELHI; MARCH 12, 2019")
    re.compile(r"new delhi[;,.]?\s*" + _MON + r"\s+(\d{1,2}),?\s+((?:19|20)\d{2})", re.I),
]

_BENCH_HEADER = re.compile(r"(?:CORAM|BENCH|Coram|Bench)\s*[:\-]\s*(.+?)(?:\n\s*\n|JUDGMENT|ORDER|CITATION|ACT:|$)", re.S)
_HONBLE = re.compile(r"HON'?BLE\s+(?:MR\.?|MRS\.?|MS\.?|DR\.?)?\s*(?:THE\s+)?(?:CHIEF\s+)?JUSTICE[ \t]+([A-Z][A-Z. \t]{2,40})", re.I)
_SIGNED = re.compile(r"\.{3,}\s*,?\s*J\.\s*\n?\s*\[?\(?\s*([A-Z][A-Za-z.\s]{2,40}?)\s*\)?\]?\s*(?:\n|$)")
_BENCH_WORDS = {"constitution bench": 5, "full bench": 3, "division bench": 2, "single judge": 1}
_NUM_WORDS = {"two": 2, "three": 3, "five": 5, "seven": 7, "nine": 9, "eleven": 11, "thirteen": 13}


def extract_court(text: str) -> Optional[str]:
    head = text[:2000].lower()
    if "supreme court of india" in head:
        return "SC"
    if re.search(r"high court", head):
        return "HC"
    if re.search(r"district (court|judge)|sessions (court|judge)|civil judge", head):
        return "DC"
    return None


def extract_date(text: str) -> Optional[str]:
    zones = [text[:3000], text[-2500:]]
    for z in zones:
        for i, pat in enumerate(_DATE_PATTERNS):
            m = pat.search(z)
            if not m:
                continue
            g = m.groups()
            try:
                if i == 0:
                    d, mo, y = int(g[0]), int(g[1]), int(g[2])
                elif i == 1:
                    d, mo, y = int(g[0]), MONTHS[g[1].lower()], int(g[2])
                else:
                    d, mo, y = int(g[1]), MONTHS[g[0].lower()], int(g[2])
                if 1 <= d <= 31 and 1 <= mo <= 12:
                    return f"{y:04d}-{mo:02d}-{d:02d}"
            except (ValueError, KeyError):
                continue
    return None


def extract_bench_size(text: str) -> Optional[int]:
    """Judge names after CORAM:/BENCH:, else HON'BLE ... JUSTICE lines, else signature blocks."""
    head = text[:4000]
    m = _BENCH_HEADER.search(head)
    if m:
        names = [n for n in re.split(r",|\band\b|\n|&", m.group(1)) if re.search(r"[A-Za-z]{3,}", n)]
        if 0 < len(names) <= 15:
            return len(names)
    hon = {re.sub(r"\s+", " ", x.strip(" .")).upper() for x in _HONBLE.findall(head)}
    if hon:
        return len(hon)
    m = re.search(r"\b(two|three|five|seven|nine|eleven|thirteen)[- ]judge bench\b", head.lower())
    if m:
        return _NUM_WORDS[m.group(1)]
    for w, n in _BENCH_WORDS.items():
        if w in head.lower():
            return n
    sig = {re.sub(r"\s+", " ", x.strip(" .")).upper() for x in _SIGNED.findall(text[-4000:])}
    return len(sig) or None


def fill_metadata(doc: Judgment) -> Dict[str, str]:
    """Fill missing fields from the text. Returns {field: 'metadata'|'regex'|'missing'}."""
    how = {}
    for field, fn in (("court", extract_court), ("date", extract_date), ("bench_size", extract_bench_size)):
        if getattr(doc, field) not in (None, ""):
            how[field] = "metadata"
            continue
        v = fn(doc.text)
        setattr(doc, field, v)
        how[field] = "regex" if v not in (None, "") else "missing"
    return how


def coverage(reports: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    out = {}
    n = max(len(reports), 1)
    for field in ("court", "date", "bench_size"):
        c = Counter(r[field] for r in reports)
        out[field] = {k: round(c.get(k, 0) / n, 4) for k in ("metadata", "regex", "missing")}
        out[field]["found"] = round(1 - out[field]["missing"], 4)
    return out
