"""Legal-aware sentence splitter.

Standard splitters break on 'v.', 'Ltd.', 'S. 74', 'AIR 1963 SC 1405', 'para 12.3', 'i.e.' etc.
Strategy: protect abbreviation/citation periods with a placeholder, split on terminal
punctuation followed by whitespace + capital/quote/digit-paren, then restore.
Validate against a manually checked sample with `evaluate_splitter`.
"""
from __future__ import annotations

import re
from typing import List, Sequence, Tuple

ABBREVIATIONS = [
    "v", "vs", "viz", "i.e", "e.g", "etc", "cf", "ibid", "id", "supra", "ors", "anr", "ltd", "pvt", "co",
    "corpn", "corp", "inc", "no", "nos", "s", "ss", "sec", "secs", "art", "arts", "cl", "cls", "r", "o",
    "p", "pp", "para", "paras", "vol", "ed", "edn", "j", "jj", "cj", "hon'ble", "mr", "mrs", "ms", "dr",
    "sri", "shri", "smt", "st", "govt", "dept", "deptt", "admn", "u", "a", "b", "c", "d", "m", "n", "k",
    "ch", "sch", "ord", "reg", "regd", "rs", "approx", "fig", "w.e.f", "a.i.r", "s.c.r", "s.c.c", "u.p",
    "m.p", "a.p", "h.p", "w.b", "t.n", "j&k", "dist", "distt", "addl", "asst", "jt", "sr", "jr", "bros",
    "illus", "expln", "prov", "subs", "ins", "amend", "mad", "bom", "cal", "del", "all", "ker", "kar",
    "guj", "raj", "ori", "pat", "p&h", "lj", "crlj", "cri", "civ", "crl", "misc", "wp", "slp", "c.a",
    "crl.a", "w.p", "s.l.p", "o.s", "a.s", "s.a",
]
_ABBR_RE = re.compile(
    r"(?<![\w.])(" + "|".join(re.escape(a) for a in sorted(ABBREVIATIONS, key=len, reverse=True)) + r")\.",
    re.IGNORECASE)
_DECIMAL = re.compile(r"(\d)\.(\d)")
_INITIALS = re.compile(r"\b([A-Z])\.(?=\s?[A-Z][\.a-z])")
_ENUM = re.compile(r"(^|\n|\s)(\(?[ivxlcdm]+\)|\(?\d{1,3}\)|\(?[a-z]\))\s", re.IGNORECASE)
_PH = "⁣"  # invisible separator used as protected-period placeholder

_SPLIT = re.compile(r"(?<=[.?!;:])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])|\n\s*\n")


def split_sentences(text: str, min_chars: int = 3) -> List[str]:
    if not text:
        return []
    t = text.replace("\r", "")
    t = _DECIMAL.sub(lambda m: f"{m.group(1)}{_PH}{m.group(2)}", t)
    t = _ABBR_RE.sub(lambda m: m.group(1) + _PH, t)
    t = _INITIALS.sub(lambda m: m.group(1) + _PH, t)
    # numbered paragraph markers ("12. The question ...") stay attached to their sentence
    t = re.sub(r"(^|\n)\s*(\d{1,3})\.\s+", lambda m: f"{m.group(1)}{m.group(2)}{_PH} ", t)
    # inline paragraph numbers after a sentence end: "... Act. 12. The next issue" -> new sentence
    t = re.sub(r"(?<=[.?!])\s+(\d{1,3})\.\s+(?=[A-Z])", lambda m: f"\n\n{m.group(1)}{_PH} ", t)
    # semicolons/colons only split when followed by a long clause — keep list items intact
    parts = []
    for chunk in _SPLIT.split(t):
        if chunk is None:
            continue
        chunk = chunk.replace(_PH, ".").strip()
        chunk = re.sub(r"\s+", " ", chunk)
        if len(chunk) >= min_chars:
            parts.append(chunk)
    return _merge_fragments(parts)


def _merge_fragments(parts: List[str]) -> List[str]:
    """Merge splits that produced a dangling citation fragment or very short clause."""
    out: List[str] = []
    for p in parts:
        if out and (len(p) < 12 or re.match(r"^(\(\d{4}\)|\d{4}\b|SCC|AIR|SCR)", p)):
            out[-1] = out[-1] + " " + p
        elif out and out[-1].endswith((":", ";")) and len(out[-1]) < 40:
            out[-1] = out[-1] + " " + p
        else:
            out.append(p)
    return out


def evaluate_splitter(gold: Sequence[Sequence[str]], texts: Sequence[str]) -> Tuple[float, float, float]:
    """Boundary P/R/F1 against a manually split sample (Week-2 200-sentence validation)."""
    tp = fp = fn = 0
    for g, t in zip(gold, texts):
        pred = split_sentences(t)
        gb, pb = _boundaries(g), _boundaries(pred)
        tp += len(gb & pb)
        fp += len(pb - gb)
        fn += len(gb - pb)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _boundaries(sents: Sequence[str]) -> set:
    pos, out = 0, set()
    for s in sents:
        pos += len(re.sub(r"\s+", "", s))
        out.add(pos)
    return out
