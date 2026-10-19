"""Resolve citation mentions to case_ids.

Order of evidence (the method is kept on every edge):
  key        reporter key matches a case's own SCR / INSC citation (metadata or corpus aliases)
  name_year  reporter not in metadata (SCC, AIR, ...): party names fuzzy-matched (rapidfuzz) against
             case titles decided in the citation year or the year before
  name_only  bare "X v. Y" with no reporter: fuzzy match against all earlier cases, stricter threshold
A match must be unambiguous (best beats runner-up by `margin`) and not later than the citing judgment.
"""
from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from stage0_corpus.citations import normalise_case_name

from .extract import Mention
from .nodes import Node

_ABBR = {r"\bu\.?\s?p\b": "uttar pradesh", r"\bm\.?\s?p\b": "madhya pradesh", r"\ba\.?\s?p\b": "andhra pradesh",
         r"\bh\.?\s?p\b": "himachal pradesh", r"\bw\.?\s?b\b": "west bengal", r"\bt\.?\s?n\b": "tamil nadu",
         r"\bj\s?&\s?k\b": "jammu and kashmir", r"\bnct\b": "delhi", r"\bgovt\b": "government",
         r"\buoi\b": "union of india", r"\bcit\b": "commissioner of income tax", r"\bmp\b": "madhya pradesh",
         r"\bup\b": "uttar pradesh"}
GENERIC = set("""state union india of the and government commissioner income tax ltd limited pvt private co company
corporation corpn bank municipal through secretary others anr ors another m s mr mrs smt shri sri dr director
general chief officer district collector superintendent police board authority council registrar""".split())


def norm_party(p: str) -> str:
    p = p.lower().replace("m/s", " ")
    for pat, rep in _ABBR.items():
        p = re.sub(pat, rep, p)
    p = re.sub(r"\(.*?\)", " ", p)                                   # "(dead) through lrs"
    p = re.sub(r"\b(through|thr|by)\b.*$", " ", p)                   # "Union of India through Secretary ..."
    p = re.sub(r"(?<=[a-z])\d+\b", " ", p)                            # footnote markers: "Union of India8"
    return normalise_case_name(p)


def _rare_tokens(s: str) -> List[str]:
    return [t for t in s.split() if t not in GENERIC and len(t) > 2 and not t.isdigit()]


class Resolver:
    def __init__(self, nodes: Dict[str, Node], name_threshold: float = 85, name_only_threshold: float = 92,
                 margin: float = 5):
        self.nodes = nodes
        self.t_year, self.t_only, self.margin = name_threshold, name_only_threshold, margin
        keys: Dict[str, set] = defaultdict(set)
        for n in nodes.values():
            for k in n.keys:
                keys[k].add(n.case_id)
        self.by_key = {k: next(iter(v)) for k, v in keys.items() if len(v) == 1}
        self.ambiguous_keys = {k for k, v in keys.items() if len(v) > 1}
        self.ranges: Dict[tuple, List[tuple]] = defaultdict(list)       # (reporter, year, vol) -> [(first, last, id)]
        for n in nodes.values():
            if n.scr_range:
                rep, y, v, a, b = n.scr_range
                self.ranges[(rep, y, v)].append((a, b, n.case_id))
        self.parties: Dict[str, Tuple[str, str]] = {}
        self.by_token: Dict[str, set] = defaultdict(set)
        for n in nodes.values():
            pet, resp = norm_party(n.petitioner), norm_party(n.respondent)
            if not pet:
                continue
            self.parties[n.case_id] = (pet, resp)
            for t in set(_rare_tokens(pet) + _rare_tokens(resp)):
                self.by_token[t].add(n.case_id)

    # ------------------------------------------------------------------ names
    def _score(self, qp: str, qr: str, cid: str) -> float:
        cp, cr = self.parties[cid]
        def sim(x, y):
            # token_set alone scores 100 when one name is a subset of the other ("munna" in "... alias munna")
            return 0.5 * fuzz.token_set_ratio(x, y) + 0.5 * fuzz.token_sort_ratio(x, y)

        def pair(a, b, c, d):
            s1 = sim(a, c)
            s2 = sim(b, d) if b and d else 50.0
            return 0.6 * s1 + 0.4 * s2 if min(s1, s2) >= 60 else 0.0
        return max(pair(qp, qr, cp, cr), pair(qp, qr, cr, cp))        # appeals often swap party order

    @lru_cache(maxsize=200_000)
    def _match_name(self, name: str, lo: Optional[int], hi: Optional[int], threshold: float) -> Tuple[Optional[str], float]:
        parts = name.split(" v ", 1)
        if len(parts) != 2:
            return None, 0.0
        qp, qr = norm_party(parts[0]), norm_party(parts[1])
        toks = _rare_tokens(qp) + _rare_tokens(qr)
        if not toks:                                  # "State of Bihar v. Union of India": too generic to resolve
            return None, 0.0
        cands = set().union(*(self.by_token.get(t, set()) for t in toks))
        scored = []
        for cid in cands:
            y = self.nodes[cid].year
            if (lo is not None and (y is None or y < lo)) or (hi is not None and y is not None and y > hi):
                continue
            s = self._score(qp, qr, cid)
            if s >= threshold:
                scored.append((s, cid))
        if not scored:
            return None, 0.0
        scored.sort(reverse=True)
        if len(scored) > 1 and scored[0][0] - scored[1][0] < self.margin:
            return None, 0.0                          # ambiguous
        return scored[0][1], scored[0][0]

    def _in_range(self, m: Mention) -> Optional[str]:
        """SCR page inside one reported case's page range (pinpoint or slightly wrong start page)."""
        if m.kind not in ("SCR", "SCRSUPP"):
            return None
        _, y, v, page = m.key.split("|")
        if not page.isdigit():
            return None
        hits = [cid for a, b, cid in self.ranges.get((m.kind, y, v), []) if a <= int(page) <= b]
        return hits[0] if len(hits) == 1 else None

    # ------------------------------------------------------------------ public
    def resolve(self, m: Mention, citing_year: Optional[int]) -> Tuple[Optional[str], str, float]:
        """-> (case_id or None, method, score). Method is 'unresolved:<why>' when None."""
        if m.key:
            cid = self.by_key.get(m.key)
            if cid:
                y = self.nodes[cid].year
                if citing_year and y and y > citing_year:
                    return None, "unresolved:cited_after_citing", 0.0
                return cid, "key", 100.0
            if m.key in self.ambiguous_keys:
                return None, "unresolved:ambiguous_key", 0.0
            cid = self._in_range(m)
            if cid:
                y = self.nodes[cid].year
                if not (citing_year and y and y > citing_year):
                    return cid, "key_range", 90.0
            if m.name and m.year:
                hi = min(m.year, citing_year) if citing_year else m.year
                cid, s = self._match_name(m.name, m.year - 1, hi, self.t_year)
                if cid:
                    return cid, "name_year", s
                return None, "unresolved:name_no_match", 0.0
            return None, "unresolved:no_name", 0.0
        cid, s = self._match_name(m.name, None, citing_year, self.t_only)
        if cid:
            return cid, "name_only", s
        return None, "unresolved:name_no_match", 0.0
