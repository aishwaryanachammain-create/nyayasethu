"""Text normalisation and legal-aware tokenisation (domain-specific tokenisation)."""
from __future__ import annotations

import re
import unicodedata
from typing import List

_QUOTES = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-",
           " ": " ", "​": ""}

_PAGE_ARTIFACTS = re.compile(
    r"(^\s*page\s+\d+\s+of\s+\d+\s*$)|(^\s*indian kanoon - http\S*\s*$)|(^\s*\d+\s*$)",
    re.IGNORECASE | re.MULTILINE)


def normalise(text: str) -> str:
    """Unicode NFKC, straight quotes, strip page furniture, collapse whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    for k, v in _QUOTES.items():
        text = text.replace(k, v)
    text = _PAGE_ARTIFACTS.sub(" ", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Legal stopwords: general function words + high-frequency judgment boilerplate that
# carries no issue signal ("appellant", "learned", "counsel" ...).
STOPWORDS = set("""
a an the and or but if of to in on at by for with from as is are was were be been being it its this that
these those there here which who whom whose what when where why how all any both each few more most other
some such no nor not only own same so than too very can will just should now also into upon under over
again further then once i me my we our you your he him his she her they them their has have had having do
does did doing would could shall may might must ought said thus hence therefore however whereas
learned counsel appellant appellants respondent respondents petitioner petitioners hon'ble honble court
para paras page dated sri shri smt mr ms dr j jj vs v versus case cases matter submitted submission
""".split())

_TOKEN = re.compile(
    r"""(?x)
    (?:section|sec|s|art|article|rule|order|o)\.?\s*\d+[a-z]*(?:\(\d+[a-z]*\))*   # statutory refs kept whole
    | \d{4}\s+\(\d+\)\s+scc\s+\d+                                                # 1963 (3) SCC 41 style
    | [a-z]+(?:'[a-z]+)?                                                         # words
    | \d+[a-z]*                                                                  # numbers / 12a
    """)


def tokenize(text: str, drop_stopwords: bool = True) -> List[str]:
    """Lower-cased tokens; statutory references like 'section 74' become one token 'section_74'."""
    out = []
    for m in _TOKEN.finditer(text.lower()):
        tok = m.group(0)
        if " " in tok or "." in tok:
            tok = re.sub(r"[\s.]+", "_", tok)
            tok = re.sub(r"^(sec|s)_", "section_", tok)
            tok = re.sub(r"^art_", "article_", tok)
        if drop_stopwords and tok in STOPWORDS:
            continue
        if len(tok) == 1 and not tok.isdigit():
            continue
        out.append(_light_stem(tok))
    return out


def _light_stem(tok: str) -> str:
    """Conservative suffix stripping (keeps legal terms readable; avoids Porter over-stemming)."""
    if "_" in tok or tok.isdigit() or len(tok) <= 4:
        return tok
    for suf, rep in (("ies", "y"), ("ments", "ment"), ("ings", "ing"), ("ed", ""), ("es", ""), ("s", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            if suf == "es" and not tok.endswith(("ses", "xes", "zes", "ches", "shes")):
                continue
            return tok[: -len(suf)] + rep
    return tok
