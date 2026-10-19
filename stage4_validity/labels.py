"""Treatment labels for one citing mention, and the two weak labellers.

* `editorial_label`: maps the SCR editors' "Case Law Reference" phrase ("relied on", "held inapplicable",
  "overruled", ...) to a class. This is the primary weak label (decision Q4, docs/INSPECTION_REPORT.md).
* `cue_label`: the keyword rules from the project brief, applied to the citing sentence. This is the
  rule baseline and a secondary weak label for documents with no editorial block.

Classes: followed, distinguished, overruled, plus `neutral` for a plain reference ("referred to"). The
brief asks for three classes; neutral is kept as a fourth so that plain references are not forced
into "followed" (see docs/MEMBER_B_REPORT.md, deviations). Ambiguous phrases map to None and are dropped.
"""
from __future__ import annotations

import re
from typing import Optional

TREATMENTS = ["followed", "distinguished", "overruled"]
LABELS = TREATMENTS + ["neutral"]

# Editorial phrases. The PDF text layer sometimes drops a first letter ("eferred to"), so match stems.
# Order matters: negative phrases ("not approved") are checked before the positive stems inside them.
_ED = [
    ("overruled", re.compile(r"overrul|not (?:good|correct) law|not correct|incorrect law|disapprov|"
                             r"not approved|per incur")),
    ("distinguished", re.compile(r"distinguish|inapplicable|not applicable|inappropriate")),
    ("followed", re.compile(r"relied|follow|approv|affirm|held applicable|held correct law|applied")),
    ("neutral", re.compile(r"ferred|cited|discussed|explained|clarified")),
]


def editorial_label(phrase: str) -> Optional[str]:
    p = (phrase or "").lower()
    if "reconsider" in p:                      # "needs reconsideration": doubted, not a clear class
        return None
    for lab, pat in _ED:
        if pat.search(p):
            return lab
    return None


# Cue rules from the brief, on the citing sentence (and the next one, where the treatment is often stated).
_CUES = {
    "followed": re.compile(r"\bfollow(?:ed|ing)?\b|\brel(?:ied|y|ying) (?:up)?on\b|\bappl(?:ied|y|ying)\b|"
                           r"\bapproved\b|\breiterated\b|\bsquarely (?:covers|applies|governs)\b", re.I),
    "distinguished": re.compile(r"\bdistinguishable\b|\bdistinguish(?:ed|ing)?\b|facts (?:\w+ ){0,6}(?:are|were) "
                                r"(?:\w+ )?(?:different|distinguishable)|\bnot applicable\b|\binapplicable\b|"
                                r"\bhas no application\b|\bno application\b|\bnot attracted\b", re.I),
    "overruled": re.compile(r"\boverrul(?:ed|es|ing)\b|\bno longer (?:a )?good law\b|\bnot (?:a )?good law\b|"
                            r"\bwrongly decided\b|\bdoes not lay down (?:the )?correct law\b|\bper incuriam\b", re.I),
}
_NEG = re.compile(r"\b(?:not|no|never|cannot|can ?not|neither|nor)\s+(?:\w+\s+){0,2}$", re.I)
# "the following words", "to the following effect": not a treatment
_NOT_TREATMENT = re.compile(r"\b(?:the|as)\s+following\b|\bfollowing\s+(?:words|effect|terms|passage|paragraphs?|"
                            r"observations?|discussion|manner|questions?|reasons?|extract|lines|view:)", re.I)
# a party's reliance ("learned counsel relied upon X") is an argument, not the court's treatment
_PARTY_RELIANCE = re.compile(r"\b(?:counsel|learned|appellants?|respondents?|petitioners?|Mr|Ms|Dr|Sr|Advocate|"
                             r"Solicitor|Attorney|parties|amicus)\b[^.;]{0,80}?\brel(?:ied|ying|y)\b", re.I)


def cue_label(text: str) -> Optional[str]:
    """One class if exactly one class's cues fire (un-negated), else None (no cue, or ambiguous)."""
    fired = set()
    text = _NOT_TREATMENT.sub(" ", _PARTY_RELIANCE.sub(" ", text or ""))
    for lab, pat in _CUES.items():
        for m in pat.finditer(text):
            if _NEG.search(text[max(0, m.start() - 30):m.start()]):
                if lab == "followed":             # "not followed" / "cannot be applied": not a clean class
                    return None
                continue
            fired.add(lab)
    if "overruled" in fired:                      # the strongest cue wins over "followed" in the same sentence
        fired.discard("followed")
    return fired.pop() if len(fired) == 1 else None
