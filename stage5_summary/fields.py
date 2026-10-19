"""Which sentences of a judgment feed each summary field.

The brief names generic roles (Issue, Facts, Reasoning, Decision). Member A's Stage 1 labels are the
Bhattacharya et al. (2019) roles, so the mapping is:

| field              | brief's source          | A's roles used                                             |
|--------------------|-------------------------|------------------------------------------------------------|
| issue              | Issue                   | Issue                                                      |
| material_facts     | Facts                   | Facts                                                      |
| ratio              | Reasoning + Decision    | Ratio of the decision + Ruling by Present Court            |
| decision           | Decision                | Ruling by Present Court                                    |
| relevance_to_query | Reasoning, by query sim | Ratio of the decision + Precedent + Statute, ranked by     |
|                    |                         | TF-IDF cosine to the query (top `relevance_top` sentences) |

Fallbacks (recorded in `provenance`, never silent):
  * issue: A's gold data has no Issue sentences, so when none are labelled we take Facts/Ratio sentences
    that state a question ("whether", "the question", "issue", "point for consideration");
  * decision: when no Ruling by Present Court is labelled, the last `decision_tail` sentences;
  * any other empty field: the field is left empty (no text is invented).
"""
from __future__ import annotations

import re
from typing import Dict, List, Sequence, Tuple

FIELDS = ["issue", "material_facts", "ratio", "decision", "relevance_to_query"]

FIELD_ROLES = {
    "issue": ["Issue"],
    "material_facts": ["Facts"],
    "ratio": ["Ratio of the decision", "Ruling by Present Court"],
    "decision": ["Ruling by Present Court"],
    "relevance_to_query": ["Ratio of the decision", "Precedent", "Statute"],
}

_ISSUE_CUE = re.compile(r"\bwhether\b|\bthe (?:sole |only |short |main |real )?questions?\b|\bissues?\b|"
                        r"\bpoints? (?:for|that arise for) (?:consideration|determination)\b|\bfalls? for consideration\b", re.I)


def rank_by_query(query: str, sents: Sequence[str], top: int) -> List[int]:
    """Indices of the `top` sentences most similar to the query (TF-IDF cosine), kept in document order."""
    if not sents or not query:
        return list(range(min(top, len(sents))))
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    try:
        m = v.fit_transform(list(sents) + [query])
    except ValueError:                                   # only stop words
        return list(range(min(top, len(sents))))
    sim = (m[:-1] @ m[-1].T).toarray().ravel()
    best = sorted(range(len(sents)), key=lambda i: -sim[i])[:top]
    return sorted(best)


def field_sources(sentences: Sequence[Tuple[str, str]], query: str = "", relevance_top: int = 8,
                  decision_tail: int = 5) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """sentences: [(text, role)] in document order -> ({field: [source sentences]}, {field: provenance})."""
    by_role: Dict[str, List[str]] = {}
    for t, r in sentences:
        by_role.setdefault(r, []).append(t)
    src, prov = {}, {}
    for f in FIELDS:
        roles = FIELD_ROLES[f]
        s = [t for t, r in sentences if r in roles]
        prov[f] = "roles:" + "+".join(r for r in roles if by_role.get(r))
        if f == "relevance_to_query":
            s = [s[i] for i in rank_by_query(query, s, relevance_top)]
        if not s and f == "issue":
            s = [t for t, r in sentences if r in ("Facts", "Ratio of the decision") and _ISSUE_CUE.search(t)][:6]
            prov[f] = "fallback:question_cues"
        if not s and f == "decision":
            s = [t for t, _ in sentences[-decision_tail:]]
            prov[f] = f"fallback:last_{decision_tail}_sentences"
        if not s:
            prov[f] = "empty"
        src[f] = s
    return src, prov
