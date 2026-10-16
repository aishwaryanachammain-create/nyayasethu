"""Splitting a query into (facts, full text) for routing (step 22).

* Long queries (a whole judgment, as in SC-PCR) are role-labelled with the Stage-1 labeller and
  their Facts sentences form the facts query.
* Short narratives (AILA, users) are mostly facts; sentences that ask the legal question are
  removed from the facts query by cue words.
The FULL text always goes to the Reasoning index.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from common.config import cfg
from common.schema import Judgment, Sentence
from stage0_corpus.splitter import split_sentences

QUESTION = re.compile(r"\?|\b(whether|can (they|he|she|it|i|we)|is it (legal|lawful|valid)|entitled to|liable|"
                      r"without (proving|proof|showing)|legally)\b", re.I)


def split_query(text: str, labeller=None, long_threshold: int = 15) -> Tuple[str, str]:
    sents = split_sentences(text) or [text]
    facts_roles = set(cfg()["stage2"]["facts_roles"])
    if labeller is not None and len(sents) >= long_threshold:
        d = Judgment(case_id="query", sentences=[Sentence(i, s) for i, s in enumerate(sents)])
        labeller.label([d])
        facts = " ".join(s.text for s in d.sentences if s.role in facts_roles)
        if facts.strip():
            return facts, text
    facts = " ".join(s for s in sents if not QUESTION.search(s))
    return (facts or text), text
