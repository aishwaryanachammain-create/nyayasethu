"""Stage 5 (Member B): query-focused, role-aware summary of one judgment, filtered for faithfulness.

    from stage5_summary import summarize
    summarize("landlord kept my deposit without proving loss", "sc:ESCR010000012018")
    # -> {status, case_id, query, issue, material_facts, ratio, decision, relevance_to_query,
    #     rejected: [str], rejected_detail, provenance, roles_source, generator, nli}

Pipeline: roles (docs.roles_for) -> field sources (fields.field_sources) -> generation (Legal-LED, or
extractive) -> NLI filter per field against that field's source sentences (nli.filter_text).
In extractive mode every sentence is copied from the source, so the NLI filter is skipped.
"""
from __future__ import annotations

import functools
from typing import Dict, Optional

from .fields import FIELDS


@functools.lru_cache(maxsize=2)
def _generator(mode: str):
    from .generate import make_generator
    return make_generator(mode)


@functools.lru_cache(maxsize=1)
def _scorer():
    from .nli import NLIScorer
    return NLIScorer()


def summarize(query: str, case_id: str, mode: Optional[str] = None, use_nli: bool = True,
              generator=None, scorer=None) -> Dict:
    from common.config import cfg
    from .docs import get_judgment, roles_for
    from .fields import field_sources
    from .generate import generate_fields
    from .nli import filter_text

    s5 = cfg()["stage5"]
    mode = mode or s5.get("mode", "abstractive")
    doc = get_judgment(case_id)
    if doc is None or not doc.sentences:
        return {"status": "not_found", "case_id": case_id, "query": query, "note": "judgment not in paths.judgments"}
    sents, roles_source = roles_for(doc)
    src, prov = field_sources(sents, query, s5.get("relevance_top", 8))
    gen = generator or _generator(mode)
    draft = generate_fields(gen, src, query)
    out = {"status": "ok", "case_id": case_id, "query": query, "title": doc.title}
    rejected, detail = [], []
    nli_on = use_nli and mode != "extractive" and getattr(gen, "name", "") != "extractive"
    for f in FIELDS:
        if nli_on and draft[f]:
            kept, rej = filter_text(draft[f], src[f], scorer or _scorer())
            out[f] = kept
            rejected += [f"{f}: {r['sentence']}" for r in rej]
            detail += [{"field": f, **r} for r in rej]
        else:
            out[f] = draft[f]
    out.update({"rejected": rejected, "rejected_detail": detail, "provenance": prov, "roles_source": roles_source,
                "generator": getattr(gen, "name", type(gen).__name__),
                "nli": (cfg()["models"]["nli"] if nli_on else "skipped")})
    return out
