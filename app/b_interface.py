"""Member B's functions, imported if present (step 30).

Member B delivers `validity.get_validity(case_id) -> dict` and `summary.summarize(query, case_id) -> dict`.
Until those modules are in the repo, the API returns an explicit 'pending' marker instead of
inventing a verdict or summary.
"""
from __future__ import annotations

from typing import Dict

try:
    from validity import get_validity as _get_validity   # Member B: validity/__init__.py
except ImportError:
    _get_validity = None
try:
    from summary import summarize as _summarize          # Member B: summary/__init__.py
except ImportError:
    _summarize = None


def get_validity(case_id: str) -> Dict:
    if _get_validity is None:
        return {"status": "pending", "label": None, "note": "Member B's Stage 4 not integrated yet"}
    return _get_validity(case_id)


def summarize(query: str, case_id: str) -> Dict:
    if _summarize is None:
        return {"status": "pending", "fields": None, "note": "Member B's Stage 5 not integrated yet"}
    return _summarize(query, case_id)


def available() -> Dict[str, bool]:
    return {"get_validity": _get_validity is not None, "summarize": _summarize is not None}
