"""Re-export of Member B's Stage 5 for A's API (app/b_interface.py imports `summary.summarize`).

The API summarises every top-k hit on each /search, so it uses `stage5.api_mode` (default: extractive,
under a second per case). Legal-LED with the NLI filter takes minutes per case on a laptop, so the UI runs
it on demand for one case at a time.
"""
from typing import Dict, Optional


def summarize(query: str, case_id: str, mode: Optional[str] = None) -> Dict:
    from common.config import cfg
    from stage5_summary import summarize as _summarize
    return _summarize(query, case_id, mode=mode or cfg()["stage5"].get("api_mode", "extractive"))


__all__ = ["summarize"]
