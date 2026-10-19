"""Load config.yaml once; scripts call `cfg()`."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]

# Hand-off files that have a stand-in under paths.mock (see data/CONTRACT.md).
MOCKABLE = {"judgments", "segments", "edges", "pagerank", "retrieval", "validity", "summary"}


@functools.lru_cache(maxsize=4)
def cfg(path: str | None = None) -> Dict[str, Any]:
    p = Path(path or os.environ.get("NYAYASETHU_CONFIG", ROOT / "config.yaml"))
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Resolve a paths.* entry relative to the repo root.

    With `use_mock: true`, hand-off files resolve to their mock stand-in in paths.mock.
    """
    c = cfg()
    p = ROOT / c["paths"][key]
    if c.get("use_mock") and key in MOCKABLE:
        return ROOT / c["paths"]["mock"] / p.name
    return p
