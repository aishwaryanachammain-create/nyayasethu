"""Load config.yaml once; scripts call `cfg()`."""
from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any, Dict

import yaml

ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=4)
def cfg(path: str | None = None) -> Dict[str, Any]:
    p = Path(path or os.environ.get("NYAYASETHU_CONFIG", ROOT / "config.yaml"))
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Resolve a paths.* entry relative to the repo root."""
    return ROOT / cfg()["paths"][key]
