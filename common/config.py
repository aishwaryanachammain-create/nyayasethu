from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load(open(ROOT / "config.yaml"))
MOCKABLE = {"judgments", "segments", "edges", "pagerank", "retrieval", "validity", "summary"}

def path(key: str) -> Path:
    p = ROOT / CFG["paths"][key]
    if CFG.get("use_mock") and key in MOCKABLE:
        return ROOT / CFG["paths"]["mock"] / p.name
    return p
