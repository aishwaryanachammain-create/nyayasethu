"""Download IN-Abs / IN-Ext / UK-Abs (Shukla et al., AACL 2022; Zenodo 10.5281/zenodo.7152317, open access).

    python scripts/download_in_abs.py        # -> data/raw/in_abs_ext/dataset/{IN-Abs,IN-Ext,UK-Abs}
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.config import path

URL = "https://zenodo.org/records/7152317/files/{}?download=1"
OUT = path("raw") / "in_abs_ext"

OUT.mkdir(parents=True, exist_ok=True)
for name in ("dataset.zip", "README.md"):
    dst = OUT / name
    if not dst.exists():
        print("downloading", name)
        urllib.request.urlretrieve(URL.format(name), dst)
if not (OUT / "dataset" / "IN-Abs").exists():
    zipfile.ZipFile(OUT / "dataset.zip").extractall(OUT)
for d in ("IN-Abs/train-data/judgement", "IN-Abs/test-data/judgement", "IN-Ext/judgement"):
    print(d, len(list((OUT / "dataset" / d).glob("*.txt"))), "files")
