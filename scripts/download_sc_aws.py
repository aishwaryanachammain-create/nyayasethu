import tarfile, requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "sc_aws"
BASE = "https://indian-supreme-court-judgments.s3.amazonaws.com"
YEARS = range(2018, 2024)   # start small; widen once it works

def fetch(url, dest):
    if dest.exists():
        return True
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        print(f"  {r.status_code} for {url}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    return True

for y in YEARS:
    ydir = OUT / f"year={y}"
    print("Year", y)
    tar_path = ydir / "english.tar"
    if fetch(f"{BASE}/data/tar/year={y}/english/english.tar", tar_path):
        with tarfile.open(tar_path) as t:
            t.extractall(ydir / "pdf")
        print("  PDFs:", len(list((ydir / "pdf").rglob("*.pdf"))))
    fetch(f"{BASE}/metadata/tar/year={y}/metadata.index.json", ydir / "metadata.index.json")