"""Step 3 — download the open datasets (no forms, logins or gated downloads). Standard library only.

    python -m stage0_corpus.download                      # AILA + Bhattacharya + SC metadata (all years) + SC PDFs for config years
    python -m stage0_corpus.download --years 2021-2023    # choose SC PDF years
    python -m stage0_corpus.download --skip-pdfs          # only the small files

InLegalBERT weights (law-ai/InLegalBERT, MIT, not gated) are fetched automatically by
`transformers` the first time Stage 1 runs.
"""
import argparse
import json
import shutil
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

from common.config import cfg, path


def fetch(url: str, dest: Path, quiet: bool = False) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
        if not quiet:
            print(f"  FAILED {url}: {e}")
        tmp.unlink(missing_ok=True)
        return False
    tmp.rename(dest)
    if not quiet:
        print(f"  {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return True


def main():
    c = cfg()
    p = argparse.ArgumentParser()
    p.add_argument("--years", help="SC PDF years, e.g. 2023 or 2018-2023 (default: config sc_years)")
    p.add_argument("--skip-pdfs", action="store_true")
    a = p.parse_args()
    raw = path("raw")
    ds = c["datasets"]

    print("AILA 2019")
    fetch(ds["aila2019_zip"], raw / "aila2019" / "AILA_2019_dataset.zip")
    print("Bhattacharya 2019 rhetorical roles")
    fetch(ds["bhattacharya2019_zip"], raw / "bhattacharya2019" / "semantic-segmentation.zip")

    print("SC metadata parquet, 1950-2025 (small)")
    got = sum(fetch(f"{ds['sc_bucket']}/metadata/parquet/year={y}/metadata.parquet",
                    raw / "sc-judgments" / "metadata" / f"year={y}" / "metadata.parquet", quiet=True)
              for y in range(1950, 2026))
    print(f"  {got} years of metadata")

    if not a.skip_pdfs:
        if a.years:
            lo, _, hi = a.years.partition("-")
            years = range(int(lo), int(hi or lo) + 1)
        else:
            years = c["datasets"]["sc_years"]
        for y in years:
            print(f"SC English judgments {y}")
            base = f"{ds['sc_bucket']}/data/tar/year={y}/english"
            idx = raw / "sc-judgments" / "tar" / str(y) / "english.index.json"
            parts = ["english.tar"]
            if fetch(f"{base}/english.index.json", idx, quiet=True):
                info = json.loads(idx.read_text())
                parts = [x["name"] for x in info.get("parts", [])] or parts   # large years are split
                print(f"  {info.get('file_count', '?')} judgments, {info.get('total_size_human', '?')}, {len(parts)} part(s)")
            out = raw / "sc-judgments" / "pdf" / str(y)
            for part in parts:
                tar = raw / "sc-judgments" / "tar" / str(y) / part
                if fetch(f"{base}/{part}", tar):
                    with tarfile.open(tar) as t:
                        try:
                            t.extractall(out, filter="data")
                        except TypeError:          # Python < 3.12
                            t.extractall(out)
                    print(f"  extracted {part} -> {out}")
    print("Done. Note the download date and DOIs in data/README.md.")


if __name__ == "__main__":
    main()
