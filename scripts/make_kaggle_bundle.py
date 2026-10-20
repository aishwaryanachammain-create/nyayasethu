"""Pack the local data the Kaggle GPU notebook needs (notebooks/member_a_gpu.ipynb) into one zip, laid out like the
repo, so Kaggle runs on exactly the same corpus, role labels and benchmark as this machine.

    python scripts/make_kaggle_bundle.py           # -> data/processed/kaggle/nyayasethu-data.zip

Upload the zip as a private Kaggle dataset named `nyayasethu-data`.
"""
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed" / "kaggle" / "nyayasethu-data.zip"
INCLUDE = [
    "data/processed/judgments.jsonl",            # corpus (benchmark candidates come from here, with roles)
    "data/processed/judgments.jsonl.offsets.json",
    "data/processed/segments.parquet",           # Stage 1 role labels of the whole corpus (LSA labeller)
    "data/processed/sc-pcr",                     # benchmark built locally (fixed resolver)
    "data/processed/indegree.csv",
    "data/processed/roles_gold.jsonl",           # hand-annotated Issue judgments, if done
    "data/raw/aila2019",                         # AILA 2019 benchmark
    "data/raw/bhattacharya2019",                 # Stage 1 training data
    "models/stage1",                             # the LSA labeller (query splitting + role labels)
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = []
    for rel in INCLUDE:
        p = ROOT / rel
        if not p.exists():
            print(f"  (skipped, not present) {rel}")
            continue
        files += [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, f.relative_to(ROOT))
    print(f"{len(files)} files, {OUT.stat().st_size / 1e6:.0f} MB -> {OUT}")


if __name__ == "__main__":
    main()
