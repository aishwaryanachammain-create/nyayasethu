import re
from pathlib import Path
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
CIT = re.compile(
    r"\(\d{4}\)\s*\d+\s*SCC\s*\d+"            # (2005) 3 SCC 123
    r"|AIR\s*\d{4}\s*SC\s*\d+"                # AIR 1963 SC 1405
    r"|\[\d{4}\]\s*\d+\s*S\.?C\.?R\.?\s*\d+"  # [1964] 1 SCR 515
    r"|\d{4}\s*\(\d+\)\s*SCR\s*\d+"           # 1964 (1) SCR 515
    r"|\d{4}\s*SCC\s*OnLine\s*[A-Za-z]+\s*\d+"
    r"|\d{4}\s*INSC\s*\d+", re.I)
MASK = re.compile(r"<\s*CITATION\s*>|\[CITATION\]|<CIT>", re.I)

ds = load_from_disk(str(ROOT / "data/raw/il_tur/cjpe"))
print(ds)
split = list(ds.keys())[0]
rows = ds[split].select(range(min(100, len(ds[split]))))

def as_text(v):
    if isinstance(v, str): return v
    if isinstance(v, list): return " ".join(as_text(x) for x in v)
    return ""

text_col = max(rows.column_names, key=lambda c: len(as_text(rows[0][c])))
print("Text column:", text_col)

hits = [len(CIT.findall(as_text(r[text_col]))) for r in rows]
masks = [len(MASK.findall(as_text(r[text_col]))) for r in rows]
print(f"Docs with >=1 citation: {sum(h > 0 for h in hits)}/{len(rows)}")
print(f"Mean citations per doc: {sum(hits)/len(rows):.1f}")
print(f"Docs with masked citations: {sum(m > 0 for m in masks)}/{len(rows)}")

for i in range(3):
    print(f"\n--- Doc {i} ending ---\n", as_text(rows[i][text_col])[-400:])
    