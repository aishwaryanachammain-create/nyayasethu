"""Loaders for the open-access datasets. Each returns `Judgment` objects (common/schema.py).

| Dataset | Licence | Access |
|---|---|---|
| Bhattacharya et al. 2019 (JURIX) — 50 SC judgments, sentence roles | MIT | github.com/Law-AI/semantic-segmentation |
| AILA 2019 (FIRE) — 50 queries, 2,914 candidates, qrels | CC BY 4.0 | Zenodo 10.5281/zenodo.4063986 |
| Indian Supreme Court Judgments — ~35K English PDFs + metadata | CC BY 4.0 | AWS Open Data, s3://indian-supreme-court-judgments |
| SC-PCR — our citation-masked benchmark built from the above | CC BY 4.0 (derived) | stage0_corpus/build_sc_pcr.py |
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common.schema import Judgment, Sentence

# ----------------------------------------------------------------------------- Bhattacharya 2019
BHATT_LABELS = ["Facts", "Ruling by Lower Court", "Argument", "Statute", "Precedent",
                "Ratio of the decision", "Ruling by Present Court"]


def load_bhattacharya(src: str | Path) -> List[Judgment]:
    """`src` is the repo zip or an extracted folder. Uses data/text/*.txt: `sentence<TAB>label` per line.
    categories.txt (domain of each judgment) is stored in Judgment.source as 'Bhattacharya2019/<domain>'."""
    src = Path(src)
    files: Dict[str, str] = {}
    cats: Dict[str, str] = {}
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                if "/data/text/" in n and n.endswith(".txt"):
                    files[Path(n).stem] = z.read(n).decode("utf-8", errors="ignore")
                elif n.endswith("categories.txt"):
                    cats_txt = z.read(n).decode("utf-8", errors="ignore")
                    cats = _parse_categories(cats_txt)
    else:
        for p in sorted(src.rglob("data/text/*.txt")) or sorted(src.rglob("*.txt")):
            if p.name != "categories.txt":
                files[p.stem] = p.read_text(encoding="utf-8", errors="ignore")
        c = next(iter(src.rglob("categories.txt")), None)
        if c:
            cats = _parse_categories(c.read_text(encoding="utf-8", errors="ignore"))
    out = []
    for stem, txt in sorted(files.items()):
        j = Judgment(case_id=f"bhatt:{stem}", source=f"Bhattacharya2019/{cats.get(stem, '?')}", court="SC",
                     date=stem.split("_")[0] if stem[:4].isdigit() else None)
        for line in txt.splitlines():
            if "\t" not in line:
                continue
            text, lab = line.rsplit("\t", 1)
            text, lab = text.strip(), lab.strip()
            if text:
                j.sentences.append(Sentence(idx=len(j.sentences), text=text, role=lab if lab in BHATT_LABELS else None))
        j.text = " ".join(s.text for s in j.sentences)
        out.append(j)
    return out


def _parse_categories(txt: str) -> Dict[str, str]:
    out = {}
    for line in txt.splitlines():
        if "\t" in line:
            cat, docs = line.split("\t", 1)
            for d in docs.split():
                out[d.strip()] = cat.strip()
    return out


# ----------------------------------------------------------------------------- AILA 2019
def load_aila(src: str | Path) -> Tuple[List[Judgment], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Returns (candidate judgments, {qid: query text}, {qid: {case_id: rel}}). `src` = zip or folder."""
    src = Path(src)
    texts: Dict[str, str] = {}
    if src.suffix == ".zip":
        with zipfile.ZipFile(src) as z:
            for n in z.namelist():
                if not n.endswith("/"):
                    texts[n] = z.read(n).decode("utf-8", errors="ignore")
    else:
        for p in src.rglob("*"):
            if p.is_file() and p.suffix in (".txt", ""):
                texts[str(p.relative_to(src))] = p.read_text(encoding="utf-8", errors="ignore")
    docs, queries, qrels = [], {}, defaultdict(dict)
    for name, t in sorted(texts.items()):
        base = Path(name).name
        if "Object_casedocs" in name and base.endswith(".txt"):
            docs.append(Judgment(case_id=f"aila:{Path(base).stem}", source="AILA2019", court="SC", text=t))
        elif base.startswith("Query_doc"):
            for line in t.splitlines():
                if "||" in line:
                    qid, q = line.split("||", 1)
                    queries[qid.strip()] = q.strip()
        elif "relevance_judgments_priorcases" in base:
            for line in t.splitlines():
                parts = line.split()
                if len(parts) >= 4:
                    qrels[parts[0]][f"aila:{parts[2]}"] = int(parts[3])
    return docs, queries, dict(qrels)


# ----------------------------------------------------------------------------- SC judgments (AWS)
def pdf_to_text(path: str | Path) -> str:
    """PyMuPDF if installed (fast), else pypdf (pure Python)."""
    try:
        import fitz
        with fitz.open(str(path)) as doc:
            return "\n".join(page.get_text() for page in doc)
    except ImportError:
        from pypdf import PdfReader
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)


def bench_from_judges(judges) -> Optional[int]:
    if judges is None or (isinstance(judges, float)):
        return None
    if isinstance(judges, (list, tuple)):
        names = [j for j in judges if str(j).strip()]
    else:
        names = [n for n in re.split(r",|;|\band\b|\n", str(judges)) if re.search(r"[A-Za-z]", n)]
    return len(names) or None


def load_sc_metadata(meta_dir: str | Path) -> Dict[str, Dict]:
    """All metadata.parquet files under meta_dir, keyed by PDF stem, cnr and case_id."""
    import pandas as pd
    meta: Dict[str, Dict] = {}
    for f in sorted(Path(meta_dir).rglob("*.parquet")):
        for r in pd.read_parquet(f).to_dict("records"):
            for k in ("path", "cnr", "case_id"):
                v = r.get(k)
                if v:
                    meta[Path(str(v)).stem] = r
    return meta


def load_sc_judgments(pdf_dir: str | Path, meta_dir: Optional[str | Path] = None, limit: Optional[int] = None,
                      min_chars: int = 500) -> List[Judgment]:
    """English PDFs from the extracted english.tar archives. Text is cached as <pdf>.txt."""
    meta = load_sc_metadata(meta_dir) if meta_dir and Path(meta_dir).exists() else {}
    out = []
    for i, p in enumerate(sorted(Path(pdf_dir).rglob("*.pdf"))):
        if limit and i >= limit:
            break
        cache = p.with_suffix(".txt")
        if cache.exists():
            text = cache.read_text(encoding="utf-8", errors="ignore")
        else:
            try:
                text = pdf_to_text(p)
            except Exception as e:
                print(f"skip {p.name}: {e}")
                continue
            cache.write_text(text, encoding="utf-8")
        if len(text) < min_chars:
            continue
        m = meta.get(p.stem, {})
        date = str(m.get("decision_date") or "")[:10] or None
        cit = m.get("citation")
        aliases = [c.strip() for c in re.split(r";|\|", str(cit)) if c and c.strip()] if cit else []
        out.append(Judgment(case_id=f"sc:{m.get('cnr') or p.stem}", source="AWS-SC", court="SC", date=date,
                            bench_size=bench_from_judges(m.get("judge")), title=str(m.get("title") or ""),
                            aliases=aliases, text=text))
    return out


# ----------------------------------------------------------------------------- SC-PCR / IL-PCR-style
def load_pcr(root: str | Path, split: str = "test"):
    """Benchmark written by build_sc_pcr.py: {split}_queries.jsonl + candidates.jsonl."""
    root = Path(root)
    docs = [Judgment(case_id=r["id"], source="SC-PCR", court="SC", text=r["text"])
            for r in map(json.loads, open(root / "candidates.jsonl", encoding="utf-8"))]
    queries, qrels = {}, {}
    for r in map(json.loads, open(root / f"{split}_queries.jsonl", encoding="utf-8")):
        queries[r["id"]] = r["text"]
        qrels[r["id"]] = {c: 1 for c in r["relevant_candidates"]}
    return docs, queries, qrels
