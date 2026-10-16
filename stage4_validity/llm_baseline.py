"""Zero-shot LLM baseline for treatment classification, run on the gold contexts.

The API key comes from .env (gitignored), never from the repo:
    LLM_PROVIDER=anthropic          # or: openai  (any OpenAI-compatible endpoint via LLM_BASE_URL)
    ANTHROPIC_API_KEY=...           # or OPENAI_API_KEY=...
    LLM_MODEL=claude-sonnet-5-5     # optional

    python -m stage4_validity.llm_baseline            # -> eval/stage4_llm_preds.csv (resumes if interrupted)

Without a key the script stops and nothing is reported: the baseline stays PENDING.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time

import httpx
import pandas as pd

from common.config import ROOT

from .export_gold import GOLD
from .labels import LABELS

OUT = ROOT / "eval" / "stage4_llm_preds.csv"

PROMPT = """You are classifying how an Indian court judgment treats a precedent it cites.

Cited case: {title}
The citation is marked with [[ ]] in the citing sentence.

Citing sentence:
{sentence}

Surrounding text:
{context}

Classify the citing court's treatment of the cited case:
- followed: the court relies on, applies, follows or approves it
- distinguished: the court holds it inapplicable or distinguishes it on facts or law
- overruled: the court overrules it, or holds it is not good law / wrongly decided / per incuriam
- neutral: it is only referred to, summarised, or cited by a party, with no treatment by the court

Answer with JSON only: {{"label": "<followed|distinguished|overruled|neutral>"}}"""


def _call(prompt: str, provider: str, model: str, client: httpx.Client) -> str:
    if provider == "anthropic":
        r = client.post("https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                        json={"model": model, "max_tokens": 50, "temperature": 0,
                              "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    r = client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                    json={"model": model, "temperature": 0, "max_tokens": 50,
                          "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def parse(text: str):
    m = re.search(r'"label"\s*:\s*"(\w+)"', text or "")
    lab = (m.group(1) if m else (text or "").strip().split()[0] if text and text.strip() else "").lower()
    return lab if lab in LABELS else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", default=str(GOLD))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key):
        raise SystemExit(f"No {key} in the environment or .env: the LLM baseline stays PENDING.")
    model = os.environ.get("LLM_MODEL", "claude-sonnet-5-5" if provider == "anthropic" else "gpt-4o-mini")
    g = pd.read_csv(a.gold, keep_default_na=False)
    done = set(pd.read_csv(a.out).gold_id) if os.path.exists(a.out) else set()
    todo = g[~g.gold_id.isin(done)].head(a.limit) if a.limit else g[~g.gold_id.isin(done)]
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f, httpx.Client(timeout=60) as client:
        w = csv.writer(f)
        if new:
            w.writerow(["gold_id", "provider", "model", "llm_label", "raw"])
        for r in todo.itertuples():
            prompt = PROMPT.format(title=r.cited_title or "(unknown)", sentence=r.sentence, context=r.context)
            for attempt in range(4):
                try:
                    raw = _call(prompt, provider, model, client)
                    break
                except httpx.HTTPError:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt * 5)
            w.writerow([r.gold_id, provider, model, parse(raw) or "", json.dumps(raw)])
            f.flush()
    print(f"{len(todo)} new predictions -> {a.out}")


if __name__ == "__main__":
    main()
