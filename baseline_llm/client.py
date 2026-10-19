"""Minimal LLM client. The API key comes from .env (gitignored), never from the repo:

    LLM_PROVIDER=anthropic          # or: openai  (any OpenAI-compatible endpoint via LLM_BASE_URL)
    ANTHROPIC_API_KEY=...           # or OPENAI_API_KEY=...
    LLM_MODEL=claude-sonnet-5-5     # optional
"""
from __future__ import annotations

import os
import time
from typing import Tuple

import httpx

from common.config import ROOT

DEFAULT_MODEL = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-4o-mini"}


def settings() -> Tuple[str, str]:
    """-> (provider, model). Stops with a PENDING message when no key is configured."""
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    key = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key):
        raise SystemExit(f"No {key} in the environment or .env: this LLM baseline stays PENDING.")
    return provider, os.environ.get("LLM_MODEL", DEFAULT_MODEL.get(provider, DEFAULT_MODEL["openai"]))


def call(prompt: str, provider: str, model: str, client: httpx.Client, max_tokens: int = 1000) -> str:
    if provider == "anthropic":
        r = client.post("https://api.anthropic.com/v1/messages",
                        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
                        json={"model": model, "max_tokens": max_tokens, "temperature": 0,
                              "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"]
    base = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
    r = client.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                    json={"model": model, "temperature": 0, "max_tokens": max_tokens,
                          "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def call_with_retry(prompt: str, provider: str, model: str, client: httpx.Client, max_tokens: int = 1000,
                    tries: int = 4) -> str:
    for attempt in range(tries):
        try:
            return call(prompt, provider, model, client, max_tokens)
        except httpx.HTTPError:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt * 5)
