"""Step 2: product facts -> creative brief.

The brief has the same shape as the AdProducer benchmark prompts: the product is named by description, one
selling point is given, and the details that must be correct are listed without their values (the agent has to
find them).  Any OpenAI-compatible chat endpoint can write it (BRIEF_BASE_URL / BRIEF_MODEL / BRIEF_API_KEY).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import requests

PROMPT = (Path(__file__).parent / "prompts" / "brief_writer.txt").read_text()


def write_brief(facts: dict, *, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                auth_header: str | None = None, timeout: int = 300) -> str:
    base_url = base_url or os.environ["BRIEF_BASE_URL"]
    model = model or os.environ["BRIEF_MODEL"]
    api_key = api_key or os.environ.get("BRIEF_API_KEY", "EMPTY")
    auth_header = auth_header or os.environ.get("BRIEF_AUTH_HEADER", "Authorization")
    headers = {auth_header: (f"Bearer {api_key}" if auth_header.lower() == "authorization" else api_key)}
    page = {k: facts.get(k) for k in ("title", "bullets", "specs", "url")}
    body = {"model": model, "temperature": 0.3, "max_tokens": 4096,
            "messages": [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": "Product page:\n" + json.dumps(page, ensure_ascii=False, indent=1)}]}
    r = requests.post(base_url.rstrip("/") + "/chat/completions", json=body, headers=headers, timeout=timeout)
    r.raise_for_status()
    text = (r.json()["choices"][0]["message"].get("content") or "").strip()
    if not text.startswith("Make a 10-second"):
        raise RuntimeError(f"unexpected brief: {text[:200]}")
    return text
