"""Step 1: product link -> product facts.

Reads the merchant's product page and extracts what a brief needs: title, brand, bullet points and the
specification table.  Amazon pages are parsed field by field; any other page falls back to the OpenGraph /
<meta> description and the visible <li> items.  The result is a plain dict saved as product.json.
"""
from __future__ import annotations

import html
import json
import re
from urllib.parse import urlparse

import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/130.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}
SPEC_KEYS = ("Brand", "Model Number", "Model Name", "Color", "Wattage", "Heating Method", "Special Feature",
             "Product Dimensions", "Item Weight", "Form Factor", "Material", "Capacity", "Power Source")


def _text(fragment: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", fragment))).strip()


def fetch(url: str, timeout: int = 30) -> str:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    if "captcha" in r.text.lower() and "productTitle" not in r.text:
        raise RuntimeError("the page returned a captcha; save the page as HTML and pass --html instead")
    return r.text


def parse_amazon(page: str) -> dict:
    title = re.search(r'id="productTitle"[^>]*>(.*?)<', page, re.S)
    bullets = []
    block = re.search(r'id="feature-bullets".*?</ul>', page, re.S)
    if block:
        bullets = [_text(li) for li in re.findall(r"<li.*?</li>", block.group(0), re.S)]
    specs = {}
    for k in SPEC_KEYS:
        m = re.search(re.escape(k) + r"\s*</t[hd]>\s*<td[^>]*>(.*?)</td>", page, re.S) or \
            re.search(re.escape(k) + r".{0,400}?<td[^>]*>(.*?)</td>", page, re.S)
        if m:
            specs[k] = _text(m.group(1))[:200]
    images = re.findall(r'"hiRes":"(https://m\.media-amazon\.com/images/I/[^"]+)"', page)
    return {"title": _text(title.group(1)) if title else "", "bullets": [b for b in bullets if b],
            "specs": specs, "images": list(dict.fromkeys(images))[:8]}


def parse_generic(page: str) -> dict:
    def meta(name):
        m = re.search(rf'<meta[^>]+(?:property|name)="{name}"[^>]+content="([^"]*)"', page, re.I)
        return html.unescape(m.group(1)).strip() if m else ""
    title = meta("og:title") or _text((re.search(r"<title>(.*?)</title>", page, re.S | re.I) or [None, ""])[1])
    items = [_text(li) for li in re.findall(r"<li[^>]*>(.*?)</li>", page, re.S)]
    items = [t for t in items if 30 <= len(t) <= 400][:12]
    return {"title": title, "bullets": ([meta("og:description")] if meta("og:description") else []) + items,
            "specs": {}, "images": [meta("og:image")] if meta("og:image") else []}


def product_facts(url: str, page_html: str | None = None) -> dict:
    page = page_html if page_html is not None else fetch(url)
    host = urlparse(url).netloc.lower()
    facts = parse_amazon(page) if "amazon." in host else parse_generic(page)
    if not facts["title"]:
        facts = parse_generic(page)
    facts["url"] = url
    return facts


if __name__ == "__main__":
    import sys
    print(json.dumps(product_facts(sys.argv[1]), ensure_ascii=False, indent=1))
