"""AdProducer: product link -> brief -> search agent -> 10-second ad.

    python -m adproducer run  URL  --out out/my_product          # all four steps
    python -m adproducer brief URL --out out/my_product          # steps 1-2 only (product.json, brief.txt)
    python -m adproducer agent --out out/my_product              # step 3 from brief.txt
    python -m adproducer render --out out/my_product             # step 4 from prediction.json

Configuration comes from the environment (see .env.example): POLICY_BASE_URL / POLICY_MODEL for the agent,
BRIEF_* for the brief writer, SERPER_KEY_ID / JINA_API_KEYS / BROWSE_SUMMARY_* for the tools, H3_URLS for H3.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def step_brief(a, out: Path):
    from .brief import write_brief
    from .product_page import product_facts
    page = Path(a.html).read_text(errors="ignore") if a.html else None
    facts = product_facts(a.url, page)
    (out / "product.json").write_text(json.dumps(facts, ensure_ascii=False, indent=1))
    print(f"[1/4] product: {facts['title'][:100]}")
    brief = a.brief_text or write_brief(facts)
    (out / "brief.txt").write_text(brief + "\n")
    print(f"[2/4] brief:\n{brief}\n")


def step_agent(a, out: Path):
    from .agent import run_agent

    def show(turn, text, obs):
        import re
        act = re.search(r"<tool_call>(.*?)</tool_call>", text, re.S)
        print(f"  turn {turn + 1}: " + (act.group(1).strip()[:160].replace("\n", " ") if act else "answer"), flush=True)

    brief = (out / "brief.txt").read_text().strip()
    for attempt in range(1, a.attempts + 1):
        print(f"[3/4] search agent (attempt {attempt}) ...", flush=True)
        res = asyncio.run(run_agent(brief, out / "agent", base_url=os.environ["POLICY_BASE_URL"],
                                    model=os.environ.get("POLICY_MODEL", "adproducer"),
                                    api_key=os.environ.get("POLICY_API_KEY", "EMPTY"),
                                    temperature=a.temperature, top_p=a.top_p, on_turn=show))
        print(f"  termination={res['termination']} tool_calls={res['tool_calls']} gate={res['gate_errors'] or 'ok'}")
        if res["prediction"] and not res["gate_errors"]:
            keep = {k: v for k, v in res.items() if k != "trajectory"}
            (out / "prediction.json").write_text(json.dumps(keep, ensure_ascii=False, indent=1))
            return
    sys.exit("no usable answer from the agent")


def step_render(a, out: Path):
    from .render import render
    res = json.loads((out / "prediction.json").read_text())
    print("[4/4] rendering with MiniMax H3 (Ref2VA) ...", flush=True)
    mp4 = asyncio.run(render(res["prediction"], Path(res["run_dir"]), out / "ad.mp4"))
    print(f"  done: {mp4}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("run", "brief", "agent", "render"))
    ap.add_argument("url", nargs="?")
    ap.add_argument("--out", required=True)
    ap.add_argument("--html", help="saved product page (if the site blocks scripted requests)")
    ap.add_argument("--brief-text", help="use this brief instead of writing one")
    ap.add_argument("--attempts", type=int, default=1, help="agent attempts until an answer passes the checks")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=1.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd in ("run", "brief"):
        if not a.url:
            ap.error("a product URL is required")
        step_brief(a, out)
    if a.cmd in ("run", "agent"):
        step_agent(a, out)
    if a.cmd in ("run", "render"):
        step_render(a, out)


if __name__ == "__main__":
    main()
