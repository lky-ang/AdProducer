"""Step 3: brief -> grounded multi-shot prompt + reference images (the search agent).

The policy model (our Qwen3.5-9B AdProducer checkpoint, served by any OpenAI-compatible server such as
sglang) runs the agent loop of core/adproducer_rl.py: in each round it thinks and either calls one tool
(search / image_search / browse) or answers with a MiniMax H3 Ref2VA prompt that cites the chosen
images.  The loop, tools, caps and answer checks are exactly the ones used in training and evaluation.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
from datetime import date
from pathlib import Path


def _core(work_dir: Path):
    os.environ["ADP_WORK_DIR"] = str(work_dir.resolve())
    os.environ.setdefault("ADP_DATA_DIR", str(work_dir.resolve()))
    from .core import adproducer_rl as adp   # reads its configuration from the environment at import time
    return adp


def _to_openai(msg: dict) -> dict:
    parts = []
    for p in msg["content"]:
        if p["type"] == "text":
            parts.append({"type": "text", "text": p["text"]})
        else:
            buf = io.BytesIO()
            p["image"].save(buf, format="JPEG", quality=95)
            parts.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
    return {"role": msg["role"], "content": parts}


def _chat(adp, base_url, model, api_key, messages, temperature, top_p):
    rq = adp._requests()
    body = {"model": model, "messages": messages, "temperature": temperature, "top_p": top_p,
            "max_tokens": adp.GEN_MAX_NEW_TOKENS_PER_TURN}
    last = None
    for attempt in range(5):
        try:
            r = rq.post(base_url.rstrip("/") + "/chat/completions", json=body, timeout=1800,
                        headers={"Authorization": f"Bearer {api_key}"})
            if r.status_code == 400 and "context" in r.text.lower():
                return "", "context"
            if r.status_code == 200:
                c = r.json()["choices"][0]
                return c["message"].get("content") or "", c.get("finish_reason") or ""
            last = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(2 ** attempt)
    raise RuntimeError(f"policy endpoint failed: {last}")


async def run_agent(brief: str, work_dir: Path, *, base_url: str, model: str, api_key: str = "EMPTY",
                    temperature: float = 0.7, top_p: float = 1.0, item_id: str = "case",
                    on_turn=None) -> dict:
    """Returns {termination, prediction, run_dir, trajectory, tool_calls, gate_errors}."""
    adp = _core(work_dir)
    label = {"id": item_id, "brief": brief, "gt_facts": [], "official_images": []}
    env = adp.AdProducerEnv(item_id, label, adp.MAX_TOOL_CALLS + 1)
    messages = [{"role": "system", "content": adp.load_system_prompt(date.today().isoformat())},
                {"role": "user", "content": brief}]
    termination = None
    for turn in range(env.max_turns):
        text, finish = await asyncio.to_thread(_chat, adp, base_url, model, api_key, messages, temperature, top_p)
        if finish in ("length", "context"):
            termination = "truncated"
            break
        messages.append({"role": "assistant", "content": text})
        obs, done, _ = await env.astep(text)
        if on_turn:
            on_turn(turn, text, obs)
        if done:
            break
        messages.append(_to_openai(env.format_observation(obs)))
        if turn + 1 >= env.max_turns:
            env.termination = "max_turns"
    termination = termination or env.termination
    env.run_dir.mkdir(parents=True, exist_ok=True)
    traj = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else
             "".join(p.get("text", "<image>") for p in m["content"])} for m in messages]
    (env.run_dir / "trajectory.json").write_text(json.dumps(traj, ensure_ascii=False, indent=1))
    errs = adp.gate_errors(env.prediction, env.run_dir, None) if env.prediction else ["no answer"]
    return {"termination": termination, "prediction": env.prediction, "run_dir": str(env.run_dir),
            "tool_calls": env.tool_calls, "image_searches": env.image_searches,
            "gate_errors": [e for e in errs if not e.startswith("P3")], "trajectory": traj}
