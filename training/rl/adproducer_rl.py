"""AdProducer RL: slime custom rollout + reward for the commercial-video search agent.

One file, four parts (search for the === markers):
  1. Protocol & tools   - the Gen-Searcher agent loop (think / tool_call / answer, IMG_### registry,
                          8-call cap, FINAL_MESSAGE, repeated-response stop) and the three tools
                          (search / image_search / browse), in the same format as the SFT data, so the
                          policy sees exactly what the SFT data shows.
  2. slime integration  - `generate(args, sample, sampling_params)`: async multi-turn VLM rollout built on the
                          helpers of examples/geo3k_vlm_multi_turn/rollout.py (token/logprob/loss-mask/image
                          bookkeeping is theirs; the loop is ours so tool calls can be awaited).
  3. Reward             - Gen-Searcher's dual reward, R = (1-a)*R_video + a*R_text, a = 0.5:
                          gate (compute_reward's zero cases + H3 format) -> R_text (Appendix B prompt) and
                          R_video (prompts/reward_video.txt on the whole generated video).  Both judges get the
                          brief, the verified facts, the official images and the GT video; R_text also sees the
                          chosen reference images.  Failures are handled as in Gen-Searcher (see the section
                          header).  `reward_func(args, samples)` is the batched slime RM entry point (--group-rm);
                          slime's default GRPO normalization is used.
  4. CLIs               - `prepare-data` (rl_data.json -> slime jsonl), `text-required` (per-brief text requirement).

Everything configurable is an environment variable (see README.md / .env.example).  Nothing here depends on
slime at import time, so the CLIs run on a machine without it.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# ============================================================================================
# Configuration (environment)
# ============================================================================================

HERE = Path(__file__).resolve().parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


# --- agent protocol (must match the SFT data; do not change without re-running SFT) ---
PROMPTS_DIR = HERE / "prompts"
SYSTEM_PROMPT_PATH = Path(_env("ADP_SYSTEM_PROMPT", str(PROMPTS_DIR / "system_prompt.txt")))
MAX_TOOL_CALLS = 8                   # Gen-Searcher cap (MAX_LLM_CALL_PER_RUN = 9)
MAX_IMAGE_SEARCHES = 2               # system_prompt.txt: "At most 2 image_search calls"
MAX_IMAGES_FOR_MODEL = 5             # images shown per image_search
SEARCH_RESULTS_PER_QUERY = 10        # Gen-Searcher _text_search_sync default; Serper charges per query
MAX_REF_IMAGES = 5                   # answer keeps the first 5 reference images
MIN_REF_IMAGES = 1
VIDEO_DURATION_S = 10
MIN_SHOTS = 3
MIN_LAST_SHOT_S = 1.0
IMAGE_MAX_PIXELS = _env_int("ADP_IMAGE_MAX_PIXELS", 4096 * 32 * 32)   # 4096 tokens per image; 0 disables the cap
BROWSE_CHUNK_CHARS = 350_000

# --- services ---
SERPER_KEY = _env("SERPER_KEY_ID")
SERPER_SEARCH_URL = _env("TEXT_SEARCH_API_BASE_URL", "https://google.serper.dev/search")
SERPER_IMAGE_URL = _env("IMAGE_SEARCH_API_BASE_URL", "https://google.serper.dev/images")
JINA_KEY = _env("JINA_API_KEYS")
JINA_READER_URL = "https://r.jina.ai/"
BROWSE_MODEL = _env("BROWSE_SUMMARY_MODEL", "Qwen/Qwen3-VL-30B-A3B-Instruct")
BROWSE_BASE_URL = _env("BROWSE_SUMMARY_BASE_URL")
BROWSE_API_KEY = _env("BROWSE_SUMMARY_API_KEY", "EMPTY")
TOOL_CONCURRENCY = _env_int("ADP_TOOL_CONCURRENCY", 64)

H3_URLS = [u for u in _env("H3_URLS").split(",") if u.strip()]        # e.g. http://h3-a:30011,http://h3-b:30011
# Dynamic alternative: a shared directory with one "<host>.url" file per live H3 server (written by the server job when
# healthy, removed when it stops).  Re-read every 15 s, so servers can join or leave while training runs.
H3_URLS_DIR = _env("H3_URLS_DIR")
H3_BAD_COOLDOWN_S = 300                   # a server that refused/dropped a connection is skipped for 5 min
H3_SEED = _env_int("H3_SEED", 1101)       # the GT videos were generated with seed 1101
H3_STEPS = _env_int("H3_STEPS", 50)        # same as the GT videos: 50 steps,
H3_FLOW_SHIFT = _env_float("H3_FLOW_SHIFT", 12.0)               # flow_shift 12.0,
H3_AUDIO_FLOW_SHIFT = _env_float("H3_AUDIO_FLOW_SHIFT", 3.0)    # audio_flow_shift 3.0 (audio is dropped afterwards)
H3_SHORT_EDGE = _env_int("H3_SHORT_EDGE", 480)
H3_CONCURRENCY_PER_SERVER = _env_int("H3_CONCURRENCY_PER_SERVER", 1)   # sglang batching_max_size defaults to 1
H3_TIMEOUT_S = _env_int("H3_TIMEOUT_S", 1800)
H3_RETRIES = 3

JUDGE_BACKEND = _env("JUDGE_BACKEND", "gemini")          # gemini (Google REST) | openai (OpenAI-compatible gateway)
JUDGE_MODEL = _env("JUDGE_MODEL", "gemini-3.1-pro-preview")
JUDGE_API_KEY = _env("JUDGE_API_KEY")
JUDGE_BASE_URL = _env("JUDGE_BASE_URL")                  # gemini: empty = Google; openai: the gateway's /v1 URL
JUDGE_AUTH_HEADER = _env("JUDGE_AUTH_HEADER", "Authorization")   # header name, if your gateway needs another one
JUDGE_MAX_TOKENS = _env_int("JUDGE_MAX_TOKENS", 16384)   # Gen-Searcher used 8192 (GPT-4.1); Gemini 3 also spends
                                                         # output tokens on thinking, so leave room
JUDGE_RETRIES = 3                                        # VideoGen-Agent D.4
JUDGE_CONCURRENCY = _env_int("ADP_JUDGE_CONCURRENCY", 16)
TEXT_REWARD_COEF = _env_float("GEN_REWARD_TEXT_COEF", 0.5)   # alpha; Gen-Searcher paper uses 0.5

# --- paths ---
DATA_DIR = Path(_env("ADP_DATA_DIR", str(HERE / "data")))   # gt_video/, cached_image/ 
WORK_DIR = Path(_env("ADP_WORK_DIR", str(HERE / "work")))                             # per-rollout outputs
GEN_MAX_NEW_TOKENS_PER_TURN = _env_int("GEN_MAX_NEW_TOKENS_PER_TURN", 6144)

IMAGE_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# ============================================================================================
# === 1. Protocol & tools ====================================================================
# ============================================================================================

FINAL_MESSAGE = """
=== FINAL STEP: OUTPUT ANSWER NOW ===
This is your FINAL step. You have NO more chances.

CRITICAL RULES:
1. Tool calls are ABSOLUTELY FORBIDDEN. Any <tool_call> will be IGNORED.
2. You MUST output <answer>...</answer> immediately.
3. Even if information is incomplete or uncertain, you MUST generate an answer with what you have.
4. Use available IMG_### from previous searches. If none, describe the shots without references.
6. "gen_prompt" keeps the six H3 sections (subject_definitions, summary, retention_analysis,
   detailed_description, overall_soundscape, non_diegetic_music) and refers to images only as <Picture N>.
   The video is 10 seconds and silent: overall_soundscape and non_diegetic_music are "N/A".
5. Do NOT write <think>. Do NOT write <tool_call>. Do NOT explain or apologize.

FORMAT (output this EXACTLY):
<answer>
{
  "gen_prompt": "subject_definitions:\\n...\\n\\nsummary:\\n...\\n\\nretention_analysis:\\n...\\n\\ndetailed_description:\\n...\\n\\noverall_soundscape:\\nN/A\\n\\nnon_diegetic_music:\\nN/A",
  "reference_images": [{"img_id": "IMG_###", "note": "..."}]
}
</answer>

Output your answer JSON NOW:
""".strip()

FORMAT_ERROR_OBS = ("Your previous reply was not used: {problem}. Reply again in the required format: write your "
                    "reasoning inside <think> ... </think> first, then exactly ONE <tool_call> ... </tool_call> or "
                    "<answer> ... </answer>, and stop. Do not write tool results yourself.")
IMAGE_SEARCH_CAP_OBS = (f"you have already used your {MAX_IMAGE_SEARCHES} image searches; choose from the images "
                        "you were shown and continue with search/browse or give the answer")
BROWSE_PROMPT = ("Please read the source content and answer a following question:\n"
                 "--- begin of source content ---\n{content}\n--- end of source content ---\n\n"
                 "If there is no relevant information, please clearly refuse to answer. "
                 "Now answer the question based on the above content:\n{query}")

ACTION_RE = re.compile(r"<tool_call>(.*?)</tool_call>|<answer>(.*?)</answer>", re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
TOOL_NAMES = ("search", "image_search", "browse")


def load_system_prompt(date: str | None = None) -> str:
    """The stored SFT system prompt + today's date on its own last line (as in sft_data.json)."""
    body = SYSTEM_PROMPT_PATH.read_text().rstrip("\n")
    return body + "\n" + (date or time.strftime("%Y-%m-%d"))


def parse_round(raw: str):
    """Gen-Searcher agent loop: act on the first <tool_call> or <answer>; drop hallucinated <tool_response>
    tails.  Returns (kind, payload, has_think) or a problem string."""
    raw = (raw or "").split("<tool_response>")[0]
    if not raw.strip():
        return "the reply was empty"
    m = ACTION_RE.search(raw)
    if not m:
        if re.search(r"<(tool_call|answer)>", raw):
            return "the reply was cut off before its closing tag; be MUCH more concise and keep <think> short"
        return "no <tool_call>...</tool_call> or <answer>...</answer> block"
    kind, body = ("tool_call", m.group(1)) if m.group(1) is not None else ("answer", m.group(2))
    try:
        payload = json.loads(body.strip())
    except json.JSONDecodeError as e:
        return f"the JSON inside <{kind}> does not parse ({e.msg})"
    if kind == "tool_call" and (not isinstance(payload, dict) or payload.get("name") not in TOOL_NAMES
                                or not isinstance(payload.get("arguments"), dict)):
        return 'a tool call must be {"name": "search"|"image_search"|"browse", "arguments": {...}}'
    if kind == "answer" and not isinstance(payload, dict):
        return "the answer must be one JSON object"
    has_think = any(t.group(1).strip() for t in THINK_RE.finditer(raw[:m.start()]))
    return kind, payload, has_think


# ---- tools (sync bodies; run in a thread so the rollout event loop is never blocked) ----

def _requests():
    import requests  # local import: keep the module importable without it
    return requests


SERPER_REQUESTS = [0]   # billed requests sent from this process (monitoring only; logged as adp/cost/serper_requests)


def _serper(url: str, query: str, num: int) -> dict:
    rq = _requests()
    for attempt in range(5):
        try:
            SERPER_REQUESTS[0] += 1
            r = rq.post(url, headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
                        json={"q": query, "num": num}, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(20, 2 ** attempt))
                continue
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            time.sleep(min(20, 2 ** attempt))
    return {}


def tool_search(args: dict) -> str:
    queries = args.get("queries") or args.get("query") or []
    if isinstance(queries, str):
        queries = [queries]
    blocks = []
    for q in queries:
        organic = _serper(SERPER_SEARCH_URL, str(q), SEARCH_RESULTS_PER_QUERY).get("organic") or []
        rows = [f"[{o.get('title', '')}]({o.get('link', '')}) {o.get('snippet', '')}"
                for o in organic[:SEARCH_RESULTS_PER_QUERY]]
        body = "\n\n".join(rows) if rows else "No results found."
        blocks.append(f"--- search result for [{q}] ---\n{body}\n--- end of search result ---")
    return "\n\n".join(blocks)


def _host_root(url: str) -> str:
    m = re.match(r"(https?://[^/]+)", url)
    return (m.group(1) + "/") if m else ""


def save_image(url: str, page_url: str, item_id: Any, cache_dir: Path) -> str | None:
    """Download with Gen-Searcher's headers, store as <id>-<md5(url)>.jpg, return './cached_image/<name>'."""
    from PIL import Image
    name = f"{item_id}-{hashlib.md5(url.encode('utf-8')).hexdigest()}.jpg"
    path = cache_dir / name
    if not path.exists():
        try:
            r = _requests().get(url, headers={**IMAGE_DOWNLOAD_HEADERS, "Referer": page_url or _host_root(url)},
                                timeout=30)
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content)).convert("RGB")
        except Exception:  # noqa: BLE001
            return None
        if min(im.size) < 64:
            return None
        cache_dir.mkdir(parents=True, exist_ok=True)
        im.save(path, "JPEG", quality=92)
    return f"./cached_image/{name}"


def tool_image_search(args: dict, item_id: Any, cache_dir: Path, registry: dict, counter: list[int]) -> tuple[str, list[dict]]:
    query = str(args.get("query", "")).strip()
    refs = []
    for im in _serper(SERPER_IMAGE_URL, query, 10).get("images") or []:
        if len(refs) >= MAX_IMAGES_FOR_MODEL:
            break
        url, page = im.get("imageUrl") or "", im.get("link") or ""
        local = save_image(url, page, item_id, cache_dir) if url else None
        if not local:
            continue
        counter[0] += 1
        ref = {"img_id": f"IMG_{counter[0]:03d}", "url": url, "local_path": local,
               "title": im.get("title") or "image", "page_url": page}
        registry[ref["img_id"]] = ref
        refs.append(ref)
    if not refs:
        return (f"[ImageSearch] No image results found for '{query}'. Your search query is too long or too "
                "specific. You MUST retry image_search using <= 3 words that describe only the main visual "
                'subject (e.g. "person name", "city skyline", "logo").'), []
    lines = [f"--- image search result for [{query}] ---"]
    for r in refs:
        lines += [f"{r['img_id']}: title: {r['title']}", f"  url: {r['url']}", f"  local_path: {r['local_path']}"]
        if r["page_url"]:
            lines.append(f"  page_url: {r['page_url']}")
    lines.append("--- end of image search result ---")
    return "\n".join(lines), refs


def _jina_read(url: str) -> str | None:
    rq = _requests()
    headers = {"Authorization": f"Bearer {JINA_KEY}"} if JINA_KEY else {}
    for attempt in range(3):
        try:
            resp = rq.get(JINA_READER_URL + url, headers=headers, timeout=60)
        except Exception:  # noqa: BLE001
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200 and resp.text.strip():
            return resp.text
        if resp.status_code == 429:
            time.sleep(4 + random.uniform(2, 4))
            continue
        return None
    return None


def call_chat(messages: list[dict], *, api_key: str, base_url: str, model: str, temperature: float = 0.0,
              max_tokens: int = 8192, timeout: float = 300, retries: int = 5) -> str:
    """Minimal OpenAI-compatible chat completion with retry/backoff."""
    rq = _requests()
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    last = None
    for attempt in range(retries):
        try:
            resp = rq.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                           json=payload, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                last = f"HTTP {resp.status_code}: {resp.text[:200]}"
                time.sleep(min(30, 2 ** attempt) + random.uniform(0, 1))
                continue
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"].get("content") or ""
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(30, 2 ** attempt) + random.uniform(0, 1))
    raise RuntimeError(f"chat call failed: {last}")


def tool_browse(args: dict) -> str:
    url, query = str(args.get("url", "")).strip(), str(args.get("query", "")).strip()
    source = _jina_read(url) if url.startswith("http") else None
    if not source or not source.strip():
        return "Browse error. Please try again."
    query = query or "Detailed summary of the page."

    def summarize(text: str) -> str:
        try:
            return call_chat([{"role": "user", "content": BROWSE_PROMPT.format(content=text, query=query)}],
                             api_key=BROWSE_API_KEY, base_url=BROWSE_BASE_URL, model=BROWSE_MODEL).strip()
        except Exception:  # noqa: BLE001
            return ""

    if len(source) <= BROWSE_CHUNK_CHARS:
        return summarize(source) or "Browse error. Please try again."
    n = len(source) // BROWSE_CHUNK_CHARS + 1
    step = len(source) // n
    parts = [summarize(source[i * step:(i + 1) * step + 4000]) for i in range(n)]
    return ("Since the content is too long, the result is split and answered separately. "
            "Please combine the results to get the complete answer.\n" +
            "\n".join(f"--- part {i + 1} ---\n{p}" for i, p in enumerate(parts) if p))


# ---- H3 prompt parser + answer checks (same rules as the SFT data) ----

H3_SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description",
               "overall_soundscape", "non_diegetic_music")
_SECTION_RE = re.compile(r"^(" + "|".join(H3_SECTIONS) + r"):[ \t]*", re.M)
_SHOT_RE = re.compile(r"\[Shot (\d+)\](?:\s+At (\d{2}):(\d{2})\.(\d{3}),?)?")


def parse_h3_prompt(gen_prompt: str, duration: float = VIDEO_DURATION_S) -> dict:
    text = gen_prompt if isinstance(gen_prompt, str) else ""
    heads = list(_SECTION_RE.finditer(text))
    sections = {m.group(1): text[m.end():(heads[i + 1].start() if i + 1 < len(heads) else len(text))].strip()
                for i, m in enumerate(heads)}
    order = [m.group(1) for m in heads]
    errors = []
    if order != list(H3_SECTIONS):
        missing = [h for h in H3_SECTIONS if h not in order]
        errors.append(f"sections missing {missing}" if missing else f"sections out of order {order}")
    desc = sections.get("detailed_description", "")
    marks = list(_SHOT_RE.finditer(desc))
    shots = []
    for i, m in enumerate(marks):
        n = int(m.group(1))
        start = (0.0 if i == 0 else None) if m.group(2) is None else \
            int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)) / 1000
        if n != i + 1:
            errors.append(f"[Shot {n}] out of sequence")
        if i == 0 and m.group(2) is not None:
            errors.append("[Shot 1] must not carry a timestamp")
        if i > 0 and m.group(2) is None:
            errors.append(f"[Shot {n}] has no 'At MM:SS.mmm' cut time")
        end_pos = marks[i + 1].start() if i + 1 < len(marks) else len(desc)
        shots.append({"shot_index": n, "start_s": start, "end_s": None, "description": desc[m.end():end_pos].strip()})
    for i, sh in enumerate(shots):
        sh["end_s"] = shots[i + 1]["start_s"] if i + 1 < len(shots) else float(duration)
        if sh["start_s"] is not None and sh["end_s"] is not None and sh["end_s"] <= sh["start_s"]:
            errors.append(f"[Shot {sh['shot_index']}] has no duration ({sh['start_s']}-{sh['end_s']} s)")
    pictures = sorted({int(n) for n in re.findall(r"<Picture (\d+)>", text)})
    return {"sections": sections, "order": order, "shots": shots, "pictures": pictures, "errors": errors}


def build_prediction(answer: dict, registry: dict) -> dict:
    refs = []
    for r in (answer.get("reference_images") or [])[:MAX_REF_IMAGES]:
        if not isinstance(r, dict):
            continue
        meta = registry.get(r.get("img_id"), {})
        refs.append({"img_id": r.get("img_id"), "url": meta.get("url"), "local_path": meta.get("local_path"),
                     "title": meta.get("title"), "page_url": meta.get("page_url"), "note": r.get("note", "")})
    return {"gen_prompt": answer.get("gen_prompt") or "", "reference_images": refs}


def gate_errors(prediction: dict | None, root: Path, image_search_calls: int | None = None) -> list[str]:
    """Reward = 0 cases.  G1-G3 are Gen-Searcher compute_reward(); P3/P5 are the H3-format rules the video needs
    - without them the prompt cannot be sent to H3 nor cut into shot clips."""
    errs = []
    if not prediction or not isinstance(prediction, dict) or "error" in prediction:
        return ["G1 prediction missing or has an error"]
    gen_prompt = str(prediction.get("gen_prompt") or "")
    if not gen_prompt.strip():
        return ["G2 empty gen_prompt"]
    refs = [r for r in prediction.get("reference_images") or [] if isinstance(r, dict)]
    valid = [r for r in refs if r.get("local_path") and (root / r["local_path"]).exists()]
    if not MIN_REF_IMAGES <= len(valid) <= MAX_REF_IMAGES:
        errs.append(f"G3 {len(valid)} reference images with a file (need {MIN_REF_IMAGES}-{MAX_REF_IMAGES})")
    if image_search_calls is not None and image_search_calls < 1:
        errs.append("P3 no image_search call")
    parsed = parse_h3_prompt(gen_prompt)
    errs += [f"P5 {e}" for e in parsed["errors"]]
    shots = parsed["shots"]
    if len(shots) < MIN_SHOTS:
        errs.append(f"P5 {len(shots)} shots (< {MIN_SHOTS})")
    if shots and shots[-1]["start_s"] is not None and shots[-1]["end_s"] - shots[-1]["start_s"] < MIN_LAST_SHOT_S:
        errs.append(f"P5 last shot shorter than {MIN_LAST_SHOT_S:g} s")
    for sec in ("overall_soundscape", "non_diegetic_music"):
        if sec in parsed["sections"] and parsed["sections"][sec].strip().rstrip(".") != "N/A":
            errs.append(f"P5 {sec} must be N/A (silent video)")
    if re.search(r"IMG_\d{3}", gen_prompt) or re.search(r"https?://", gen_prompt):
        errs.append("P5 gen_prompt contains IMG_### or a URL")
    if "<d>" in gen_prompt or re.search(r"\(S\d+\)", gen_prompt):
        errs.append("P5 gen_prompt contains dialogue or a speaker id")
    over = [n for n in parsed["pictures"] if n > len(refs)]
    if over:
        errs.append(f"P5 cites <Picture {over[0]}> but only {len(refs)} reference images chosen")
    unused = [n for n in range(1, len(refs) + 1) if n not in parsed["pictures"]]
    if unused:
        errs.append(f"P5 <Picture {unused[0]}> chosen but never cited in gen_prompt")
    return errs


# ---- the interaction environment ----

class AdProducerEnv:
    """One rollout's state.  `astep(response_text)` -> (observation | None, done, info).

    observation = {"text": str, "images": [{"img_id", "path"}]}; images are interleaved into the user
    message right after their 'IMG_###: ' line by `format_observation`, the position of <image> in sft_data.
    """

    def __init__(self, item_id: Any, label: dict, max_turns: int):
        self.item_id = item_id
        self.label = label
        self.max_turns = max_turns
        self.run_dir = WORK_DIR / "rollouts" / str(item_id) / uuid.uuid4().hex[:8]
        self.cache_dir = self.run_dir / "cached_image"
        self.registry: dict[str, dict] = {}
        self.counter = [0]
        self.turn = 0
        self.tool_calls = 0
        self.image_searches = 0
        self.forced_final = False
        self.prev_assistant: str | None = None
        self.prediction: dict | None = None
        self.termination = "max_turns"
        self.malformed_turns = 0
        self.tool_log: list[dict] = []

    # observations ---------------------------------------------------------------------------------
    @staticmethod
    def _tool_response(text: str) -> str:
        return "<tool_response>\n" + text + "\n</tool_response>"

    def format_observation(self, obs: dict) -> dict:
        """-> {"role": "user", "content": [text/image parts]} with each image after its 'IMG_###: ' line."""
        from PIL import Image
        text, parts, pos = obs["text"], [], 0
        for im in obs.get("images") or []:
            m = re.search(rf"^{re.escape(im['img_id'])}: ", text[pos:], re.M)
            if not m:
                continue
            cut = pos + m.end()
            parts.append({"type": "text", "text": text[pos:cut]})
            pil = Image.open(im["path"]).convert("RGB")
            if IMAGE_MAX_PIXELS > 0 and pil.width * pil.height > IMAGE_MAX_PIXELS:   # only the largest images
                scale = (IMAGE_MAX_PIXELS / (pil.width * pil.height)) ** 0.5
                pil = pil.resize((max(28, int(pil.width * scale)), max(28, int(pil.height * scale))))
            parts.append({"type": "image", "image": pil})
            pos = cut
        parts.append({"type": "text", "text": text[pos:]})
        return {"role": "user", "content": parts}

    # step -----------------------------------------------------------------------------------------
    async def astep(self, response_text: str):
        self.turn += 1
        info: dict[str, Any] = {}
        text = (response_text or "").strip()
        if self.prev_assistant is not None and text == self.prev_assistant:   # Gen-Searcher: identical repeat
            self.termination = "repeated_response"
            return None, True, {"termination": self.termination}
        self.prev_assistant = text

        res = parse_round(text)
        if isinstance(res, str):
            self.malformed_turns += 1
            if self.forced_final:
                self.termination = "answer_parse_failed"
                return None, True, {"termination": self.termination, "problem": res}
            return {"text": FORMAT_ERROR_OBS.format(problem=res), "images": []}, False, {"malformed": res}
        kind, payload, _has_think = res

        if kind == "answer":
            self.prediction = build_prediction(payload, self.registry)
            self.termination = "no_tool_call"
            return None, True, {"termination": self.termination}

        if self.forced_final:                               # tool call after FINAL_MESSAGE: ignored, episode over
            self.termination = "answer_parse_failed"
            return None, True, {"termination": self.termination, "problem": "tool call after final message"}

        name, args = payload["name"], payload["arguments"]
        if name == "image_search" and self.image_searches >= MAX_IMAGE_SEARCHES:
            self.malformed_turns += 1
            return {"text": FORMAT_ERROR_OBS.format(problem=IMAGE_SEARCH_CAP_OBS), "images": []}, False, {}

        t0 = time.time()
        images: list[dict] = []
        if name == "search":
            body = await asyncio.to_thread(tool_search, args)
        elif name == "browse":
            body = await asyncio.to_thread(tool_browse, args)
        else:
            body, shown = await asyncio.to_thread(tool_image_search, args, self.item_id, self.cache_dir,
                                                  self.registry, self.counter)
            self.image_searches += 1
            images = [{"img_id": r["img_id"], "path": str(self.run_dir / r["local_path"])} for r in shown]
        self.tool_calls += 1
        self.tool_log.append({"name": name, "arguments": args, "seconds": round(time.time() - t0, 1)})

        obs_text = self._tool_response(body)
        if self.tool_calls >= MAX_TOOL_CALLS:               # budget used up: one last chance to answer
            obs_text += "\n\n" + FINAL_MESSAGE
            self.forced_final = True
        return {"text": obs_text, "images": images}, False, info

    def close(self):
        return


# ============================================================================================
# === 2. slime integration ===================================================================
# ============================================================================================

def _label_of(sample) -> dict:
    lab = sample.label
    if isinstance(lab, str):
        try:
            return json.loads(lab)
        except json.JSONDecodeError:
            return {}
    return dict(lab or {})


async def generate(args, sample, sampling_params):
    """slime custom generate: multi-turn VLM rollout with awaited tool calls.

    Token/logprob/loss-mask/image bookkeeping reuses examples.geo3k_vlm_multi_turn.rollout; only the loop
    differs (async env step, per-turn generation cap GEN_MAX_NEW_TOKENS_PER_TURN, FINAL_MESSAGE semantics).
    """
    from examples.geo3k_vlm_multi_turn.rollout import (  # type: ignore
        GenerateState, _append_to_sample, _encode_observation_for_generation, _finalize_sample,
        _prepare_start_state, _run_inference_step, _should_stop_on_finish, _update_budget,
        _update_multimodal_state)
    from slime.utils.types import Sample  # type: ignore

    assert not args.partial_rollout, "partial rollout is not supported"
    max_turns = int(getattr(args, "max_turns", None) or MAX_TOOL_CALLS + 1)
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    label = _label_of(sample)
    env = AdProducerEnv(label.get("id", sample.index), label, max_turns)
    sample.metadata = sample.metadata or {}

    sampling_params = sampling_params.copy()
    current_image_data, response_tokens, budget, mm_buffer = _prepare_start_state(sample, state, args, sampling_params)
    try:
        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            return sample
        for turn_idx in range(max_turns):
            cur = dict(sampling_params)
            cur["max_new_tokens"] = min(GEN_MAX_NEW_TOKENS_PER_TURN, budget) if budget is not None else GEN_MAX_NEW_TOKENS_PER_TURN
            response_text, new_tokens, new_logprobs, finish_type, meta_info = await _run_inference_step(
                url, sample.tokens, cur, current_image_data, state.tokenizer)
            _append_to_sample(sample, response_tokens, new_tokens, new_logprobs, loss_mask_val=1, args=args, meta_info=meta_info)
            budget = _update_budget(budget, len(new_tokens))
            if _should_stop_on_finish(sample, finish_type):     # context length / abort
                break
            if finish_type == "length" or (budget is not None and budget <= 0):
                sample.status = Sample.Status.TRUNCATED
                break

            obs, done, _info = await env.astep(response_text)
            if done:
                sample.status = Sample.Status.COMPLETED
                break
            msg = env.format_observation(obs)
            obs_ids, obs_image_data, obs_mm_inputs, obs_mm_train = _encode_observation_for_generation(
                state.tokenizer, state.processor, msg, sample.metadata, args.apply_chat_template,
                args.apply_chat_template_kwargs)
            bos = state.tokenizer.bos_token_id
            if bos is not None and obs_ids and obs_ids[0] == bos:
                obs_ids = obs_ids[1:]
            # observation tokens are non-trainable: slime@540ea91 Sample.append_response_tokens rejects log probs for
            # them (it fills zeros itself); the geo3k example still passes [0.0]*n and fails with a ValueError.
            _append_to_sample(sample, response_tokens, obs_ids, None, loss_mask_val=0)
            budget = _update_budget(budget, len(obs_ids))
            current_image_data = _update_multimodal_state(sample, current_image_data, obs_image_data,
                                                          obs_mm_inputs, obs_mm_train, mm_buffer)
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            if turn_idx + 1 >= max_turns:
                env.termination = "max_turns"
                sample.status = Sample.Status.COMPLETED
                break
        sample.metadata.update({
            "prediction": env.prediction, "termination": env.termination, "tool_calls": env.tool_calls,
            "image_searches": env.image_searches, "malformed_turns": env.malformed_turns,
            "forced_final": env.forced_final, "run_dir": str(env.run_dir), "tool_log": env.tool_log,
        })
        return _finalize_sample(sample, state.tokenizer, response_tokens, mm_buffer)
    finally:
        env.close()


# ============================================================================================
# === 3. Reward ==============================================================================
# ============================================================================================
#
#   R = (1 - a) * R_video + a * R_text,  a = GEN_REWARD_TEXT_COEF = 0.5          (Gen-Searcher)
#
#   Failures follow Gen-Searcher (rllm agent_workflow_engine.py compact filtering + gen_image_deepresearch_reward.py):
#   reward 0, masked   truncated, repeated response, no answer (max turns), answer not parsable -> reward 0 and
#                      remove_sample: it still enters the group mean/std (slime default, like rllm which filters
#                      after compute_advantage) but gets no gradient itself
#   reward 0           gate (empty gen_prompt, 1-5 reference images, no image_search, H3 format), H3 failure
#   component 0        a judge that never returns valid JSON gives 0 for its own term only
#   group masked       every rollout of the prompt has reward <= 0 (mask_all_failed_episodes)
#   R_text        Appendix B prompt: brief + verified facts + official images + GT video + answer
#                 (gen_prompt, reference metadata AND the chosen images themselves)
#   R_video       prompts/reward_video.txt (RL reward): brief + verified facts + text requirement + <= 3 official
#                 images + generated video (whole, default sampling as in VideoGen-Agent, with shot time ranges
#                 only) + GT video (aesthetics bar); ADP_JUDGE_KEYFRAMES=1,3,5,7,9 adds full-resolution frames.  Five dimensions,
#                 mixed scale (VideoGen-Agent: fine scale on the core dimensions, coarse on the minor ones):
#                   visual_correctness 0.4, text_accuracy 0.2, selling_point 0.2   five levels {0,.25,.5,.75,1}
#                   shot_design 0.1, aesthetics 0.1                                three levels {0,.5,1}
#                 text_required is decided once per item from the brief alone (cli text-required, stored in the
#                 label by prepare-data); items without the flag fall back to the judge deciding it from the brief.
#                 When text is not required, its 0.2 is spread over the other four dimensions pro rata.
#                 The judge never sees the agent's gen_prompt, so the agent cannot make text "not required".
#   Benchmark     metric PG-Score = this R_video (same prompt, judge and inputs; Gen-Searcher also uses its K-Score
#                 as both RL reward and benchmark metric).

# --- Gen-Searcher Appendix B (text reward), adapted to video. ---
TEXT_REWARD_SYSTEM_PROMPT = (PROMPTS_DIR / "reward_text.txt").read_text().strip()

# --- RL video reward (five dimensions, see above). ---
VIDEO_REWARD_SYSTEM_PROMPT = (PROMPTS_DIR / "reward_video.txt").read_text().strip()
TEXT_REQUIRED_SYSTEM_PROMPT = (PROMPTS_DIR / "text_required.txt").read_text().strip()
FIVE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)
THREE_LEVELS = (0.0, 0.5, 1.0)
V_WEIGHTS = {"visual_correctness": 0.4, "text_accuracy": 0.2, "selling_point": 0.2, "shot_design": 0.1,
             "aesthetics": 0.1}
V_LEVELS = {"visual_correctness": FIVE_LEVELS, "text_accuracy": FIVE_LEVELS, "selling_point": FIVE_LEVELS,
            "shot_design": THREE_LEVELS, "aesthetics": THREE_LEVELS}
# Optional full-resolution frames for the video judge, in seconds (e.g. ADP_JUDGE_KEYFRAMES=1,3,5,7,9 gives every
# 3-4-shot video >= 1 frame per shot).  Off by default.
KEYFRAME_TIMES = tuple(int(t) for t in _env("ADP_JUDGE_KEYFRAMES", "").split(",") if t.strip())

# terminations that mean "tool budget used up / turns exhausted without an answer": trained with this reward, not
# masked (see score_sample).  repeated_response and truncation stay masked as in Gen-Searcher.
NO_ANSWER_TERMINATIONS = ("answer_parse_failed", "max_turns")
NO_ANSWER_REWARD = 0.0
TEXT_REWARD_VALID = (0.0, 0.25, 0.5, 0.75, 1.0)
OFFICIAL_IMAGES_FOR_JUDGE = 3




def video_score(v: dict, text_required: bool) -> float:
    """RL R_video: weighted sum of the five dimensions; without required text its weight is spread pro rata."""
    w = {k: x for k, x in V_WEIGHTS.items() if text_required or k != "text_accuracy"}
    return round(sum(x * float(v[k]) for k, x in w.items()) / sum(w.values()), 4)


def _as_bool(x) -> bool:
    return x in (True, 1, "true", "True", "yes", "YES")


def _snap(x, allowed=(0.0, 0.5, 1.0)) -> float:
    try:
        return min(allowed, key=lambda v: abs(v - float(x)))
    except (TypeError, ValueError):
        return 0.0


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", (text or "").strip(), flags=re.S)
    if not m:
        raise ValueError("no JSON object in judge reply")
    return json.loads(m.group(0))


def answer_text(prediction: dict) -> str:
    """Gen-Searcher _format_prediction_as_answer_text: gen_prompt + reference image metadata."""
    parts = ["gen_prompt:", prediction.get("gen_prompt", ""), "", "reference_images:"]
    for i, r in enumerate(prediction.get("reference_images") or [], 1):
        parts.append(f"<Picture {i}> img_id={r.get('img_id')} title={r.get('title')!r} note={r.get('note')!r} "
                     f"page_url={r.get('page_url')}")
    return "\n".join(parts)


def facts_block(label: dict) -> str:
    lines = [f"- {f['slot']}: {f['answer']}" for f in label.get("gt_facts") or []]
    return "\n".join(lines) or "(none)"


def shot_times(gen_prompt: str) -> str:
    """Shot indices and time ranges only: enough to locate the shots (H3 follows the planned cut times), without showing the judge anything the agent wrote."""
    return "\n".join(f"[Shot {s['shot_index']}] {s['start_s']:.1f}s-{s['end_s']:.1f}s"
                     for s in parse_h3_prompt(gen_prompt)["shots"])


def official_images(label: dict) -> list[Path]:
    paths = [DATA_DIR / p for p in label.get("official_images") or [] if p]
    return [p for p in paths if p.exists()][:OFFICIAL_IMAGES_FOR_JUDGE]


# ---- media helpers (ffmpeg) ----

def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def strip_audio(src: Path, dst: Path) -> None:
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-an", "-c:v", "copy", str(dst)])


def extract_keyframes(video: Path, out_dir: Path) -> list[tuple[int, Path]]:
    """Full-resolution JPEG frames at KEYFRAME_TIMES (H3 renders 864x480); frames that fail are skipped."""
    if not KEYFRAME_TIMES:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in KEYFRAME_TIMES:
        p = out_dir / f"frame_{t:02d}s.jpg"
        p.unlink(missing_ok=True)
        _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-ss", str(t), "-i", str(video),
              "-frames:v", "1", "-q:v", "2", str(p)], timeout=120)
        if p.exists() and p.stat().st_size > 0:
            frames.append((t, p))
    return frames


# ---- H3 client ----

def _data_uri(path: Path) -> str:
    """data:image/...;base64 URI (MiniMax H3 material_io accepts data/base64/http/file URIs)."""
    raw = Path(path).read_bytes()
    mime = ("image/png" if raw[:8] == b"\x89PNG\r\n\x1a\n" else "image/webp" if raw[8:12] == b"WEBP"
            else "image/gif" if raw[:4] == b"GIF8" else "image/jpeg")
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


class H3Client:
    """MiniMax H3 Ref2VA via the sglang /v1/videos API.  Servers come from H3_URLS or, dynamically, from H3_URLS_DIR;
    each request goes to the least-loaded healthy server (each server runs one generation at a time)."""

    def __init__(self):
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._load: dict[str, int] = {}
        self._bad: dict[str, float] = {}
        self._cache: tuple[float, list[str]] = (0.0, [])
        if not (H3_URLS or H3_URLS_DIR):
            raise RuntimeError("no H3 server: set H3_URLS or H3_URLS_DIR")

    def urls(self) -> list[str]:
        if H3_URLS_DIR:
            now = time.time()
            if now - self._cache[0] > 15:
                found = []
                for f in sorted(Path(H3_URLS_DIR).glob("*.url")):
                    try:
                        u = f.read_text().strip()
                    except OSError:
                        continue
                    if u:
                        found.append(u)
                self._cache = (now, found)
            urls = list(self._cache[1])
        else:
            urls = list(H3_URLS)
        for u in urls:
            self._sems.setdefault(u, asyncio.Semaphore(H3_CONCURRENCY_PER_SERVER))
            self._load.setdefault(u, 0)
        return urls

    async def _acquire(self) -> str:
        """Wait until some healthy server has a free slot and claim it (load += 1).  Requests are bound to a server
        only when it can start them, so servers that join later get work right away; no await between the check and
        the claim, so concurrent callers cannot over-book a server."""
        empty_since = None
        while True:
            urls = self.urls()
            if urls:
                empty_since = None
                now = time.time()
                good = [u for u in urls if self._bad.get(u, 0) < now] or urls
                free = [u for u in good if self._load[u] < H3_CONCURRENCY_PER_SERVER]
                if free:
                    u = min(free, key=lambda x: self._load[x])
                    self._load[u] += 1
                    return u
            else:
                empty_since = empty_since or time.time()
                if time.time() - empty_since > 600:
                    raise RuntimeError("no H3 server registered for 10 min")
            await asyncio.sleep(2)

    async def generate(self, gen_prompt: str, image_paths: list[Path], out_mp4: Path) -> None:
        import aiohttp
        req = {"task": "ref2va", "prompt": gen_prompt,
               # images go inline as data URIs, so H3 servers on other machines need no shared file system
               "conditions": [{"type": "image", "uri": _data_uri(p), "role": "reference"} for p in image_paths],
               "target": {"short_edge": H3_SHORT_EDGE, "aspect_ratio": "16:9", "duration_seconds": VIDEO_DURATION_S},
               # same sampling settings as the GT videos, so R_video's GT comparison is like for like
               "seconds": VIDEO_DURATION_S, "num_outputs_per_prompt": 1, "num_inference_steps": H3_STEPS,
               "flow_shift": H3_FLOW_SHIFT, "audio_flow_shift": H3_AUDIO_FLOW_SHIFT, "seed": H3_SEED}
        last = None
        for attempt in range(H3_RETRIES):
            base = await self._acquire()
            try:
                async with self._sems[base]:
                    try:
                        timeout = aiohttp.ClientTimeout(total=H3_TIMEOUT_S)
                        async with aiohttp.ClientSession(timeout=timeout) as s:
                            async with s.post(f"{base}/v1/videos", json=req) as r:
                                r.raise_for_status()
                                vid = (await r.json())["id"]
                            t0 = time.time()
                            while True:
                                async with s.get(f"{base}/v1/videos/{vid}") as r:
                                    st = (await r.json()).get("status", "")
                                if st == "completed":
                                    break
                                if st in ("failed", "cancelled", "error"):
                                    raise RuntimeError(f"H3 status {st}")
                                if time.time() - t0 > H3_TIMEOUT_S:
                                    raise TimeoutError("H3 generation timed out")
                                await asyncio.sleep(5)
                            raw = out_mp4.with_suffix(".raw.mp4")
                            async with s.get(f"{base}/v1/videos/{vid}/content") as r:
                                r.raise_for_status()
                                raw.write_bytes(await r.read())
                        await asyncio.to_thread(strip_audio, raw, out_mp4)
                        raw.unlink(missing_ok=True)
                        if out_mp4.exists() and out_mp4.stat().st_size > 0:
                            return
                        raise RuntimeError("empty video")
                    except Exception as e:  # noqa: BLE001
                        last = e
                        if isinstance(e, (aiohttp.ClientConnectionError, asyncio.TimeoutError)):
                            self._bad[base] = time.time() + H3_BAD_COOLDOWN_S
            finally:
                self._load[base] -= 1
            await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"H3 failed after {H3_RETRIES} attempts: {last}")


# ---- judge client ----

class Judge:
    """Gemini 3.1 Pro as the judge; the whole video is given to it directly with the model's default sampling, as in
    VideoGen-Agent (App. D.1).  Two ways to reach it:
      gemini  Google REST generateContent (key from AI Studio)
      openai  any OpenAI-compatible gateway that forwards video to Gemini
              (JUDGE_BASE_URL=<gateway>/v1, JUDGE_AUTH_HEADER=<header name>,
              JUDGE_MODEL=<provider>/gemini-3.1-pro-preview); videos go in as data:video/mp4 image_url parts."""

    def __init__(self):
        self.sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
        self.usage = {"calls": 0, "prompt_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "thought_tokens": 0}

    # parts: list of ("text", str) | ("image", Path) | ("video", Path)
    def _call_gemini(self, system: str, parts: list[tuple[str, Any]]) -> str:
        """Gemini REST generateContent (no SDK needed); images and videos inline."""
        req_parts = []
        for kind, v in parts:
            if kind == "text":
                req_parts.append({"text": v})
            else:
                mime = "image/jpeg" if kind == "image" else "video/mp4"
                req_parts.append({"inlineData": {"mimeType": mime, "data": base64.b64encode(Path(v).read_bytes()).decode()}})
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": req_parts}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": JUDGE_MAX_TOKENS,
                                     "responseMimeType": "application/json"}}
        url = f"{JUDGE_BASE_URL or 'https://generativelanguage.googleapis.com/v1beta'}/models/{JUDGE_MODEL}:generateContent"
        r = _requests().post(url, headers={"x-goog-api-key": JUDGE_API_KEY, "Content-Type": "application/json"},
                             json=body, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        u = data.get("usageMetadata") or {}
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += u.get("promptTokenCount", 0)
        self.usage["cached_tokens"] += u.get("cachedContentTokenCount", 0) or 0
        self.usage["output_tokens"] += u.get("candidatesTokenCount", 0)
        self.usage["thought_tokens"] += u.get("thoughtsTokenCount", 0)
        cand = (data.get("candidates") or [{}])[0]
        text = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts") or [] if not p.get("thought"))
        if not text and cand.get("finishReason"):
            raise RuntimeError(f"Gemini returned no text (finishReason={cand.get('finishReason')})")
        return text

    def _call_openai(self, system: str, parts: list[tuple[str, Any]]) -> str:
        """OpenAI-compatible /chat/completions; images and videos as base64 data URLs."""
        content = []
        for kind, v in parts:
            if kind == "text":
                content.append({"type": "text", "text": v})
            else:
                mime = "image/jpeg" if kind == "image" else "video/mp4"
                b64 = base64.b64encode(Path(v).read_bytes()).decode()
                content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        auth = ({JUDGE_AUTH_HEADER: JUDGE_API_KEY} if JUDGE_AUTH_HEADER.lower() != "authorization"
                else {"Authorization": f"Bearer {JUDGE_API_KEY}"})
        r = _requests().post(JUDGE_BASE_URL.rstrip("/") + "/chat/completions",
                             headers={**auth, "Content-Type": "application/json"}, timeout=600,
                             json={"model": JUDGE_MODEL, "temperature": 0.0, "max_tokens": JUDGE_MAX_TOKENS,
                                   "response_format": {"type": "json_object"},     # JSON only, nothing around it
                                   "messages": [{"role": "system", "content": system},
                                                {"role": "user", "content": content}]})
        if r.status_code != 200:
            raise RuntimeError(f"judge HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        u = data.get("usage") or {}
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        self.usage["cached_tokens"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
        self.usage["output_tokens"] += u.get("completion_tokens", 0)
        self.usage["thought_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        return (data["choices"][0]["message"].get("content") or "")

    async def score(self, system: str, parts: list[tuple[str, Any]], required: tuple[str, ...]) -> dict | None:
        """Up to JUDGE_RETRIES attempts; None if every reply is invalid (the caller scores that term 0)."""
        fn = self._call_gemini if JUDGE_BACKEND == "gemini" else self._call_openai
        for _ in range(JUDGE_RETRIES):
            async with self.sem:
                try:
                    out = _extract_json(await asyncio.to_thread(fn, system, parts))
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(1)
                    continue
            if all(k in out for k in required):
                return out
        return None


def _reference_parts(label: dict, gt_video: Path) -> list[tuple[str, Any]]:
    parts: list[tuple[str, Any]] = []
    for i, p in enumerate(official_images(label), 1):
        parts += [("text", f"OFFICIAL IMAGE {i} (true identity of the object):"), ("image", p)]
    if gt_video.exists():
        parts += [("text", "GT VIDEO (ground-truth reference video for this task):"), ("video", gt_video)]
    return parts


# ---- the two rewards ----

async def text_reward(judge: Judge, label: dict, prediction: dict, run_dir: Path, gt_video: Path) -> dict | None:
    parts: list[tuple[str, Any]] = [("text",
        f"Sample id: {label.get('id')}\n\nTask prompt (what video we want to generate):\n{label.get('brief', '')}\n\n"
        f"Verified facts (ground truth):\n{facts_block(label)}\n")]
    parts += _reference_parts(label, gt_video)
    parts.append(("text", f"\nModel's <answer> content (gen_prompt + reference_images):\n{answer_text(prediction)}\n\n"
                          "The chosen reference images follow, in order:"))
    for i, r in enumerate(prediction.get("reference_images") or [], 1):
        p = run_dir / r["local_path"] if r.get("local_path") else None
        if p and p.exists():
            parts += [("text", f"<Picture {i}> ({r.get('img_id')}):"), ("image", p)]
    parts.append(("text", "\nFrom both text and visual angles, score whether this answer is sufficient for a correct and "
                          "effective video for the task, applying the hard constraints and score caps "
                          "(0, 0.25, 0.5, 0.75, or 1.0). Output JSON with 'score' and 'rationale'."))
    out = await judge.score(TEXT_REWARD_SYSTEM_PROMPT, parts, ("score",))
    if out is not None:
        out["score"] = _snap(out.get("score"), TEXT_REWARD_VALID)
    return out


async def video_reward(judge: Judge, label: dict, prediction: dict, video: Path, gt_video: Path,
                       frames_dir: Path) -> dict | None:
    """RL R_video (prompts/reward_video.txt).  Returns the judge JSON plus 'video_score', or None."""
    tr = label.get("text_required")
    req = ("decide (decide it yourself from the brief alone)" if tr is None else
           "YES - the brief requires readable text" if tr else "NO - the brief does not require readable text")
    parts: list[tuple[str, Any]] = [("text",
        f"Sample id: {label.get('id')}\n\nTask prompt (video requirement):\nBrief: {label.get('brief', '')}\n\n"
        f"Verified facts:\n{facts_block(label)}\n\nText requirement: {req}\n")]
    # Inputs shared by all rollouts of this prompt come first (brief, facts, official images, GT video) so that the
    # judge backend can reuse them as a cached prefix; the rollout's own video and frames come last.
    for i, p in enumerate(official_images(label), 1):
        parts += [("text", f"OFFICIAL IMAGE {i} (true identity of the object):"), ("image", p)]
    if gt_video.exists():
        parts += [("text", "GT VIDEO (reference implementation; the quality bar for aesthetics):"), ("video", gt_video)]
    parts += [("text", f"GENERATED VIDEO (to be evaluated). Shot time ranges:\n{shot_times(prediction['gen_prompt'])}"),
              ("video", video)]
    frames = await asyncio.to_thread(extract_keyframes, video, frames_dir)
    if frames:
        parts.append(("text", "FULL-RESOLUTION FRAMES of the generated video:"))
        for t, p in frames:
            parts += [("text", f"Frame at {t} s:"), ("image", p)]
    parts.append(("text", "Output a single JSON object with rationale, text_required and the five scores."))
    out = await judge.score(VIDEO_REWARD_SYSTEM_PROMPT, parts, tuple(V_WEIGHTS) + ("text_required",))
    if out is None:
        return None
    text_req = bool(tr) if tr is not None else _as_bool(out.get("text_required"))
    for k in V_WEIGHTS:
        out[k] = None if (k == "text_accuracy" and not text_req) else _snap(out.get(k), V_LEVELS[k])
    out["text_required"] = text_req
    out["text_required_source"] = "precomputed" if tr is not None else "judge"
    out["n_frames"] = len(frames)
    out["video_score"] = video_score(out, text_req)
    return out


async def decide_text_required(judge: Judge, items: list[tuple[Any, str]]) -> dict[str, dict]:
    """From the brief alone, whether readable text is required; several briefs per text-only call (cheap).
    items: [(id, brief)] -> {str(id): {"text_required": bool, "required_text": [...]}} (ids missing on failure)."""
    body = "\n\n".join(f"### id={i}\n{brief}" for i, brief in items)
    out = await judge.score(TEXT_REQUIRED_SYSTEM_PROMPT, [("text", body)], ("results",))
    res = {}
    for r in (out or {}).get("results") or []:
        if isinstance(r, dict) and "id" in r and "text_required" in r:
            res[str(r["id"])] = {"text_required": _as_bool(r["text_required"]),
                                 "required_text": r.get("required_text") or []}
    want = {str(i) for i, _ in items}
    return {k: v for k, v in res.items() if k in want}


async def score_sample(sample, judge: Judge, h3: H3Client | None) -> dict:
    """Full reward for one rollout -> {"reward", "text_reward", "video_reward", "gate", ...}."""
    from slime.utils.types import Sample  # type: ignore
    md = sample.metadata or {}
    label = _label_of(sample)
    run_dir = Path(md.get("run_dir") or WORK_DIR / "rollouts" / "unknown" / uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"id": label.get("id"), "termination": md.get("termination"),
                              "tool_calls": md.get("tool_calls"), "image_searches": md.get("image_searches"),
                              "malformed_turns": md.get("malformed_turns"), "status": str(sample.status),
                              "reward": 0.0, "text_reward": None, "video_reward": None, "masked": False}

    # Gen-Searcher compact filtering (mask_max_response_length_exceeded, mask_repeated_response, mask_unknown):
    # reward 0, kept in the group statistics, no gradient of its own.  Used for failures that are not a tool-budget
    # decision of the policy: overlong (truncated), identical repeated reply, unknown termination.
    term = md.get("termination")
    if sample.status == Sample.Status.TRUNCATED or term not in ("no_tool_call",) + NO_ANSWER_TERMINATIONS:
        sample.remove_sample = True
        result.update(masked=True, gate=[f"masked: status={sample.status} termination={term}"])
        return result
    # Over the tool budget without a usable answer (kept calling tools after FINAL_MESSAGE, an unparsable reply after
    # it, or ran out of turns): reward 0 that DOES train, so the policy is penalised for blind tool calling instead of
    # the rollout being silently dropped.  Deviation from Gen-Searcher (which masks these as mask_unknown).
    if term in NO_ANSWER_TERMINATIONS:
        result.update(reward=NO_ANSWER_REWARD, gate=[f"no answer within the tool budget: termination={term}"])
        return result

    prediction = md.get("prediction")
    errs = gate_errors(prediction, run_dir, md.get("image_searches"))
    result["gate"] = errs
    if errs:                                    # compute_reward's zero cases (+ H3 format)
        return result

    gt_video = DATA_DIR / label.get("gt_video", "")
    refs = [run_dir / r["local_path"] for r in prediction["reference_images"] if r.get("local_path")]
    out_mp4 = run_dir / "generated.mp4"

    use_video = TEXT_REWARD_COEF < 1.0
    if use_video:
        # Generate first: a failed generation makes the whole reward 0 (Gen-Searcher), so no judge call is spent on
        # it.  The two judges then run in parallel, so the latency is the same as judging text alongside H3.
        try:
            await h3.generate(prediction["gen_prompt"], refs, out_mp4)
        except Exception as e:  # noqa: BLE001
            result["gate"] = [f"h3_failed: {e}"]
            return result
    t_obj, v_obj = await asyncio.gather(
        text_reward(judge, label, prediction, run_dir, gt_video),
        video_reward(judge, label, prediction, out_mp4, gt_video, run_dir / "frames") if use_video
        else asyncio.sleep(0, result=None))
    # Gen-Searcher: a judge that fails gives 0 for its own term
    result["text_reward"], result["text_judge"] = (t_obj["score"], t_obj) if t_obj else (0.0, None)
    result["video_reward"], result["video_judge"] = (v_obj["video_score"], v_obj) if v_obj else (0.0, None)
    if t_obj is None:
        result["gate"].append("text judge failed -> text_reward 0")
    if use_video and v_obj is None:
        result["gate"].append("video judge failed -> video_reward 0")
    result["reward"] = round((1 - TEXT_REWARD_COEF) * result["video_reward"] + TEXT_REWARD_COEF * result["text_reward"], 4)
    return result


_JUDGE: Judge | None = None
_H3: H3Client | None = None


async def reward_func(args, samples, **kwargs) -> list[float]:
    """slime batched custom RM (--custom-rm-path adproducer_rl.reward_func, with --group-rm)."""
    global _JUDGE, _H3
    _JUDGE = _JUDGE or Judge()
    _H3 = _H3 or (H3Client() if TEXT_REWARD_COEF < 1.0 else None)
    results = await asyncio.gather(*(score_sample(s, _JUDGE, _H3) for s in samples))
    # Gen-Searcher mask_all_failed_episodes: if every rollout of this prompt scored <= 0, the group gets no gradient.
    # With --group-rm, `samples` is exactly one prompt's group.
    if results and max(r["reward"] for r in results) <= 0:
        for s, r in zip(samples, results):
            s.remove_sample = True
            r["masked"] = True
            r["gate"] = (r.get("gate") or []) + ["group masked: every rollout scored <= 0"]
    rewards = []
    for s, r in zip(samples, results):
        s.metadata = s.metadata or {}
        s.metadata["reward_detail"] = r
        run_dir = s.metadata.get("run_dir")
        if run_dir:
            try:
                (Path(run_dir) / "result.json").write_text(json.dumps(
                    {**r, "prediction": s.metadata.get("prediction"), "tool_log": s.metadata.get("tool_log")},
                    ensure_ascii=False, indent=1))
            except OSError:
                pass
        rewards.append(float(r["reward"]))
    return rewards


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """slime --custom-rollout-log-function-path: task metrics under adp/* (wandb / tensorboard).
    Returns False so slime's default rollout/* and perf/* metrics are still logged.
    Note: tool calls and reward_func run in the rollout process, so the cost counters below are per run
    (cumulative) only when this function runs in that same process (slime's RolloutManager)."""
    try:
        from slime.observability import logging_utils  # type: ignore
        from slime.observability.metric_utils import compute_rollout_step  # type: ignore
    except Exception:  # noqa: BLE001
        return False
    flat = []
    for s in samples:
        flat += s if isinstance(s, list) else [s]
    det = [(s.metadata or {}).get("reward_detail") for s in flat]
    det = [d for d in det if d]
    if not det:
        return False
    n = len(det)
    mean = lambda xs: float(sum(xs) / len(xs)) if xs else 0.0   # noqa: E731
    unmasked = [d for d in det if not d.get("masked")]
    m = {
        "adp/reward_mean": mean([d["reward"] for d in det]),
        "adp/reward_pos_frac": mean([d["reward"] > 0 for d in det]),
        "adp/text_reward_mean": mean([d["text_reward"] for d in det if d.get("text_reward") is not None]),
        "adp/video_reward_mean": mean([d["video_reward"] for d in det if d.get("video_reward") is not None]),
        "adp/masked_frac": mean([bool(d.get("masked")) for d in det]),
        "adp/group_masked_frac": mean([any("group masked" in g for g in d.get("gate") or []) for d in det]),
        "adp/gate_fail_frac": mean([bool(d.get("gate")) and d["reward"] == 0 for d in unmasked]),
        "adp/no_answer_penalised_frac": mean([d.get("termination") in NO_ANSWER_TERMINATIONS and not d.get("masked")
                                             for d in det]),
        "adp/h3_fail_frac": mean([any(str(g).startswith("h3_failed") for g in d.get("gate") or []) for d in det]),
        "adp/text_judge_fail_frac": mean([any("text judge failed" in g for g in d.get("gate") or []) for d in det]),
        "adp/video_judge_fail_frac": mean([any("video judge failed" in g for g in d.get("gate") or []) for d in det]),
        "adp/tool_calls_mean": mean([d.get("tool_calls") or 0 for d in det]),
        "adp/tool_calls_max_frac": mean([(d.get("tool_calls") or 0) >= MAX_TOOL_CALLS for d in det]),
        "adp/image_searches_mean": mean([d.get("image_searches") or 0 for d in det]),
        "adp/no_image_search_frac": mean([(d.get("image_searches") or 0) == 0 for d in det]),
        "adp/malformed_turns_mean": mean([d.get("malformed_turns") or 0 for d in det]),
        "adp/n_samples": n,
        "adp/cost/serper_requests": SERPER_REQUESTS[0],
    }
    # per-dimension means of the video judge (text_accuracy only over items that require text)
    vj = [d["video_judge"] for d in det if d.get("video_judge")]
    for k in V_WEIGHTS:
        xs = [v[k] for v in vj if v.get(k) is not None]
        if xs:
            m[f"adp/video/{k}"] = mean(xs)
    if vj:
        m["adp/video/text_required_frac"] = mean([bool(v.get("text_required")) for v in vj])
    # GRPO signal: fraction of prompt groups whose rewards are all equal (zero advantage, no learning signal)
    groups: dict[Any, list[float]] = {}
    for s, d in ((s, (s.metadata or {}).get("reward_detail")) for s in flat):
        if d:
            groups.setdefault(s.group_index if s.group_index is not None else d.get("id"), []).append(d["reward"])
    if groups:
        m["adp/zero_variance_group_frac"] = mean([max(r) - min(r) < 1e-6 for r in groups.values()])
    # agreement of the two judges
    pairs = [(d["text_reward"], d["video_reward"]) for d in det
             if d.get("text_reward") is not None and d.get("video_reward") is not None]
    if len(pairs) >= 3:
        tx, vx = zip(*pairs)
        mt, mv = mean(tx), mean(vx)
        cov = sum((a - mt) * (b - mv) for a, b in pairs)
        sd = (sum((a - mt) ** 2 for a in tx) * sum((b - mv) ** 2 for b in vx)) ** 0.5
        if sd > 0:
            m["adp/text_video_corr"] = cov / sd
    for t in ("no_tool_call", "max_turns", "repeated_response", "answer_parse_failed"):
        m[f"adp/termination/{t}"] = mean([d.get("termination") == t for d in det])
    if _JUDGE is not None:
        for k, v in _JUDGE.usage.items():
            m[f"adp/cost/judge_{k}"] = v
    m["rollout/step"] = compute_rollout_step(args, rollout_id)
    print(f"[adp metrics] rollout {rollout_id}: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                              for k, v in m.items()}), flush=True)
    logging_utils.log(args, m, step_key="rollout/step")
    return False


# ============================================================================================
# === 4. CLIs ================================================================================
# ============================================================================================

def cli_prepare_data(a):
    """rl_data.json / bench_data.json -> jsonl for slime (--input-key prompt --label-key label)."""
    rl = json.load(open(a.rl_data))
    official = {}
    if a.official_json:   # data/official_images.json: {id: [./cached_image/...]} (= construction.reference_images)
        official = {int(k): v for k, v in json.load(open(a.official_json)).items()}
    elif a.all_data:
        for x in json.load(open(a.all_data)):
            official[x["id"]] = [im.get("local_path") for im in (x.get("construction") or {}).get("reference_images") or []
                                 if im.get("local_path")]
    exclude = set()
    if a.exclude:
        txt = Path(a.exclude).read_text()
        exclude = {int(t) for t in re.findall(r"\d+", txt)}
    rl = [x for x in rl if x["id"] not in exclude]
    text_req = {}
    if a.text_required:   # cli text-required output: {id: {"text_required": bool, ...}}
        text_req = {int(k): bool(v["text_required"]) for k, v in json.load(open(a.text_required)).items()}
    system = load_system_prompt(a.date)
    with open(a.out, "w") as f:
        for x in rl:
            label = {"id": x["id"], "brief": x["prompt"], "gt_video": x.get("gt_video"), "gt_facts": x.get("gt_facts") or [],
                     "meta": x.get("meta") or {}, "official_images": official.get(x["id"], [])}
            if x["id"] in text_req:
                label["text_required"] = text_req[x["id"]]
            row = {"prompt": [{"role": "system", "content": system}, {"role": "user", "content": x["prompt"]}],
                   "label": json.dumps(label, ensure_ascii=False), "metadata": x.get("meta") or {}}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rl)} rows -> {a.out}  (excluded {len(exclude)}; text_required for "
          f"{sum(x['id'] in text_req for x in rl)}; system prompt dated {system.splitlines()[-1]})")


def cli_text_required(a):
    """Decide once per item, from the brief alone, whether readable text is required (text-only judge calls).
    Resumable: ids already in --out are skipped.  Feed the file to prepare-data --text-required."""
    rl = json.load(open(a.rl_data))
    if a.ids:
        want = {int(t) for t in re.findall(r"\d+", a.ids)}
        rl = [x for x in rl if x["id"] in want]
    done = json.load(open(a.out)) if Path(a.out).exists() else {}
    todo = [(x["id"], x["prompt"]) for x in rl if str(x["id"]) not in done]
    batches = [todo[k:k + a.batch] for k in range(0, len(todo), a.batch)]
    judge = Judge()

    async def run():
        for r in await asyncio.gather(*(decide_text_required(judge, bt) for bt in batches)):
            done.update(r)
    asyncio.run(run())
    Path(a.out).write_text(json.dumps(done, ensure_ascii=False, indent=1))
    n_req = sum(v["text_required"] for v in done.values())
    print(f"{len(done)} items decided ({len(todo)} asked in {len(batches)} calls, "
          f"{sum(str(i) not in done for i, _ in todo)} missing -> rerun to retry); text required: {n_req}/{len(done)}; "
          f"judge usage {judge.usage}")








def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare-data"); p.add_argument("--rl-data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--all-data", default=None); p.add_argument("--date", default=None)
    p.add_argument("--official-json", default=None, help="data/official_images.json; replaces --all-data for this field")
    p.add_argument("--exclude", default=None, help="file with item ids to leave out")
    p.add_argument("--text-required", default=None, help="output of `text-required` ({id: {text_required: bool}})")
    p.set_defaults(fn=cli_prepare_data)
    p = sub.add_parser("text-required"); p.add_argument("--rl-data", required=True); p.add_argument("--out", required=True)
    p.add_argument("--ids", default=None, help="only these ids (comma/space separated)")
    p.add_argument("--batch", type=int, default=25, help="briefs per judge call")
    p.set_defaults(fn=cli_text_required)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
