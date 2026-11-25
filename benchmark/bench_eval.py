"""AdCreation-Bench (305 briefs) evaluation (Gen-Searcher KnowGen protocol, video version).  Reuses adproducer_rl.py for the
agent loop, the tools, the H3 client and the judge client (the RL file is only imported, never modified).

  gen-gt    GT-Video/gen_prompt.json -> <bench>/gt_video/answer_<id>.mp4 (H3 Ref2VA, same settings as training)
  text-required  once per benchmark: from the brief alone, is readable text required -> <bench>/text_required.json
  infer     the agent under test (OpenAI-compatible endpoint, e.g. vLLM) answers every brief with the three tools
            -> <out>/predictions.jsonl        (--baseline brief: no agent, the brief itself is the prompt)
  generate  H3 video for every successful prediction -> <out>/videos/<id>.mp4
            (agent: Ref2VA with its reference images; --baseline brief: H3 t2va on an fl2va server, H3_T2V_URLS)
  score     PG-Score (Product-Grounding Score) by the Gemini 3.1 Pro judge = the RL video reward R_video
            (prompts/reward_video.txt, adp.video_reward; as Gen-Searcher, whose K-Score is both its RL image reward
            and its benchmark metric)                                                -> <out>/scores.jsonl
  summary   mean over the scored items (failures skipped, as gpt_eval_knowgen.py), by freshness and by category

Every step resumes: items already in the output are skipped.  Configuration: the same .env as training.
    python benchmark/bench_eval.py <step> --bench-dir /path/to/AdCreation-Bench [--out-dir runs/<method>] ...
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DATE = "2026-10-20"    # date line of the agent system prompt in the benchmark runs
TERMINATION_OK = "no_tool_call"

adp = None                   # adproducer_rl, imported after ADP_DATA_DIR / ADP_WORK_DIR are set


def _import_vsr(bench_dir: Path, out_dir: Path | None):
    """adproducer_rl is only imported (never modified): same agent loop, tools, H3 Ref2VA client and judge."""
    global adp
    os.environ["ADP_DATA_DIR"] = str(bench_dir.resolve())
    if out_dir is not None:
        os.environ["ADP_WORK_DIR"] = str(out_dir.resolve())
    sys.path.insert(0, str(HERE.parent / "inference" / "adproducer" / "core"))   # the evaluated agent code
    import adproducer_rl as m
    adp = m
    shot_times = m.shot_times                       # brief baseline has no planned shots: say so to the judge
    def _shot_times(gen_prompt):                    # local patch: shots without parseable times -> same as no shots
        try:
            return shot_times(gen_prompt) or "(not available)"
        except (TypeError, ValueError, KeyError):
            return "(not available)"
    m.shot_times = _shot_times
    for k in ("V_WEIGHTS", "video_reward", "decide_text_required", "H3_STEPS"):
        if not hasattr(m, k):
            raise SystemExit(f"adproducer_rl.py is older than this script (no {k}); use the eval_code/ copy")


def load_labels(bench_dir: Path) -> dict[int, dict]:
    """Benchmark items: brief, verified facts, official images (the judge uses the first 3), text requirement, GT video.
    Reads AdCreation-Bench.json."""
    released = bench_dir / "AdCreation-Bench.json"
    if released.exists():
        gt_key = "gt_video"                                   # MiniMax H3 ground truth
        out = {}
        for x in json.load(open(released)):
            if not x.get(gt_key):
                continue
            out[int(x["id"])] = {"id": int(x["id"]), "brief": x["prompt"], "gt_video": x[gt_key],
                                 "gt_facts": x.get("gt_facts") or [], "meta": x.get("meta") or {},
                                 "official_images": x.get("official_images") or [],
                                 **({"text_required": bool(x["text_required"])} if x.get("text_required") is not None else {})}
        return out
    raise SystemExit(f"{released} not found: download AdProducer/AdCreation-Bench")


def read_jsonl(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {int(r["id"]): r for r in rows}


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ids_of(labels: dict, a) -> list[int]:
    ids = sorted(labels)
    return ids[:a.limit] if a.limit else ids


# ---------------------------------------------------------------------------------------------------------------
# gen-gt
# ---------------------------------------------------------------------------------------------------------------

async def cli_gen_gt(a):
    bench = Path(a.bench_dir)
    items = json.load(open(bench / "GT-Video" / "gen_prompt.json"))
    if a.limit:
        items = items[:a.limit]
    out = bench / "gt_video"
    out.mkdir(exist_ok=True)
    h3 = adp.H3Client()

    async def one(x):
        mp4 = out / f"answer_{x['id']}.mp4"
        if mp4.exists():
            return
        try:
            await h3.generate(x["prompt"], [bench / "GT-Video" / p for p in x["images"]], mp4)
            print(f"[gt] {x['id']} ok", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[gt] {x['id']} FAILED: {e}", flush=True)

    await asyncio.gather(*(one(x) for x in items))
    print(f"gt_video: {len(list(out.glob('answer_*.mp4')))} / {len(items)}")


async def cli_text_required(a):
    """Same as the RL `text-required` step (prompts/text_required.txt, 25 briefs per call); resumable."""
    bench = Path(a.bench_dir)
    path = bench / "text_required.json"
    done = json.load(open(path)) if path.exists() else {}
    labels = load_labels(bench)
    todo = [(i, labels[i]["brief"]) for i in ids_of(labels, a) if str(i) not in done]
    judge = adp.Judge()
    batches = [todo[k:k + 25] for k in range(0, len(todo), 25)]
    for r in await asyncio.gather(*(adp.decide_text_required(judge, b) for b in batches)):
        done.update(r)
    path.write_text(json.dumps(done, ensure_ascii=False, indent=1))
    print(f"{len(done)} decided, text required {sum(v['text_required'] for v in done.values())}; "
          f"{sum(str(i) not in done for i, _ in todo)} missing -> rerun; judge usage {judge.usage}")


# ---------------------------------------------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------------------------------------------

def _to_openai(msg: dict) -> dict:
    """AdProducerEnv.format_observation (PIL images) -> OpenAI content parts with JPEG data URLs."""
    parts = []
    for p in msg["content"]:
        if p["type"] == "text":
            parts.append({"type": "text", "text": p["text"]})
        else:
            buf = io.BytesIO()
            p["image"].save(buf, format="JPEG", quality=95)
            parts.append({"type": "image_url",
                          "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}})
    return {"role": msg["role"], "content": parts}


def _chat(a, messages: list[dict]) -> tuple[str, str]:
    """One assistant turn -> (text, finish_reason); finish_reason 'context' if the request no longer fits."""
    rq = adp._requests()
    body = {"model": a.model, "messages": messages, "temperature": a.temperature, "top_p": a.top_p,
            "max_tokens": adp.GEN_MAX_NEW_TOKENS_PER_TURN}
    last = None
    for attempt in range(5):
        try:
            r = rq.post(a.base_url.rstrip("/") + "/chat/completions", json=body, timeout=1800,
                        headers={"Authorization": f"Bearer {a.api_key}"})
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 400 and "context" in r.text.lower():
            return "", "context"
        if r.status_code != 200:
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            time.sleep(2 ** attempt)
            continue
        c = r.json()["choices"][0]
        return c["message"].get("content") or "", c.get("finish_reason") or ""
    raise RuntimeError(f"policy endpoint failed: {last}")


async def infer_one(a, label: dict, system: str) -> dict:
    env = adp.AdProducerEnv(label["id"], label, adp.MAX_TOOL_CALLS + 1)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": label["brief"]}]
    termination = None
    for turn in range(env.max_turns):
        text, finish = await asyncio.to_thread(_chat, a, messages)
        if finish in ("length", "context"):               # truncated: failure (skipped), as in RL
            termination = "truncated"
            break
        messages.append({"role": "assistant", "content": text})
        obs, done, _ = await env.astep(text)
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
    return {"id": label["id"], "termination": termination, "prediction": env.prediction,
            "tool_calls": env.tool_calls, "image_searches": env.image_searches,
            "malformed_turns": env.malformed_turns, "run_dir": str(env.run_dir)}


async def cli_infer(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = load_labels(Path(a.bench_dir))
    pred_path = out / "predictions.jsonl"
    done = read_jsonl(pred_path)
    todo = [i for i in ids_of(labels, a) if i not in done]
    if a.baseline == "brief":                              # Gen-Searcher generator baseline: the prompt as is
        for i in todo:
            append_jsonl(pred_path, {"id": i, "termination": TERMINATION_OK, "baseline": "brief",
                                     "prediction": {"gen_prompt": labels[i]["brief"], "reference_images": []}})
        print(f"wrote {len(todo)} brief-baseline rows -> {pred_path}")
        return
    if not a.base_url or not a.model:
        raise SystemExit("--base-url and --model are required (the endpoint serving the model under test)")
    system = adp.load_system_prompt(a.date)
    sem = asyncio.Semaphore(a.parallel)

    async def one(i):
        async with sem:
            try:
                row = await infer_one(a, labels[i], system)
            except Exception as e:  # noqa: BLE001
                row = {"id": i, "termination": f"error: {e}", "prediction": None}
            append_jsonl(pred_path, row)
            print(f"[infer] {i} {row['termination']} tools={row.get('tool_calls')}", flush=True)

    await asyncio.gather(*(one(i) for i in todo))


# ---------------------------------------------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------------------------------------------

class H3TextToVideo:
    """Baseline only: H3 text-to-video, task "t2va" with no conditions (MiniMax-H3 README,
    scripts/readme/reproducible-768p-t2va-request.sh).  It is served by a separate H3 server started with
    `--model-variant fl2va` (Ref2VA servers do not take it): H3_T2V_URLS.  Same /v1/videos API, target, sampling
    settings and seed as adp.H3Client, which stays Ref2VA-only."""

    def __init__(self):
        self.task = os.environ.get("H3_T2V_TASK", "t2va").strip() or "t2va"
        self.urls = [u for u in os.environ.get("H3_T2V_URLS", "").split(",") if u.strip()]
        if not self.urls:
            raise SystemExit("brief baseline: set H3_T2V_URLS to H3 server(s) started with --model-variant fl2va")
        self._sems = {u: asyncio.Semaphore(adp.H3_CONCURRENCY_PER_SERVER) for u in self.urls}
        self._i = 0

    async def generate(self, prompt: str, out_mp4: Path) -> None:
        import aiohttp
        req = {"task": self.task, "prompt": prompt, "conditions": [],
               "target": {"short_edge": adp.H3_SHORT_EDGE, "aspect_ratio": "16:9",
                          "duration_seconds": adp.VIDEO_DURATION_S},
               # same sampling settings as adp.H3Client (= the GT videos)
               "seconds": adp.VIDEO_DURATION_S, "num_outputs_per_prompt": 1, "num_inference_steps": adp.H3_STEPS,
               "flow_shift": adp.H3_FLOW_SHIFT, "audio_flow_shift": adp.H3_AUDIO_FLOW_SHIFT, "seed": adp.H3_SEED}
        last = None
        for attempt in range(adp.H3_RETRIES):
            base = self.urls[self._i % len(self.urls)]
            self._i += 1
            async with self._sems[base]:
                try:
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=adp.H3_TIMEOUT_S)) as s:
                        async with s.post(f"{base}/v1/videos", json=req) as r:
                            r.raise_for_status()
                            vid = (await r.json())["id"]
                        t0 = time.time()
                        while True:
                            async with s.get(f"{base}/v1/videos/{vid}") as r:
                                st = (await r.json()).get("status", "")
                            if st == "completed":
                                break
                            if st in ("failed", "cancelled", "error") or time.time() - t0 > adp.H3_TIMEOUT_S:
                                raise RuntimeError(f"H3 status {st or 'timeout'}")
                            await asyncio.sleep(5)
                        raw = out_mp4.with_suffix(".raw.mp4")
                        async with s.get(f"{base}/v1/videos/{vid}/content") as r:
                            r.raise_for_status()
                            raw.write_bytes(await r.read())
                    await asyncio.to_thread(adp.strip_audio, raw, out_mp4)
                    raw.unlink(missing_ok=True)
                    if out_mp4.exists() and out_mp4.stat().st_size > 0:
                        return
                    raise RuntimeError("empty video")
                except Exception as e:  # noqa: BLE001
                    last = e
                    await asyncio.sleep(2 ** attempt)
        raise RuntimeError(f"H3 text-to-video failed after {adp.H3_RETRIES} attempts: {last}")


def usable(row: dict) -> tuple[bool, str]:
    """Success = an answer that H3 can take.  The training-only P3 rule (must call image_search) is not applied."""
    if row.get("termination") != TERMINATION_OK or not row.get("prediction"):
        return False, f"termination={row.get('termination')}"
    if row.get("baseline") == "brief":
        return True, ""
    errs = [e for e in adp.gate_errors(row["prediction"], Path(row["run_dir"]), None) if not e.startswith("P3")]
    return (not errs), "; ".join(errs)


async def cli_generate(a):
    out = Path(a.out_dir)
    vids = out / "videos"
    vids.mkdir(parents=True, exist_ok=True)
    preds = read_jsonl(out / "predictions.jsonl")
    brief = any(r.get("baseline") == "brief" for r in preds.values())
    h3_t2v = H3TextToVideo() if brief else None
    h3 = None if brief else adp.H3Client()

    async def one(row):
        i = row["id"]
        mp4 = vids / f"{i}.mp4"
        fail = vids / f"{i}.failed.txt"
        if mp4.exists() or fail.exists():
            return
        ok, why = usable(row)
        if not ok:
            fail.write_text(f"not generated: {why}\n")
            return
        try:
            if row.get("baseline") == "brief":
                await h3_t2v.generate(row["prediction"]["gen_prompt"], mp4)
            else:
                run_dir = Path(row["run_dir"])
                refs = [run_dir / r["local_path"] for r in row["prediction"]["reference_images"] if r.get("local_path")]
                await h3.generate(row["prediction"]["gen_prompt"], refs, mp4)
            print(f"[gen] {i} ok", flush=True)
        except Exception as e:  # noqa: BLE001
            fail.write_text(f"h3_failed: {e}\n")
            print(f"[gen] {i} FAILED: {e}", flush=True)

    await asyncio.gather(*(one(r) for r in preds.values()))
    print(f"videos: {len(list(vids.glob('*.mp4')))} / {len(preds)} predictions")


# ---------------------------------------------------------------------------------------------------------------
# score / summary
# ---------------------------------------------------------------------------------------------------------------

PG_KEYS = ("pg_score", "visual_correctness", "text_accuracy", "selling_point", "shot_design", "aesthetics")


async def cli_score(a):
    """PG-Score = adp.video_reward (the RL R_video): 0.4 visual_correctness + 0.2 text_accuracy + 0.2 selling_point
    (five levels) + 0.1 shot_design + 0.1 aesthetics (three levels); without required text (text_required.json) the
    0.2 is spread pro rata.  Inputs: brief + facts + text requirement + <= 3 official images + GT video + video."""
    out = Path(a.out_dir)
    bench = Path(a.bench_dir)
    labels = load_labels(bench)
    preds = read_jsonl(out / "predictions.jsonl")
    if not preds:   # your own videos (<out>/videos/<id>.mp4) without agent predictions: no planned shot times
        preds = {int(p.stem): {"id": int(p.stem), "prediction": {"gen_prompt": ""}}
                 for p in sorted((out / "videos").glob("*.mp4")) if p.stem.isdigit() and int(p.stem) in labels}
    score_path = out / "scores.jsonl"
    done = read_jsonl(score_path)
    judge = adp.Judge()

    async def one(row):
        i = row["id"]
        mp4 = out / "videos" / f"{i}.mp4"
        if i in done or not mp4.exists():
            return
        gt = bench / labels[i]["gt_video"]
        if not gt.exists():                                # the judge's aesthetics bar is the GT video
            print(f"[score] {i} skipped: no GT video {gt}", flush=True)
            return
        v = await adp.video_reward(judge, labels[i], row["prediction"], mp4, gt, out / "frames" / str(i))
        if v is not None:
            v["pg_score"] = v["video_score"]
        append_jsonl(score_path, {"id": i, "video_reward": v})   # None = judge failed (skipped in summary)
        print(f"[score] {i} {None if v is None else v['pg_score']}", flush=True)

    await asyncio.gather(*(one(r) for r in preds.values()))
    print(f"judge usage: {judge.usage}")


def _summarise(labels: dict, scores: dict, keys: tuple) -> dict:
    def mean(ids):
        rows = [scores[i] for i in ids if i in scores]
        res = {"n": len(rows)}
        for k in keys:   # R_video text_accuracy is null when text is not required: mean over the items that have it
            xs = [float(r[k]) for r in rows if r.get(k) is not None]
            res[k] = round(sum(xs) / len(xs), 4) if xs else None
        return res

    summary = {"overall": mean(labels), "total_items": len(labels), "by_freshness": {}, "by_category": {}}
    for field, dst in (("freshness", "by_freshness"), ("category", "by_category")):
        groups: dict[str, list[int]] = {}
        for i, lab in labels.items():
            groups.setdefault(lab["meta"].get(field, "?"), []).append(i)
        summary[dst] = {g: mean(ids) for g, ids in sorted(groups.items())}
    return summary


def cli_summary(a):
    out = Path(a.out_dir)
    labels = load_labels(Path(a.bench_dir))
    scores = {i: r["video_reward"] for i, r in read_jsonl(out / "scores.jsonl").items() if r.get("video_reward")}
    s = _summarise(labels, scores, PG_KEYS)
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1))
    o = s["overall"]
    print(f"PG-Score: scored {o['n']} / {s['total_items']} (failures skipped; text_accuracy over text-required items)")
    print(f"{'group':<22}{'n':>5}" + "".join(f"{k[:12]:>14}" for k in PG_KEYS))
    rows = [("overall", o)] + list(s["by_freshness"].items()) + list(s["by_category"].items())
    for g, r in rows:
        print(f"{g:<22}{r['n']:>5}" + "".join(f"{r[k]:>14}" if r[k] is not None else f"{'-':>14}" for k in PG_KEYS))


# ---------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, out=True):
        p.add_argument("--bench-dir", required=True)
        if out:
            p.add_argument("--out-dir", required=True)
        p.add_argument("--limit", type=int, default=0, help="first N items only")
        return p

    p = common(sub.add_parser("gen-gt"), out=False); p.set_defaults(fn=cli_gen_gt)
    p = common(sub.add_parser("text-required"), out=False); p.set_defaults(fn=cli_text_required)
    p = common(sub.add_parser("infer"))
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint of the model under test")
    p.add_argument("--model", default=None)
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--baseline", choices=["brief"], default=None, help="no agent: the brief is the prompt (T2V)")
    p.add_argument("--temperature", type=float, default=0.6)    # Gen-Searcher run_gen_image_eval.sh
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--parallel", type=int, default=5)
    p.add_argument("--date", default=BENCH_DATE)
    p.set_defaults(fn=cli_infer)
    p = common(sub.add_parser("generate")); p.set_defaults(fn=cli_generate)
    p = common(sub.add_parser("score")); p.set_defaults(fn=cli_score)
    p = common(sub.add_parser("summary")); p.set_defaults(fn=cli_summary)
    a = ap.parse_args()

    _import_vsr(Path(a.bench_dir), Path(a.out_dir) if getattr(a, "out_dir", None) else None)
    res = a.fn(a)
    if asyncio.iscoroutine(res):
        asyncio.run(res)


if __name__ == "__main__":
    main()
