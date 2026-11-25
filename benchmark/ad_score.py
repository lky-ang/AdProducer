"""AD-Score: judges each generated video as an advertisement, without any ground truth (complements PG-Score).
Eight 1-5 dimensions in three groups + a 1-10 holistic score (prompts/ad_judge.txt), evidence before each score:
  A brief & product : brief_fidelity, product_presentation
  B craft           : visual_quality, narrative_flow
  C effectiveness   : hook, creative_quality, persuasiveness, purchase_intent
  overall_ad_quality (1-10, holistic); ad_mean = mean of the eight dimensions.
Inputs: brief + verified facts, the first 3 official images (<= 1024 px), and the ad as frames (2 fps, <= 1280 px,
labelled "t = 3.5 s, shot 2"; shots from PySceneDetect ContentDetector, threshold 27). Judge: an OpenAI-compatible
chat endpoint with image input and JSON output (we use GPT-6-sol, reasoning_effort medium, seed 0).

    export JUDGE_BASE_URL=... JUDGE_API_KEY=... [JUDGE_AUTH_HEADER=Authorization]
    python benchmark/ad_score.py --bench-dir data/AdCreation-Bench --videos runs/adproducer/videos \
        --out runs/adproducer/ad_score.jsonl [--judge gpt-6-sol] [--concurrency 8]
Resumable: ids already in --out (without "error") are skipped. Failed calls are retried, then recorded as errors."""
import argparse, base64, json, os, subprocess, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

HERE = Path(__file__).resolve().parent
SYSTEM = (HERE / "prompts" / "ad_judge.txt").read_text().strip()
DIMS = ["brief_fidelity", "product_presentation", "visual_quality", "narrative_flow", "hook", "creative_quality",
        "persuasiveness", "purchase_intent"]
lock = threading.Lock()


def ff(args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *args], check=True)


def cut_times(video: Path) -> list[float]:
    from scenedetect import ContentDetector, detect
    return [s[0].get_seconds() for s in detect(str(video), ContentDetector(threshold=27.0))][1:]


def frames(video: Path, d: Path, fps=2, width=1280):
    d.mkdir(parents=True, exist_ok=True)
    ff(["-i", str(video), "-vf", f"fps={fps},scale='min({width},iw)':-2", "-q:v", "3", str(d / "f_%03d.jpg")])
    return [(k / fps, p) for k, p in enumerate(sorted(d.glob("f_*.jpg")))]


def jpg(p: Path, d: Path, maxside=1024) -> Path:
    o = d / (p.stem + ".jpg")
    ff(["-i", str(p), "-vf", f"scale='if(gt(iw,ih),min({maxside},iw),-2)':'if(gt(iw,ih),-2,min({maxside},ih))'",
        "-frames:v", "1", "-q:v", "3", str(o)])
    return o


def img(p: Path) -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode(),
                                               "detail": "high"}}


def facts(x) -> str:
    return "\n".join(f"- {f['slot']}: {f['answer']}" for f in x.get("gt_facts") or []) or "(none)"


def build(x, video: Path, bench: Path, tmp: Path):
    c = [{"type": "text", "text": f"AD BRIEF:\n{x['prompt']}\n\nVerified facts about the product:\n{facts(x)}\n"}]
    for k, p in enumerate(x["official_images"][:3], 1):
        c += [{"type": "text", "text": f"OFFICIAL IMAGE {k} of the product:"}, img(jpg(bench / p, tmp))]
    cuts = cut_times(video)
    gen = frames(video, tmp / "frames")
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
                               capture_output=True, text=True).stdout.strip() or 0)
    c.append({"type": "text", "text": f"THE AD (frames to judge): duration {dur:.1f} s, {len(cuts) + 1} shot(s) detected, "
                                      f"{len(gen)} frames at 2 per second:"})
    for t, p in gen:
        c += [{"type": "text", "text": f"t = {t:.1f} s, shot {1 + sum(t >= ct for ct in cuts)}"}, img(p)]
    c.append({"type": "text", "text": "Output the single JSON object now."})
    return c, {"n_frames": len(gen), "n_shots": len(cuts) + 1, "duration": round(dur, 2)}


def call(content, judge, effort):
    base = os.environ["JUDGE_BASE_URL"].rstrip("/"); key = os.environ["JUDGE_API_KEY"]
    hdr = os.environ.get("JUDGE_AUTH_HEADER", "Authorization")
    headers = {hdr: f"Bearer {key}" if hdr.lower() == "authorization" else key, "Content-Type": "application/json"}
    body = {"model": judge, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}],
            "response_format": {"type": "json_object"}, "reasoning_effort": effort, "max_completion_tokens": 8000, "seed": 0}
    last = None
    for a in range(4):
        try:
            r = requests.post(f"{base}/chat/completions", json=body, timeout=600, headers=headers)
            if r.status_code == 200:
                d = r.json()
                return json.loads(d["choices"][0]["message"]["content"]), d.get("usage") or {}
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if "content_policy" in r.text or "content safety" in r.text:
                break
        except Exception as e:  # noqa: BLE001
            last = repr(e)[:200]
        time.sleep(10 * 2 ** a)
    raise RuntimeError(last)


def score_one(x, video, a, bench, out):
    try:
        with tempfile.TemporaryDirectory() as tmp:
            content, info = build(x, video, bench, Path(tmp))
        o, usage = call(content, a.judge, a.effort)
        s = {k: min(5, max(1, int(round(float(o[k]))))) for k in DIMS}
        row = {"id": x["id"], **s, "overall_ad_quality": min(10, max(1, int(round(float(o["overall_ad_quality"]))))),
               "ad_mean": round(sum(s.values()) / len(DIMS), 4),
               **{k: o.get(k) for k in ("requirements", "shot_walkthrough", "consumer_reaction", "overall_reason")},
               "reasons": {k: o.get(k + "_reason") for k in DIMS if k != "purchase_intent"}, **info,
               "judge": a.judge, "reasoning_effort": a.effort,
               "usage": {k: usage.get(k) for k in ("prompt_tokens", "completion_tokens")}}
    except Exception as e:  # noqa: BLE001
        row = {"id": x["id"], "error": repr(e)[:300], "judge": a.judge}
    with lock:
        with open(out, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(x["id"], row.get("overall_ad_quality", row.get("error")), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench-dir", required=True); ap.add_argument("--videos", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default=os.environ.get("JUDGE_MODEL", "gpt-6-sol")); ap.add_argument("--effort", default="medium")
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()
    bench = Path(a.bench_dir); out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    items = {int(x["id"]): x for x in json.load(open(bench / "AdCreation-Bench.json"))}
    done = {json.loads(l)["id"] for l in open(out) if l.strip() and "error" not in json.loads(l)} if out.exists() else set()
    todo = [(items[int(p.stem)], p) for p in sorted(Path(a.videos).glob("*.mp4"))
            if p.stem.isdigit() and int(p.stem) in items and int(p.stem) not in done]
    print(len(todo), "videos to judge", flush=True)
    with ThreadPoolExecutor(a.concurrency) as ex:
        for x, p in todo:
            ex.submit(score_one, x, p, a, bench, out)
    rows = [json.loads(l) for l in open(out) if l.strip()]
    ok = {r["id"]: r for r in rows if "error" not in r}
    if ok:
        print(f"AD-Score over {len(ok)} videos: ad_mean {sum(r['ad_mean'] for r in ok.values()) / len(ok):.3f}, "
              f"overall {sum(r['overall_ad_quality'] for r in ok.values()) / len(ok):.2f}")


if __name__ == "__main__":
    main()
